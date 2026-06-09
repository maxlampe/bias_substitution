"""
Smoking-gun RLHF training: length penalty under GRPO with frozen Skywork RM.

Reward used during training:
    R'(x, y) = R_RM(x, y) - lam * (n_tokens(response) / 100)

This file is the single-GPU variant. Policy, reference (LoRA-disabled), and
frozen Skywork RM all live on cuda:0. The RM is accessed only through the
reward callable (never passed to the trainer), so trainer/accelerate cannot
move or wrap it.

GRPO is used as the headline algorithm. PPO is scientifically cleaner (no
loss-level length normalization) but TRL 0.15.2 PPOTrainer corrupts frozen
reward models during accelerate.prepare. The original GRPO loss has a small
within-sequence length-coupled normalization which we treat as a constant
offset across lambda values; relative substitution effects across lambda are
still detectable.

LoRA on all linear layers (attention + MLP), rank 32. Per the "LoRA Without
Regret" finding (Liu et al. 2024), all-linear LoRA roughly matches full
fine-tuning in RL, and the gap vs. attention-only LoRA is meaningful for RL
because of the noisier per-step gradient signal. MLP layers carry ~65% of
Llama-3's linear parameters; excluding them caps expressiveness in a way
that can mute the substitution effect we want to measure.

One run = one (lambda, seed) cell. Iterate this script externally for the
sweep grid. Each run writes LoRA-adapter checkpoints under
{output_root}/lam_{lam}_seed_{seed}/.

Hardware target: single A100 SXM (40 or 80 GB) or larger. For 80 GB the
defaults below fit comfortably. For 40 GB, drop --per_device_train_batch_size
and --num_generations to 2 each (preserving divisibility). On H200 (141 GB)
push per_device_train_batch_size to 16 and num_generations to 8 (set from
run_train.sh) for better utilization. Attention implementation is SDPA;
swap to flash_attention_2 only if flash-attn is installed in the env.

Three HuggingFace models are gated and need terms acceptance:
    meta-llama/Llama-3.2-3B-Instruct        (policy)
    meta-llama/Llama-3.1-8B-Instruct        (chat template fallback for RM)
    Skywork/Skywork-Reward-V2-Llama-3.1-8B  (reward model)
Run `huggingface-cli login` once before launching.

Quick smoke test (about 5 minutes on A100 80GB):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    HF_HUB_ENABLE_HF_TRANSFER=0 \\
    uv run train.py --lam 0.0 --seed 0 \\
        --max_steps 3 --num_train_samples 8 \\
        --per_device_train_batch_size 2 --num_generations 2 \\
        --gradient_accumulation_steps 1 \\
        --max_completion_length 128 \\
        --save_steps 3 --report_to none

Production cell (one lambda, one seed):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    HF_HUB_ENABLE_HF_TRANSFER=0 \\
    uv run train.py --lam 0.0 --seed 0
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import GRPOConfig, GRPOTrainer


DEFAULT_POLICY = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_REWARD = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
DEFAULT_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
DEFAULT_SPLIT = "train_prefs"
DEFAULT_RM_DEVICE = "cuda:0"

# LoRA target modules. All linear layers (attention + MLP) per Liu et al. 2024
# "LoRA Without Regret" and TRL's own recommendation for RL fine-tuning.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention (~35% of linear params)
    "gate_proj", "up_proj", "down_proj",      # MLP        (~65% of linear params)
]


def _report_attn_impl(name, model):
    impl = getattr(model.config, "_attn_implementation", "unknown")
    print(f"  [{name}] attn_implementation = {impl}")


def load_frozen_reward_model(model_name, device):
    """
    Load Skywork RM frozen on the specified device.

    SDPA (not FA2) for the RM: FA2 + output_hidden_states has known issues
    that produce zero scores in some transformers reward paths. SDPA is
    correct and only mildly slower for the per-call inference we do.
    """
    print(f"Loading reward model {model_name} on {device}")
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        num_labels=1,
        device_map={"": device},
    )
    rm.eval()
    for p in rm.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return rm, tokenizer


@torch.no_grad()
def sanity_check_rm(rm, rm_tokenizer, device):
    """Verify RM produces sane scores on known-good vs known-bad pair."""
    good = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    bad = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "asdkjfh banana whatever I have no idea."},
    ]

    def score_one(conv):
        text = rm_tokenizer.apply_chat_template(conv, tokenize=False)
        if (rm_tokenizer.bos_token is not None
                and text.startswith(rm_tokenizer.bos_token)):
            text = text[len(rm_tokenizer.bos_token):]
        ids = rm_tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=4096).to(device)
        out = rm(**ids)
        return float(out.logits[0, 0].item())

    s_good = score_one(good)
    s_bad = score_one(bad)
    print(f"  [RM sanity] good response score = {s_good:.4f}")
    print(f"  [RM sanity] bad response score  = {s_bad:.4f}")
    if abs(s_good) < 1e-6 and abs(s_bad) < 1e-6:
        print("  [RM sanity] WARNING: scores essentially zero. "
              "RM is not producing usable signal.")


def build_reward_fn(rm, rm_tokenizer, rm_device, lam, divisor=100.0,
                    rm_max_length=4096):
    """
    Callable reward function matching TRL 0.15.2 GRPOTrainer.reward_funcs API.

    Scores each (prompt, completion) pair with the frozen RM, subtracts a
    per-completion length penalty (lam * n_response_tokens / divisor),
    returns a list of floats (one per completion).
    """

    bos = rm_tokenizer.bos_token

    @torch.no_grad()
    def reward_fn(prompts, completions, **kwargs):
        completion_ids = kwargs.get("completion_ids", None)

        rewards = []
        for i, (prompt_item, completion_item) in enumerate(zip(prompts, completions)):
            if isinstance(prompt_item, list):
                conv = list(prompt_item) + list(completion_item)
                completion_text = (completion_item[-1]["content"]
                                   if completion_item else "")
            else:
                conv = [
                    {"role": "user", "content": prompt_item},
                    {"role": "assistant", "content": completion_item},
                ]
                completion_text = completion_item

            text = rm_tokenizer.apply_chat_template(conv, tokenize=False)
            if bos is not None and text.startswith(bos):
                text = text[len(bos):]
            inputs = rm_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=rm_max_length,
            ).to(rm_device)
            score = float(rm(**inputs).logits[0, 0].item())

            if completion_ids is not None:
                n_tokens = len(completion_ids[i])
            else:
                n_tokens = len(rm_tokenizer.encode(
                    completion_text, add_special_tokens=False))

            reward = score - lam * (n_tokens / divisor)
            rewards.append(float(reward))

        return rewards

    reward_fn.__name__ = "length_penalized_rm"
    return reward_fn


def build_dataset(dataset_name, split, seed, max_samples, max_prompt_chars=2000):
    """Load UltraFeedback prompts in GRPO conversational format."""
    print(f"Loading dataset {dataset_name} split {split}")
    ds = load_dataset(dataset_name, split=split)

    def to_conv(example):
        prompt = example["prompt"]
        if len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars]
        return {"prompt": [{"role": "user", "content": prompt}]}

    ds = ds.map(to_conv, remove_columns=ds.column_names)
    if max_samples > 0 and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    print(f"Dataset size after processing: {len(ds)}")
    return ds


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lam", type=float, required=True,
                        help="Length penalty coefficient")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", type=str, default=DEFAULT_POLICY)
    parser.add_argument("--reward_model_name", type=str, default=DEFAULT_REWARD)
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--dataset_split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--output_root", type=str, default="runs")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Total optimizer steps. 500 is a reasonable "
                             "default for length-penalty substitution "
                             "experiments. Bump higher if convergence is "
                             "incomplete; reduce to 300 if early signal "
                             "is already saturating.")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--num_train_samples", type=int, default=8000)
    # Divisibility constraint: per_device * num_processes * grad_accum must
    # be divisible by num_generations. 4 * 1 * 1 = 4, divisible by 4.
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL coefficient. 0.04 matches TRL/OpenRLHF "
                             "defaults for preference RLHF. Values below "
                             "0.01 risk policy collapse / reward hacking "
                             "at high λ, which would confound the "
                             "substitution measurement with a different "
                             "failure mode.")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank. 32 is a good default for RL "
                             "fine-tuning; 64 if compute allows.")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--rm_device", type=str, default=DEFAULT_RM_DEVICE,
                        help="Device for the frozen RM. On single-GPU, "
                             "leave at cuda:0; the RM coexists with the policy.")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="Attention implementation for the policy. "
                             "sdpa is the safe default and works on any GPU "
                             "without extra deps. flash_attention_2 requires "
                             "the flash-attn package and a Hopper/Ampere GPU.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint dir to resume from. "
                             "Pass 'auto' or a specific path. Requires that "
                             "the original run was saved with optimizer state "
                             "(save_only_model=False, the current default).")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise RuntimeError(f"Need at least 1 visible GPU, found {n_gpus}")
    print(f"Visible GPUs: {n_gpus}")

    output_dir = Path(args.output_root) / f"lam_{args.lam}_seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Policy tokenizer. Use Llama-3.x's dedicated finetune pad token (128004),
    # never the eos token. eos = <|eot_id|> is a content token and conflating
    # the two corrupts attention masks during training.
    policy_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pad_candidate = "<|finetune_right_pad_id|>"
    if pad_candidate in policy_tokenizer.get_vocab():
        policy_tokenizer.pad_token = pad_candidate
    elif policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    print(f"  [tokenizer] pad_token = {policy_tokenizer.pad_token!r} "
          f"(id={policy_tokenizer.pad_token_id})")

    # Frozen RM on cuda:0 (single-GPU).
    rm, rm_tokenizer = load_frozen_reward_model(args.reward_model_name, args.rm_device)
    _report_attn_impl("RM", rm)

    # transformers 4.46 doesn't auto-load chat_template.jinja files.
    # Skywork-V2 ships its template that way; patch from Llama-3.1 source.
    if rm_tokenizer.chat_template is None:
        template_src = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        rm_tokenizer.chat_template = template_src.chat_template

    sanity_check_rm(rm, rm_tokenizer, args.rm_device)

    reward_fn = build_reward_fn(
        rm, rm_tokenizer, args.rm_device,
        lam=args.lam, divisor=100.0,
    )

    train_dataset = build_dataset(
        args.dataset_name,
        args.dataset_split,
        args.seed,
        args.num_train_samples,
    )

    # LoRA on all linear layers (attention + MLP). See LORA_TARGET_MODULES note
    # at the top of the file for rationale.
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    # GRPOConfig kwargs supported in TRL 0.15.2. We do NOT use any v1-only
    # knobs (no loss_type, no scale_rewards, no mask_truncated_completions,
    # no log_completions, no num_completions_to_print).
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        beta=args.beta,
        bf16=True,
        # LoRA + gradient checkpointing breaks the autograd chain (frozen base
        # embeddings have requires_grad=False, blocking gradient flow back
        # through recomputed activations). On 80 GB VRAM we don't need the
        # memory savings anyway.
        gradient_checkpointing=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        # Save optimizer state too so we can resume training cleanly.
        # Without this, Adam moments reset on resume and cause a transient.
        # Cost: each checkpoint ~3x larger (~150 MB for LoRA at r=32).
        save_only_model=False,
        save_total_limit=20,
        logging_steps=10,
        max_steps=args.max_steps,
        seed=args.seed,
        report_to=args.report_to,
        # 8-bit Adam frees memory vs fp32 Adam (matters more at higher rank
        # and all-linear target).
        optim="adamw_bnb_8bit",
        # Attention implementation is controlled via CLI; defaults to sdpa
        # so the script runs on any modern GPU without flash-attn installed.
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            "attn_implementation": args.attn_implementation,
        },
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=policy_tokenizer,
    )

    # Confirm attention impl actually engaged on the policy.
    try:
        _report_attn_impl("policy", trainer.model)
    except Exception:
        pass

    print(
        f"Starting GRPO training:\n"
        f"  policy           = {args.model_name}\n"
        f"  reward_model     = {args.reward_model_name} (+ length penalty)\n"
        f"  rm_device        = {args.rm_device}\n"
        f"  lambda           = {args.lam}\n"
        f"  seed             = {args.seed}\n"
        f"  output_dir       = {output_dir}\n"
        f"  max_steps        = {args.max_steps}\n"
        f"  per_device_bs    = {args.per_device_train_batch_size}\n"
        f"  num_generations  = {args.num_generations}\n"
        f"  grad_accum       = {args.gradient_accumulation_steps}\n"
        f"  max_completion   = {args.max_completion_length}\n"
        f"  beta (KL)        = {args.beta}\n"
        f"  lora_r           = {args.lora_r} (all linear layers)\n"
        f"  attn_impl        = {args.attn_implementation}\n"
        f"  resume_from      = {args.resume_from_checkpoint}\n"
    )

    # Resume support. Pass --resume_from_checkpoint auto to pick up the
    # most recent checkpoint in output_dir, or pass an explicit path.
    resume = args.resume_from_checkpoint
    if resume == "auto":
        resume = True   # HF Trainer convention: True means "latest in output_dir"
    if resume:
        print(f"Resuming from checkpoint: {resume}")
    trainer.train(resume_from_checkpoint=resume)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    print(f"Saved final LoRA adapter to {final_dir}")


if __name__ == "__main__":
    main()