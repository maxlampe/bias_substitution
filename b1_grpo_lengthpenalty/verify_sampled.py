"""
Sampled validation of a trained LoRA adapter, matched to training conditions.

Differs from verify.py in three ways that matter for measuring length effects:
  1. do_sample=True at training temperature (default 1.0) instead of greedy.
     Greedy decoding measures argmax trajectories; the length penalty shifts
     the *sampling distribution*, so greedy can miss most of the effect.
  2. Prompts come from a held-out split of the same dataset used in training
     (UltraFeedback test_prefs) instead of 5 hand-written short prompts. This
     matches the distribution over which the length-vs-quality tradeoff was
     learned.
  3. Multiple samples per prompt (default 4), averaged within prompt before
     averaging across prompts. Reduces sampling noise.

Usage:
    HF_HUB_ENABLE_HF_TRANSFER=0 uv run verify_sampled.py \\
        --adapter_dir runs/lam_8.0_seed_0/final \\
        --num_prompts 50 --samples_per_prompt 4 --temperature 1.0
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DEFAULT_POLICY = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_REWARD = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
DEFAULT_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
DEFAULT_HELDOUT_SPLIT = "test_prefs"


def load_heldout_prompts(dataset_name, split, num_prompts, seed,
                         max_prompt_chars=2000):
    """Load held-out UltraFeedback prompts. Shuffled deterministically by seed."""
    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=seed).select(range(min(num_prompts, len(ds))))
    prompts = []
    for ex in ds:
        p = ex["prompt"]
        if len(p) > max_prompt_chars:
            p = p[:max_prompt_chars]
        prompts.append(p)
    return prompts


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, samples_per_prompt,
                   max_new_tokens, temperature, top_p, device):
    """
    Generate K samples for each prompt. Returns list of lists of
    (text, token_count) tuples, shape [n_prompts][samples_per_prompt].
    """
    results = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        # Replicate prompt K times to draw K independent samples.
        inputs = tokenizer(
            [text] * samples_per_prompt,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        per_prompt = []
        for j in range(samples_per_prompt):
            completion_ids = out[j, prompt_len:]
            # Strip trailing pad tokens for accurate length counting.
            mask = completion_ids != tokenizer.pad_token_id
            n_tokens = int(mask.sum().item())
            completion = tokenizer.decode(
                completion_ids[:n_tokens], skip_special_tokens=True,
            )
            per_prompt.append((completion, n_tokens))
        results.append(per_prompt)
    return results


@torch.no_grad()
def rm_score(rm, rm_tokenizer, prompt, completion, device, max_length=4096):
    conv = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    text = rm_tokenizer.apply_chat_template(conv, tokenize=False)
    if (rm_tokenizer.bos_token is not None
            and text.startswith(rm_tokenizer.bos_token)):
        text = text[len(rm_tokenizer.bos_token):]
    inputs = rm_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(device)
    return float(rm(**inputs).logits[0, 0].item())


def aggregate(per_prompt_values):
    """Mean across prompts of (mean within prompt). Returns mean, std across prompts."""
    import statistics
    prompt_means = [sum(v) / len(v) for v in per_prompt_values]
    if len(prompt_means) > 1:
        return statistics.mean(prompt_means), statistics.stdev(prompt_means)
    return prompt_means[0] if prompt_means else 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=DEFAULT_POLICY)
    parser.add_argument("--reward_model_name", type=str, default=DEFAULT_REWARD)
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--heldout_split", type=str, default=DEFAULT_HELDOUT_SPLIT)
    parser.add_argument("--num_prompts", type=int, default=50)
    parser.add_argument("--samples_per_prompt", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Match training max_completion_length.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Match training temperature (GRPOConfig default 1.0).")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345,
                        help="Seed for prompt shuffling and generation sampling. "
                             "Use the SAME seed across all runs you compare.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--json_out", type=str, default=None,
                        help="Optional path to dump per-prompt and summary stats as JSON.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    adapter_dir = Path(args.adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter_dir}."
        )

    print(f"Loading policy tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pad_candidate = "<|finetune_right_pad_id|>"
    if pad_candidate in tokenizer.get_vocab():
        tokenizer.pad_token = pad_candidate
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Generation with left padding so the prompt aligns to the right when
    # drawing multiple samples per prompt.
    tokenizer.padding_side = "left"
    print(f"  pad_token = {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    print(f"Loading base policy on {args.device}")
    base_policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
    )
    base_policy.eval()

    print(f"Loading LoRA adapter from {adapter_dir}")
    adapted_policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
    )
    adapted_policy = PeftModel.from_pretrained(adapted_policy, str(adapter_dir))
    adapted_policy.eval()

    print(f"Loading reward model {args.reward_model_name} on {args.device}")
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name,
        torch_dtype=torch.bfloat16,
        num_labels=1,
        device_map={"": args.device},
    )
    rm.eval()
    rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
    if rm_tokenizer.chat_template is None:
        src = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        rm_tokenizer.chat_template = src.chat_template

    print(f"Loading {args.num_prompts} held-out prompts "
          f"from {args.dataset_name}:{args.heldout_split}")
    prompts = load_heldout_prompts(
        args.dataset_name, args.heldout_split,
        args.num_prompts, args.seed,
    )
    print(f"  Loaded {len(prompts)} prompts.")

    print(f"\nGenerating {args.samples_per_prompt} samples/prompt "
          f"at temperature={args.temperature} ...")

    print("  [base] generating ...")
    base_outputs = generate_batch(
        base_policy, tokenizer, prompts,
        samples_per_prompt=args.samples_per_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        device=args.device,
    )

    print("  [adapted] generating ...")
    adapted_outputs = generate_batch(
        adapted_policy, tokenizer, prompts,
        samples_per_prompt=args.samples_per_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        device=args.device,
    )

    print("\nScoring with reward model ...")
    base_lens_per_prompt = []
    base_scores_per_prompt = []
    adapted_lens_per_prompt = []
    adapted_scores_per_prompt = []

    per_prompt_records = []
    for i, prompt in enumerate(prompts):
        b_scores = []
        b_lens = []
        a_scores = []
        a_lens = []
        for (c_b, n_b) in base_outputs[i]:
            s_b = rm_score(rm, rm_tokenizer, prompt, c_b, args.device)
            b_scores.append(s_b)
            b_lens.append(n_b)
        for (c_a, n_a) in adapted_outputs[i]:
            s_a = rm_score(rm, rm_tokenizer, prompt, c_a, args.device)
            a_scores.append(s_a)
            a_lens.append(n_a)
        base_lens_per_prompt.append(b_lens)
        base_scores_per_prompt.append(b_scores)
        adapted_lens_per_prompt.append(a_lens)
        adapted_scores_per_prompt.append(a_scores)
        per_prompt_records.append({
            "prompt": prompt[:200],
            "base_lens": b_lens, "base_scores": b_scores,
            "adapted_lens": a_lens, "adapted_scores": a_scores,
            "delta_len_mean": sum(a_lens) / len(a_lens) - sum(b_lens) / len(b_lens),
            "delta_score_mean": sum(a_scores) / len(a_scores) - sum(b_scores) / len(b_scores),
        })
        if (i + 1) % 10 == 0:
            print(f"  scored {i + 1}/{len(prompts)} prompts")

    base_len_mean, base_len_std = aggregate(base_lens_per_prompt)
    adapted_len_mean, adapted_len_std = aggregate(adapted_lens_per_prompt)
    base_score_mean, base_score_std = aggregate(base_scores_per_prompt)
    adapted_score_mean, adapted_score_std = aggregate(adapted_scores_per_prompt)

    delta_len = adapted_len_mean - base_len_mean
    delta_score = adapted_score_mean - base_score_mean
    pct_len_change = 100.0 * delta_len / base_len_mean if base_len_mean > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"Sampled validation summary  ({args.num_prompts} prompts x "
          f"{args.samples_per_prompt} samples, T={args.temperature})")
    print("=" * 70)
    print(f"  base    mean length = {base_len_mean:7.2f}  "
          f"(std across prompts = {base_len_std:.2f})")
    print(f"  adapted mean length = {adapted_len_mean:7.2f}  "
          f"(std across prompts = {adapted_len_std:.2f})")
    print(f"  delta length        = {delta_len:+7.2f}  "
          f"({pct_len_change:+.1f}%)")
    print()
    print(f"  base    mean RM score = {base_score_mean:+7.3f}  "
          f"(std across prompts = {base_score_std:.3f})")
    print(f"  adapted mean RM score = {adapted_score_mean:+7.3f}  "
          f"(std across prompts = {adapted_score_std:.3f})")
    print(f"  delta RM score        = {delta_score:+7.3f}")
    print("=" * 70)

    if args.json_out:
        summary = {
            "adapter_dir": str(adapter_dir),
            "num_prompts": args.num_prompts,
            "samples_per_prompt": args.samples_per_prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "base_len_mean": base_len_mean,
            "adapted_len_mean": adapted_len_mean,
            "delta_len": delta_len,
            "pct_len_change": pct_len_change,
            "base_score_mean": base_score_mean,
            "adapted_score_mean": adapted_score_mean,
            "delta_score": delta_score,
            "per_prompt": per_prompt_records,
        }
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
        print(f"Wrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()