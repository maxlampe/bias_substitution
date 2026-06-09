#!/usr/bin/env bash
set -euo pipefail

# Run eval_biases.py across all 12 cells + base policy.
# Same eval seed across cells so MMLU/TriviaQA subsamples and Sharma sampling
# are identical, making cross-cell comparisons apples-to-apples.

export HF_HUB_ENABLE_HF_TRANSFER=0

# Default sweep includes the three-point regime curve: lambda in {0, 4, 8}.
LAMBDAS=${LAMBDAS:-"0.0 4.0 8.0"}
SEEDS=${SEEDS:-"0 1 2 3"}
EVAL_SEED=${EVAL_SEED:-12345}

# Paper-tier subsample sizes. Override with env vars for quick runs.
MMLU_N=${MMLU_N:-1000}
SHARMA_N=${SHARMA_N:-500}             # 0 = use all (~4888) records
SHARMA_SAMPLES=${SHARMA_SAMPLES:-2}
SHARMA_TEMPERATURE=${SHARMA_TEMPERATURE:-1.0}
TIAN_N=${TIAN_N:-1000}

# Tian/TriviaQA: optional LLM-judge grading runs IN PARALLEL with the
# string-match grading. Both are reported. Set TIAN_JUDGE=none to disable
# the judge. Requires ANTHROPIC_API_KEY when anthropic.
TIAN_JUDGE=${TIAN_JUDGE:-anthropic}
JUDGE_MODEL=${JUDGE_MODEL:-claude-haiku-4-5-20251001}
JUDGE_WORKERS=${JUDGE_WORKERS:-8}

# Pin every cell to the same training-step checkpoint. Set empty to use the
# final adapter (whatever step training ended at).
CHECKPOINT_STEP=${CHECKPOINT_STEP:-600}

if [[ "$TIAN_JUDGE" == "anthropic" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: TIAN_JUDGE=anthropic but ANTHROPIC_API_KEY is not set."
    echo "Either:  export ANTHROPIC_API_KEY=...    (recommended)"
    echo "Or:      TIAN_JUDGE=none ./eval_biases_all.sh"
    exit 1
fi

OUT_DIR=${OUT_DIR:-evals}
mkdir -p "$OUT_DIR"

# Base policy first (anchor for all deltas).
BASE_OUT="${OUT_DIR}/base.json"
if [[ ! -f "$BASE_OUT" ]]; then
    echo "=== Evaluating base policy (no adapter) ==="
    uv run eval_biases.py \
        --json_out "$BASE_OUT" \
        --seed "$EVAL_SEED" \
        --mmlu_n "$MMLU_N" \
        --sharma_n "$SHARMA_N" \
        --sharma_samples "$SHARMA_SAMPLES" \
        --sharma_temperature "$SHARMA_TEMPERATURE" \
        --tian_n "$TIAN_N" \
        --tian_judge "$TIAN_JUDGE" \
        --judge_model "$JUDGE_MODEL" \
        --judge_workers "$JUDGE_WORKERS" \
        --store_per_sample
else
    echo "=== Skipping base (already evaluated) ==="
fi

for lam in $LAMBDAS; do
    for seed in $SEEDS; do
        if [[ -n "$CHECKPOINT_STEP" ]]; then
            ADAPTER="runs/lam_${lam}_seed_${seed}/checkpoint-${CHECKPOINT_STEP}"
        else
            ADAPTER="runs/lam_${lam}_seed_${seed}/final"
        fi
        OUT="${OUT_DIR}/lam_${lam}_seed_${seed}.json"
        echo "=== Evaluating lam=${lam} seed=${seed}  adapter=${ADAPTER} ==="
        if [[ ! -d "$ADAPTER" ]]; then
            echo "  [skip] $ADAPTER does not exist"
            continue
        fi
        if [[ -f "$OUT" ]]; then
            echo "  [skip] $OUT already exists"
            continue
        fi
        uv run eval_biases.py \
            --adapter_dir "$ADAPTER" \
            --json_out "$OUT" \
            --seed "$EVAL_SEED" \
            --mmlu_n "$MMLU_N" \
            --sharma_n "$SHARMA_N" \
            --sharma_samples "$SHARMA_SAMPLES" \
            --sharma_temperature "$SHARMA_TEMPERATURE" \
            --tian_n "$TIAN_N" \
            --tian_judge "$TIAN_JUDGE" \
            --judge_model "$JUDGE_MODEL" \
            --judge_workers "$JUDGE_WORKERS" \
            --store_per_sample
    done
done

echo ""
echo "=== Cross-cell aggregation ==="
LAMBDAS="$LAMBDAS" SEEDS="$SEEDS" OUT_DIR="$OUT_DIR" uv run python - <<'PY'
import json, os, statistics

lambdas = os.environ.get("LAMBDAS", "0.0 4.0 8.0").split()
seeds   = os.environ.get("SEEDS", "0 1 2 3").split()
out_dir = os.environ.get("OUT_DIR", "evals")

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def metric(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None or k not in cur:
            return default
        cur = cur[k]
    return cur

def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and (x != x))]
    if not v:
        return float("nan"), float("nan")
    if len(v) > 1:
        return statistics.mean(v), statistics.stdev(v)
    return v[0], 0.0

base = load(os.path.join(out_dir, "base.json"))

print()
print(f"{'lam':>5s} {'n':>3s} | "
      f"{'MMLU':>14s} | "
      f"{'syc_regr':>14s} | "
      f"{'syc_all':>14s} | "
      f"{'ECE_sm':>14s} | "
      f"{'ECE_jd':>14s} | "
      f"{'Brier_sm':>14s} | "
      f"{'AUROC_sm':>14s} | "
      f"{'TriviaAcc_sm':>16s} | "
      f"{'TriviaAcc_jd':>16s}")
print("-" * 180)

def fmt1(v):
    if v is None or (isinstance(v, float) and v != v):
        return "    -    "
    return f"{v:>7.3f}        "

if base is not None:
    print(f"{'base':>5s} {1:>3d} | "
          f"{fmt1(metric(base,'mmlu','accuracy')):>14s} | "
          f"{fmt1(metric(base,'sycophancy','regressive_flip_rate')):>14s} | "
          f"{fmt1(metric(base,'sycophancy','overall_flip_rate')):>14s} | "
          f"{fmt1(metric(base,'calibration','ece')):>14s} | "
          f"{fmt1(metric(base,'calibration','ece_judge')):>14s} | "
          f"{fmt1(metric(base,'calibration','brier')):>14s} | "
          f"{fmt1(metric(base,'calibration','auroc')):>14s} | "
          f"{fmt1(metric(base,'calibration','accuracy')):>16s} | "
          f"{fmt1(metric(base,'calibration','accuracy_judge')):>16s}")

for lam in lambdas:
    rows = []
    for seed in seeds:
        rows.append(load(os.path.join(out_dir, f"lam_{lam}_seed_{seed}.json")))
    rows = [r for r in rows if r is not None]
    if not rows:
        print(f"{lam:>5s} {0:>3d}    (no data)")
        continue
    mmlu     = [metric(r,'mmlu','accuracy') for r in rows]
    sycregr  = [metric(r,'sycophancy','regressive_flip_rate') for r in rows]
    sycall   = [metric(r,'sycophancy','overall_flip_rate') for r in rows]
    ece_sm   = [metric(r,'calibration','ece') for r in rows]
    ece_jd   = [metric(r,'calibration','ece_judge') for r in rows]
    brier_sm = [metric(r,'calibration','brier') for r in rows]
    au_sm    = [metric(r,'calibration','auroc') for r in rows]
    tacc_sm  = [metric(r,'calibration','accuracy') for r in rows]
    tacc_jd  = [metric(r,'calibration','accuracy_judge') for r in rows]
    def fmt(v):
        m,s = ms(v)
        if m != m:  # NaN
            return "      -       "
        return f"{m:>7.3f}±{s:<6.3f}"
    print(f"{lam:>5s} {len(rows):>3d} | "
          f"{fmt(mmlu):>14s} | "
          f"{fmt(sycregr):>14s} | "
          f"{fmt(sycall):>14s} | "
          f"{fmt(ece_sm):>14s} | "
          f"{fmt(ece_jd):>14s} | "
          f"{fmt(brier_sm):>14s} | "
          f"{fmt(au_sm):>14s} | "
          f"{fmt(tacc_sm):>16s} | "
          f"{fmt(tacc_jd):>16s}")

print()
print("Notes:")
print("  MMLU:        accuracy. Higher = better quality.")
print("  syc_regr:    Sharma regressive sycophancy: model was right, then changed (HEADLINE).")
print("  syc_all:     overall flip rate (any letter change, regardless of correctness).")
print("  ECE_sm:      Expected Calibration Error using string-match grading. Lower = better.")
print("  ECE_jd:      ECE using Tian Appendix C LLM-judge grading. Lower = better.")
print("  Brier_sm:    Brier score, string-match grading.")
print("  AUROC_sm:    confidence-vs-correctness AUROC, string-match grading.")
print("  TriviaAcc_sm/jd: TriviaQA accuracy under each grading.")
PY