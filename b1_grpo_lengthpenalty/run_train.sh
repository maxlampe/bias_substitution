#!/usr/bin/env bash
set -euo pipefail

# run_harder.sh — train aggressive-hyperparameter cells for the regime sweep.
#
# All cells use the same hyperparameters as the existing lambda=8 runs:
#   learning_rate 3e-5  (vs train.py default 1e-5)
#   beta          0.02  (vs train.py default 0.04)
#   max_steps     600   (vs train.py default 500)
#
# Default sweep produces a matched-hyperparameter regime contrast:
#   lambda 0.0  seeds 0..3  (NEW: replaces the old conservative-setting runs)
#   lambda 8.0  seeds 0..3  (only seed=3 is new; seeds 0..2 already exist)
#
# Optional: add lambda=4.0 to the sweep for a 3-point monotonicity curve:
#   LAMBDAS="0.0 4.0 8.0" ./run_harder.sh
#
# IMPORTANT before running:
#   The old conservative-setting lambda=0 adapters in runs/lam_0.0_seed_{0,1,2}
#   used lr=1e-5, beta=0.04 and must be moved out of the way first, otherwise
#   this script will skip those slots and you will be left with mixed
#   hyperparameters. Recommended move:
#
#     mkdir -p runs_v1_conservative
#     mv runs/lam_0.0_seed_* runs_v1_conservative/
#     # (optional, for tidiness:)
#     mv runs/lam_2.0_seed_* runs_v1_conservative/
#     mv runs/lam_4.0_seed_* runs_v1_conservative/
#
#   The existing aggressive lambda=8 seeds 0,1,2 in runs/lam_8.0_seed_{0,1,2}
#   stay where they are and will be auto-skipped.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_PROJECT="lengthbias-smoking-gun"

LR=${LR:-3e-5}
BETA=${BETA:-0.02}
MAX_STEPS=${MAX_STEPS:-600}

LAMBDAS=${LAMBDAS:-"0.0 4.0 8.0"}
SEEDS=${SEEDS:-"0 1 2 3"}

# Safety check: detect leftover conservative-setting adapters in the lambda=0
# slots and warn before training proceeds.
NEEDS_MOVE=0
for seed in 0 1 2; do
    OLD="runs/lam_0.0_seed_${seed}/final"
    if [[ -d "$OLD" ]]; then
        NEEDS_MOVE=1
    fi
done
if [[ "$NEEDS_MOVE" -eq 1 ]]; then
    echo ""
    echo "WARNING: existing runs/lam_0.0_seed_{0,1,2}/final adapter(s) detected."
    echo "If those are from the OLD conservative setting (lr=1e-5, beta=0.04)"
    echo "they will be SKIPPED by this script and you will end up with mixed"
    echo "hyperparameters across cells. Move them aside first:"
    echo ""
    echo "  mkdir -p runs_v1_conservative"
    echo "  mv runs/lam_0.0_seed_* runs_v1_conservative/"
    echo ""
    echo "Press Ctrl-C now to abort, or wait 15 seconds to continue anyway."
    sleep 15
fi

for lam in $LAMBDAS; do
    for seed in $SEEDS; do
        OUT_DIR="runs/lam_${lam}_seed_${seed}"
        if [[ -d "${OUT_DIR}/final" ]]; then
            echo "=== Skip lam=${lam} seed=${seed} (final adapter already at ${OUT_DIR}/final) ==="
            continue
        fi
        echo ""
        echo "=== Training lam=${lam} seed=${seed} ==="
        echo "    lr=${LR}, beta=${BETA}, max_steps=${MAX_STEPS}"
        uv run train.py \
            --lam "$lam" \
            --seed "$seed" \
            --max_steps "$MAX_STEPS" \
            --learning_rate "$LR" \
            --beta "$BETA" \
            --report_to wandb
        echo "=== Finished lam=${lam} seed=${seed} ==="
    done
done

echo ""
echo "All requested cells complete."
echo "Next: evaluate the new cells with"
echo "  SEEDS=\"0 1 2 3\" ./eval_biases_all.sh"