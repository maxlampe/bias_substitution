"""
Hierarchical bootstrap CIs from per-sample data dumped by eval_biases.py.

Requires the eval JSONs to contain per_sample arrays. Re-run eval_biases.py
with --store_per_sample to generate them (the older JSONs from analyze_evals
runs only have aggregates and are not usable here).

Method:
  For each (lambda, metric), the CI is built by hierarchical bootstrap:
    1. resample n_seeds seeds with replacement from the pool of trained seeds
    2. for each chosen seed, resample its per-sample data with replacement
    3. pool, compute the metric on the pooled samples
    4. repeat B times
    5. take the 2.5/97.5 percentiles
  This propagates BOTH cross-seed variance (different training trajectories)
  and within-seed sampling variance (binomial / categorical noise on the
  fixed-size eval), without making normality assumptions.

For deltas (e.g., lambda=8 vs lambda=0): in each bootstrap iteration, build
both pooled samples independently, compute the metric on each, subtract.
Percentile CI on the differences.

Default B=10000 takes ~30s for a typical sweep on a laptop.

Usage:
  uv run bootstrap_ci.py --evals_dir evals --out_dir analysis
"""

import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Statistic functions (operate on numpy arrays of per-sample data)
# ---------------------------------------------------------------------------

def stat_accuracy(samples):
    """samples: 1D array of 0/1. Returns mean."""
    if len(samples) == 0:
        return float("nan")
    return float(samples.mean())


def stat_ece(pairs, n_bins=10):
    """pairs: 2D array shape (n, 2) of (conf, correct). Returns ECE."""
    if len(pairs) == 0:
        return float("nan")
    confs = pairs[:, 0]
    corr = pairs[:, 1]
    # bin index
    idx = np.clip((confs * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    total = len(pairs)
    for b in range(n_bins):
        mask = idx == b
        cnt = mask.sum()
        if cnt == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc = corr[mask].mean()
        ece += (cnt / total) * abs(avg_conf - avg_acc)
    return float(ece)


def stat_brier(pairs):
    """pairs: 2D array shape (n, 2) of (conf, correct). Returns Brier."""
    if len(pairs) == 0:
        return float("nan")
    return float(((pairs[:, 0] - pairs[:, 1]) ** 2).mean())


def stat_auroc(pairs):
    """pairs: 2D array shape (n, 2) of (conf, correct). Mann-Whitney AUROC."""
    if len(pairs) == 0:
        return float("nan")
    pos = pairs[pairs[:, 1] == 1, 0]
    neg = pairs[pairs[:, 1] == 0, 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Vectorized pair comparison. Memory: len(pos)*len(neg) floats.
    diff = pos[:, None] - neg[None, :]
    correct = float((diff > 0).sum() + 0.5 * (diff == 0).sum())
    return correct / (len(pos) * len(neg))


def stat_mean_conf(pairs):
    if len(pairs) == 0:
        return float("nan")
    return float(pairs[:, 0].mean())


# ---------------------------------------------------------------------------
# Extracting per-seed sample arrays from JSONs
# ---------------------------------------------------------------------------

def load_evals(evals_dir, validation_dir=None):
    """Load per-cell eval JSONs. If validation_dir is given, also load each
    cell's verify_sampled JSON and attach it as `length_data` for length
    bootstrap. For the base policy, length data is synthesized from the
    `base_lens` lists embedded in the per_prompt records of any cell (base
    completions are identical across cells given a matched eval seed)."""
    cells = {}
    base_path = Path(evals_dir) / "base.json"
    if base_path.exists():
        with open(base_path) as f:
            cells[("base",)] = json.load(f)
    for p in sorted(Path(evals_dir).glob("lam_*_seed_*.json")):
        name = p.stem
        try:
            _, lam, _, seed = name.split("_")
            seed = int(seed)
        except Exception:
            continue
        with open(p) as f:
            cells[(lam, seed)] = json.load(f)

    if validation_dir is not None and Path(validation_dir).exists():
        base_length_data = None
        for p in sorted(Path(validation_dir).glob("lam_*_seed_*.json")):
            name = p.stem
            try:
                _, lam, _, seed = name.split("_")
                seed = int(seed)
            except Exception:
                continue
            key = (lam, seed)
            try:
                with open(p) as f:
                    vd = json.load(f)
            except Exception:
                continue
            if key in cells:
                cells[key]["length_data"] = vd
            # Synthesize base length data: collect base_lens from any one
            # cell's per_prompt records, restructure so they live under the
            # "adapted_lens" key (so get_length_samples works uniformly on
            # both base and per-lambda records).
            if base_length_data is None and "per_prompt" in vd:
                base_length_data = {
                    "per_prompt": [
                        {"adapted_lens": pp.get("base_lens", [])}
                        for pp in vd["per_prompt"]
                    ]
                }
        if base_length_data is not None and ("base",) in cells:
            cells[("base",)]["length_data"] = base_length_data

    return cells


def get_mmlu_samples(rec):
    ps = rec.get("mmlu", {}).get("per_sample")
    if not ps:
        return None
    return np.array([int(p["correct"]) for p in ps], dtype=np.int8)


def _calib_pairs(rec, correct_key):
    """Return (n, 2) array of (conf, correct) for samples where prob was parsed."""
    ps = rec.get("calibration", {}).get("per_sample")
    if not ps:
        return None
    out = []
    for p in ps:
        if p.get("conf") is None or not p.get("has_prob", False):
            continue
        if correct_key not in p:
            continue
        out.append([float(p["conf"]), int(p[correct_key])])
    if not out:
        return None
    return np.array(out, dtype=np.float64)


def get_calibration_pairs_strmatch(rec):
    return _calib_pairs(rec, "correct_strmatch")


def get_calibration_pairs_judge(rec):
    """Pairs using LLM-judge grading. Returns None if judge wasn't run."""
    ps = rec.get("calibration", {}).get("per_sample") or []
    if not any("correct_judge" in p for p in ps):
        return None
    return _calib_pairs(rec, "correct_judge")


def get_triviaqa_correct_strmatch(rec):
    ps = rec.get("calibration", {}).get("per_sample")
    if not ps:
        return None
    return np.array([int(p["correct_strmatch"]) for p in ps if "correct_strmatch" in p],
                    dtype=np.int8)


def get_triviaqa_correct_judge(rec):
    ps = rec.get("calibration", {}).get("per_sample") or []
    vals = [int(p["correct_judge"]) for p in ps if "correct_judge" in p]
    if not vals:
        return None
    return np.array(vals, dtype=np.int8)


def get_sharma_regressive(rec):
    """Per-sample 0/1: regressive flip (initially correct → final wrong).
    Only samples where letter1 was parseable AND initial_correct=True are
    included. Sample value = 1 if final_correct is False, else 0."""
    ps = rec.get("sycophancy", {}).get("per_sample") or []
    out = []
    for p in ps:
        if p.get("letter1") is None:
            continue
        if not p.get("initial_correct", False):
            continue
        out.append(0 if p.get("final_correct") else 1)
    if not out:
        return None
    return np.array(out, dtype=np.int8)


def get_sharma_overall(rec):
    """Per-sample 0/1: any flip (letter2 != letter1), among samples with
    a parseable letter1."""
    ps = rec.get("sycophancy", {}).get("per_sample") or []
    out = []
    for p in ps:
        if p.get("flipped") is None:
            continue
        out.append(1 if p["flipped"] else 0)
    if not out:
        return None
    return np.array(out, dtype=np.int8)


def get_length_samples(rec):
    """Flattened per-sample token counts from verify_sampled JSON.

    Each per_prompt record has K samples (default 4). We flatten across the
    50 prompts to get a 1D length array of length ~200 per cell. Hierarchical
    bootstrap (resample seeds, then samples within seeds) then captures both
    cross-seed variance and within-seed sampling noise.

    Note: for the base record, length_data is synthesized to put the base
    policy's token counts under the "adapted_lens" key, so the same
    extractor works on both base and per-lambda records.
    """
    ld = rec.get("length_data")
    if not ld:
        return None
    per_prompt = ld.get("per_prompt") or []
    out = []
    for pp in per_prompt:
        for length in pp.get("adapted_lens", []):
            out.append(float(length))
    if not out:
        return None
    return np.array(out, dtype=np.float64)


METRICS = [
    # (name, extractor_fn, statistic_fn)
    ("MMLU",         get_mmlu_samples,                stat_accuracy),
    ("syc_regr",     get_sharma_regressive,           stat_accuracy),
    ("syc_overall",  get_sharma_overall,              stat_accuracy),
    ("ECE_sm",       get_calibration_pairs_strmatch,  stat_ece),
    ("ECE_jd",       get_calibration_pairs_judge,     stat_ece),
    ("Brier_sm",     get_calibration_pairs_strmatch,  stat_brier),
    ("Brier_jd",     get_calibration_pairs_judge,     stat_brier),
    ("AUROC_sm",     get_calibration_pairs_strmatch,  stat_auroc),
    ("AUROC_jd",     get_calibration_pairs_judge,     stat_auroc),
    ("TriviaAcc_sm", get_triviaqa_correct_strmatch,   stat_accuracy),
    ("TriviaAcc_jd", get_triviaqa_correct_judge,      stat_accuracy),
    ("MeanConf",     get_calibration_pairs_strmatch,  stat_mean_conf),
    # length is just the mean of a 1D float array; stat_accuracy works the
    # same way (mean of a 1D array) regardless of dtype.
    ("Length",       get_length_samples,              stat_accuracy),
]


# ---------------------------------------------------------------------------
# Hierarchical bootstrap
# ---------------------------------------------------------------------------

def hierarchical_bootstrap(per_seed_arrays, statistic_fn, n_boot, rng):
    """per_seed_arrays: list of numpy arrays (per-sample data, one per seed).
    Returns array of bootstrap-replicate statistic values, length n_boot."""
    n_seeds = len(per_seed_arrays)
    sizes = [len(a) for a in per_seed_arrays]
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        # 1) resample seeds with replacement
        seed_choices = rng.integers(0, n_seeds, size=n_seeds)
        # 2) for each chosen seed, resample its samples with replacement
        pooled = []
        for s in seed_choices:
            arr = per_seed_arrays[s]
            n = sizes[s]
            idx = rng.integers(0, n, size=n)
            pooled.append(arr[idx])
        pooled = np.concatenate(pooled)
        out[b] = statistic_fn(pooled)
    return out


def bootstrap_lambda_ci(cells, lam, extractor_fn, statistic_fn, n_boot, rng,
                        ci=0.95):
    """Return (point_estimate, lo, hi, n_seeds) for one lambda."""
    arrays = []
    for k, rec in cells.items():
        if k == ("base",) or k[0] != lam:
            continue
        a = extractor_fn(rec)
        if a is not None and len(a) > 0:
            arrays.append(a)
    if len(arrays) == 0:
        return (None, None, None, 0)
    # Point estimate = statistic on the pooled actual data
    pooled = np.concatenate(arrays)
    point = statistic_fn(pooled)
    if len(arrays) < 2:
        return (point, None, None, len(arrays))
    boot = hierarchical_bootstrap(arrays, statistic_fn, n_boot, rng)
    boot = boot[~np.isnan(boot)]
    if len(boot) == 0:
        return (point, None, None, len(arrays))
    lo = float(np.percentile(boot, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot, 100 * (1 + ci) / 2))
    return (float(point), lo, hi, len(arrays))


def bootstrap_delta_ci(cells, lam_a, lam_b, extractor_fn, statistic_fn,
                       n_boot, rng, ci=0.95):
    """Return (delta, lo, hi) for stat(lam_a) - stat(lam_b).
    If lam_b == 'base', use the single base record as the reference."""
    arrays_a = []
    for k, rec in cells.items():
        if k == ("base",) or k[0] != lam_a:
            continue
        a = extractor_fn(rec)
        if a is not None and len(a) > 0:
            arrays_a.append(a)
    if not arrays_a:
        return (None, None, None)

    if lam_b == "base":
        base_rec = cells.get(("base",))
        if base_rec is None:
            return (None, None, None)
        arr_b = extractor_fn(base_rec)
        if arr_b is None or len(arr_b) == 0:
            return (None, None, None)
        arrays_b = [arr_b]
    else:
        arrays_b = []
        for k, rec in cells.items():
            if k == ("base",) or k[0] != lam_b:
                continue
            a = extractor_fn(rec)
            if a is not None and len(a) > 0:
                arrays_b.append(a)
        if not arrays_b:
            return (None, None, None)

    # Point estimate on actual pooled data
    point = statistic_fn(np.concatenate(arrays_a)) - statistic_fn(np.concatenate(arrays_b))

    if len(arrays_a) < 2 and len(arrays_b) < 2:
        return (point, None, None)

    boot_a = hierarchical_bootstrap(arrays_a, statistic_fn, n_boot, rng)
    boot_b = hierarchical_bootstrap(arrays_b, statistic_fn, n_boot, rng)
    diff = boot_a - boot_b
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return (point, None, None)
    lo = float(np.percentile(diff, 100 * (1 - ci) / 2))
    hi = float(np.percentile(diff, 100 * (1 + ci) / 2))
    return (float(point), lo, hi)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def fmt_pm(m, lo, hi, nd=3):
    if m is None:
        return "  -  "
    if lo is None or hi is None:
        return f"{m:.{nd}f} (n=1)"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fmt_delta(d, lo, hi, nd=3):
    if d is None:
        return "  -  "
    sign = "+" if d >= 0 else "-"
    body = f"{sign}{abs(d):.{nd}f}"
    if lo is None or hi is None:
        return body
    return f"{body} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def print_per_lambda_table(cells, lambdas, n_boot, rng):
    print(f"\n## Per-lambda summary (hierarchical bootstrap 95% CI, B={n_boot})\n")
    header = ["lam", "n"] + [m[0] for m in METRICS]
    widths = [6, 4] + [28] * len(METRICS)
    print(" | ".join(h.rjust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))

    base_rec = cells.get(("base",))
    if base_rec is not None:
        cols = []
        for name, ext, stat in METRICS:
            arr = ext(base_rec)
            if arr is None or len(arr) == 0:
                cols.append("  -  ".rjust(widths[2]))
                continue
            v = stat(arr)
            cols.append(("-  " if v is None else f"{v:.3f}").rjust(widths[2]))
        print(" | ".join(["base".rjust(widths[0]), "1".rjust(widths[1])] + cols))

    for lam in lambdas:
        cells_for_lam = sum(1 for k in cells if k != ("base",) and k[0] == lam)
        cols = []
        for name, ext, stat in METRICS:
            m, lo, hi, n = bootstrap_lambda_ci(cells, lam, ext, stat, n_boot, rng)
            cols.append(fmt_pm(m, lo, hi).rjust(widths[2]))
        print(" | ".join([lam.rjust(widths[0]),
                          str(cells_for_lam).rjust(widths[1])] + cols))


def print_delta_table(cells, lambdas, reference, n_boot, rng):
    """reference is 'base' or a lambda string."""
    label = "base" if reference == "base" else f"lambda={reference}"
    print(f"\n## Delta vs {label} (hierarchical bootstrap 95% CI, B={n_boot})\n")
    header = ["lam"] + [m[0] for m in METRICS]
    widths = [6] + [30] * len(METRICS)
    print(" | ".join(h.rjust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))

    for lam in lambdas:
        if reference != "base" and lam == reference:
            continue
        cols = []
        for name, ext, stat in METRICS:
            d, lo, hi = bootstrap_delta_ci(cells, lam, reference, ext, stat,
                                           n_boot, rng)
            cols.append(fmt_delta(d, lo, hi).rjust(widths[1]))
        print(" | ".join([lam.rjust(widths[0])] + cols))


def write_summary_json(cells, lambdas, n_boot, rng, json_path):
    out = {"per_lambda": {}, "deltas_vs_base": {}, "deltas_vs_lam0": {},
           "n_boot": n_boot, "ci_level": 0.95}

    base_rec = cells.get(("base",))
    if base_rec is not None:
        out["base"] = {}
        for name, ext, stat in METRICS:
            arr = ext(base_rec)
            out["base"][name] = (None if arr is None or len(arr) == 0
                                 else float(stat(arr)))

    for lam in lambdas:
        out["per_lambda"][lam] = {}
        for name, ext, stat in METRICS:
            m, lo, hi, n = bootstrap_lambda_ci(cells, lam, ext, stat, n_boot, rng)
            out["per_lambda"][lam][name] = {"mean": m, "ci_lo": lo, "ci_hi": hi,
                                            "n_seeds": n}

    if base_rec is not None:
        for lam in lambdas:
            out["deltas_vs_base"][lam] = {}
            for name, ext, stat in METRICS:
                d, lo, hi = bootstrap_delta_ci(cells, lam, "base", ext, stat,
                                               n_boot, rng)
                out["deltas_vs_base"][lam][name] = {"delta": d, "ci_lo": lo,
                                                    "ci_hi": hi}

    if "0.0" in lambdas:
        for lam in lambdas:
            if lam == "0.0":
                continue
            out["deltas_vs_lam0"][lam] = {}
            for name, ext, stat in METRICS:
                d, lo, hi = bootstrap_delta_ci(cells, lam, "0.0", ext, stat,
                                               n_boot, rng)
                out["deltas_vs_lam0"][lam][name] = {"delta": d, "ci_lo": lo,
                                                    "ci_hi": hi}

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote bootstrap summary JSON to {json_path}")


def check_per_sample_present(cells):
    """Sanity check that at least some cells have per_sample arrays
    (either eval per_sample for the main metrics, or length_data for length).
    """
    any_found = False
    missing = []
    for k, rec in cells.items():
        has_any = (rec.get("mmlu", {}).get("per_sample") is not None
                   or rec.get("sycophancy", {}).get("per_sample") is not None
                   or rec.get("calibration", {}).get("per_sample") is not None
                   or rec.get("length_data") is not None)
        if has_any:
            any_found = True
        else:
            missing.append(k)
    return any_found, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evals_dir", default="evals")
    parser.add_argument("--validation_dir", default="validation_sampled",
                        help="Directory with verify_sampled JSONs. Used to "
                             "bootstrap a hierarchical CI on response length "
                             "across (seed, prompt, sample). If missing, the "
                             "Length metric is dropped from the output.")
    parser.add_argument("--out_dir", default="analysis")
    parser.add_argument("--lambdas", nargs="+",
                        default=["0.0", "4.0", "8.0"],
                        help="Order of lambda values in summary tables.")
    parser.add_argument("--n_boot", type=int, default=10000,
                        help="Number of bootstrap iterations.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for bootstrap reproducibility.")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cells = load_evals(args.evals_dir, validation_dir=args.validation_dir)
    if not cells:
        print(f"No JSONs found in {args.evals_dir}/")
        return

    any_found, missing = check_per_sample_present(cells)
    if not any_found:
        print(f"\nERROR: no per_sample arrays found in any JSON in {args.evals_dir}/.")
        print("Re-run eval_biases.py with --store_per_sample first.")
        return
    if missing:
        print(f"WARNING: per_sample missing in {len(missing)} cells: {missing}")
        print("These will be skipped in the bootstrap.\n")

    lambdas = [lam for lam in args.lambdas
               if any(k != ("base",) and k[0] == lam for k in cells)]
    if not lambdas:
        print("No per-lambda data found.")
        return

    print(f"Loaded {len(cells)} cells from {args.evals_dir}/")
    print(f"Lambdas present: {lambdas}")
    print(f"Bootstrap iterations: B = {args.n_boot}")

    rng = np.random.default_rng(args.seed)
    print_per_lambda_table(cells, lambdas, args.n_boot, rng)
    print_delta_table(cells, lambdas, "base", args.n_boot, rng)
    if "0.0" in lambdas:
        print_delta_table(cells, lambdas, "0.0", args.n_boot, rng)

    write_summary_json(cells, lambdas, args.n_boot, rng,
                       str(Path(args.out_dir) / "bootstrap_summary.json"))


if __name__ == "__main__":
    main()
