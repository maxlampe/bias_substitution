"""
Three-panel regime-sweep figure for the length-bias substitution paper.

Reads the summary JSON produced by either:
  analyze_evals.py   -> analysis/summary.json           (t-based CIs)
  bootstrap_ci.py    -> analysis/bootstrap_summary.json (hierarchical bootstrap CIs)

Both files share the same schema:
  summary["base"][metric] = scalar
  summary["per_lambda"][lam][metric] = {mean, ci_lo, ci_hi, n_seeds/n}

The Length metric must already be present in the summary. Produce it by
running bootstrap_ci.py (or analyze_evals.py) with --validation_dir pointed
at the verify_sampled JSONs.

Panel layout (left to right):
  (a) Length (left y-axis) and MMLU accuracy (right y-axis)
  (b) Overconfidence: ECE (left y-axis) and mean verbalized confidence (right)
  (c) Sycophancy:     regressive flip rate (left) and overall flip rate (right)

Each panel:
  - errorbar markers with 95% CI
  - dashed horizontal line at the base (untrained) policy value
  - twin y-axis; the second metric's x values are offset to avoid overlap.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Font sizes  (+4 over matplotlib defaults)
# ---------------------------------------------------------------------------
AXIS_LABEL_FS = 18
TICK_LABEL_FS = 16
TITLE_FS = 18
LEGEND_FS = 13

# ---------------------------------------------------------------------------
# Errorbar appearance
# ---------------------------------------------------------------------------
MARKER_SIZE = 12
LINE_WIDTH = 2.6
CAP_SIZE = 9
CAP_THICK = 2.6
ERR_LINE_WIDTH = 2.4

# Offset (on the x-axis) for the second metric in twin-axis panels.
OFFSET_X_SECOND = 0.20


# ---------------------------------------------------------------------------
# Data accessors
# ---------------------------------------------------------------------------

def get_per_lambda(summary, lam, metric):
    d = summary.get("per_lambda", {}).get(lam, {}).get(metric)
    if d is None:
        return None, None, None
    return d.get("mean"), d.get("ci_lo"), d.get("ci_hi")


def get_base(summary, metric):
    return summary.get("base", {}).get(metric)


def to_errorbar_arrays(summary, lambdas, metric, x_offset=0.0):
    xs, ys, lo, hi = [], [], [], []
    for lam in lambdas:
        m, l, h = get_per_lambda(summary, lam, metric)
        if m is None:
            continue
        xs.append(float(lam) + x_offset)
        ys.append(m)
        lo.append(max(0.0, m - l) if l is not None else 0.0)
        hi.append(max(0.0, h - m) if h is not None else 0.0)
    return xs, ys, lo, hi


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric(ax, summary, lambdas, metric, color, label, marker="o",
                x_offset=0.0, plot_base_val=True):
    xs, ys, lo, hi = to_errorbar_arrays(summary, lambdas, metric,
                                        x_offset=x_offset)
    if not xs:
        return None, None
    ax.errorbar(
        xs, ys, yerr=[lo, hi],
        fmt=f"{marker}-", color=color, label=label,
        capsize=CAP_SIZE, capthick=CAP_THICK,
        linewidth=LINE_WIDTH, elinewidth=ERR_LINE_WIDTH,
        markersize=MARKER_SIZE, markeredgewidth=0,
    )
    base_val = get_base(summary, metric)
    if base_val is not None:
        base_label = "Base policy" if plot_base_val else None
        ax.axhline(
            base_val, linestyle="--", color=color, alpha=0.75,
            linewidth=2.6, label=base_label,
        )
    return None, None


def style_axes(ax, xlabel=None, ylabel=None, ylabel_color=None, title=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FS)
    if ylabel is not None:
        kwargs = {"fontsize": AXIS_LABEL_FS}
        if ylabel_color is not None:
            kwargs["color"] = ylabel_color
        ax.set_ylabel(ylabel, **kwargs)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FS)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FS,
                   labelcolor=ylabel_color if ylabel_color else "black")


def build_figure(summary, lambdas):
    # 3 panels, ~5.1 inches per panel (matching the previous per-panel width).
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 5.2))
    for ax in axes:
        ax.set_xticks([float(l) for l in lambdas])

    # ---- (a) Length (left) + MMLU (right) --------------------------------
    ax = axes[0]
    col_len = "tab:brown"
    col_mmlu = "tab:blue"

    plot_metric(ax, summary, lambdas, "Length", col_len,
                "Response length")
    style_axes(ax,
               xlabel=r"$\lambda$ (length penalty)",
               ylabel="Mean response length [tokens]",
               ylabel_color=col_len,
               title="(a) Length & Quality")
    ax.grid(True, alpha=0.3)

    ax_a2 = ax.twinx()
    plot_metric(ax_a2, summary, lambdas, "MMLU", col_mmlu,
                "MMLU accuracy", marker="s",
                x_offset=OFFSET_X_SECOND, plot_base_val=False)
    style_axes(ax_a2,
               ylabel=r"MMLU accuracy ($\uparrow$)",
               ylabel_color=col_mmlu)
    ax_a2.set_ylim(0.58, 0.68)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=LEGEND_FS, framealpha=0.9)

    # ---- (b) Overconfidence ----------------------------------------------
    ax = axes[1]
    col_ece = "tab:purple"
    col_conf = "tab:green"

    ece_metric = "ECE_jd" if get_base(summary, "ECE_jd") is not None else "ECE_sm"
    ece_label = ("ECE (judge)" if ece_metric.endswith("_jd")
                 else "ECE (string match)")
    plot_metric(ax, summary, lambdas, ece_metric, col_ece, ece_label)
    style_axes(ax,
               xlabel=r"$\lambda$ (length penalty)",
               ylabel=r"Expected calibration error ($\downarrow$)",
               ylabel_color=col_ece,
               title="(b) Overconfidence")
    ax.grid(True, alpha=0.3)

    ax_b2 = ax.twinx()
    plot_metric(ax_b2, summary, lambdas, "MeanConf", col_conf,
                "Verbalized \nconfidence", marker="s",
                x_offset=OFFSET_X_SECOND, plot_base_val=False)
    style_axes(ax_b2,
               ylabel="Mean verbalized confidence",
               ylabel_color=col_conf)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_b2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=LEGEND_FS, framealpha=0.9)

    # ---- (c) Sycophancy --------------------------------------------------
    ax = axes[2]
    col_regr = "tab:red"
    col_over = "tab:orange"

    plot_metric(ax, summary, lambdas, "syc_regr", col_regr,
                "Regressive flip \n(right$\\to$wrong)")
    style_axes(ax,
               xlabel=r"$\lambda$ (length penalty)",
               ylabel=r"Regressive flip rate ($\downarrow$)",
               ylabel_color=col_regr,
               title="(c) Sycophancy")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 0.705)

    ax_c2 = ax.twinx()
    plot_metric(ax_c2, summary, lambdas, "syc_overall", col_over,
                "Overall flip", marker="s",
                x_offset=OFFSET_X_SECOND, plot_base_val=False)
    style_axes(ax_c2,
               ylabel="Overall flip rate",
               ylabel_color=col_over)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_c2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=LEGEND_FS,
              framealpha=0.9)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", default="analysis/bootstrap_summary.json")
    parser.add_argument("--out_pdf", default="analysis/regime_figure.pdf")
    parser.add_argument("--out_png", default="analysis/regime_figure.png",
                        help="Set to empty string to skip PNG.")
    parser.add_argument("--lambdas", nargs="+", default=["0.0", "4.0", "8.0"])
    args = parser.parse_args()

    if not Path(args.summary).exists():
        raise FileNotFoundError(
            f"{args.summary} not found. Run analyze_evals.py or "
            f"bootstrap_ci.py first."
        )

    with open(args.summary) as f:
        summary = json.load(f)

    have = [lam for lam in args.lambdas if lam in summary.get("per_lambda", {})]
    if not have:
        raise RuntimeError(f"No lambdas with data in {args.summary}.")
    if set(have) != set(args.lambdas):
        print(f"WARNING: requested lambdas {args.lambdas} but only have data "
              f"for {have}; plotting available ones only.")

    if get_base(summary, "Length") is None:
        print("WARNING: no Length metric in the summary. The left axis of "
              "panel (a) will be empty. Re-run bootstrap_ci.py (or "
              "analyze_evals.py) with --validation_dir validation_sampled.")

    fig = build_figure(summary, have)

    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight", dpi=200)
    print(f"Wrote {args.out_pdf}")
    if args.out_png:
        fig.savefig(args.out_png, bbox_inches="tight", dpi=150)
        print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()