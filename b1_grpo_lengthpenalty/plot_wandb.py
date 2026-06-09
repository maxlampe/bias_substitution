"""
Batch-plot Weights & Biases CSV exports into publication PDFs.

Processes every *.csv in an input directory and writes one PDF (and PNG)
per file to an output directory. Handles the standard W&B grouped-export
column layout:
    <x column>,
    "Group: <name> - <metric>",
    "Group: <name> - <metric>__MIN",
    "Group: <name> - <metric>__MAX",
    "Group: <name> - _step", ...            (the _step columns are ignored)

Each group becomes one line (the mean) with a shaded band between its MIN
and MAX columns (the spread across the runs in that group, e.g. seeds).

Per-metric settings (y-axis label, y-range) live in the PER_FILE dict.
Everything shared across plots (fonts, markers, legend labels, colors) is
in the CONFIG block. Input and output directories can be overridden with
--in_dir / --out_dir.

Usage:
    uv run plot_wandb.py
    uv run plot_wandb.py --in_dir training_data --out_dir analysis
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ===========================================================================
# CONFIG  -- shared across all plots
# ===========================================================================

IN_DIR = "training_data"
OUT_DIR = "analysis"
WRITE_PNG = True                         # also write a PNG next to each PDF

# --- shared axis label / fonts ---
X_LABEL = r"Training step"
AXIS_LABEL_FS = 18
TICK_LABEL_FS = 16
TITLE_FS = 18
LEGEND_FS = 13

# --- markers / lines ---
MARKER = "o"
MARKER_SIZE = 7
MARKER_EVERY = 5                         # draw a marker every Nth point
LINE_WIDTH = 2.6
SHOW_BAND = True                         # shade MIN..MAX across the group
BAND_ALPHA = 0.18

# --- legend ---  raw W&B group name -> display label
LEGEND_LABELS = {
    "aggro_0": r"$\lambda = 0$",
    "aggro_4": r"$\lambda = 4$",
    "aggro_8": r"$\lambda = 8$",
}
LEGEND_LOC = "best"

# plot order and colors; groups not listed are appended with default colors
GROUP_ORDER = ["aggro_0", "aggro_4", "aggro_8"]
COLORS = {
    "aggro_0": "tab:blue",
    "aggro_4": "tab:orange",
    "aggro_8": "tab:red",
}

FIGSIZE = (7.0, 5.0)

# ===========================================================================
# PER_FILE  -- one entry per CSV. Key is the CSV file name.
#   ylabel : y-axis label for that metric
#   ylim   : (lo, hi) tuple or None for autoscale
#   title  : optional panel title, "" for none
# Files not listed here get a y-label derived from the CSV's metric column.
# ===========================================================================

PER_FILE = {
    "wandb_response_length.csv": {
        "ylabel": "Mean response length [tokens]",
        "xlim": [0., 600.],
        "ylim": None,
        "title": "",
    },
    "wandb_reward_length_pen.csv": {
        "ylabel": r"Penalized reward $\tilde{R}$",
        "xlim": [0., 600.],
        "ylim": None,
        "title": "",
    },
    "wandb_KL.csv": {
        "ylabel": r"KL to reference policy",
        "xlim": [0., 600.],
        "ylim": [0., 2.],
        "title": "",
    },
    "wandb_loss.csv": {
        "ylabel": r"GRPO loss",
        "xlim": [0., 600.],
        "ylim": [0., 0.025],
        "title": "",
    },
    "wandb_grad_norm.csv": {
        "ylabel": r"Gradient norm",
        "xlim": [0., 600.],
        "ylim": [0., 12.],
        "title": "",
    },
}


# ===========================================================================
# Parsing
# ===========================================================================

_GROUP_RE = re.compile(r"^Group:\s*(.+?)\s*-\s*(.+)$")


def parse_wandb_csv(path):
    """Return (x_col, dataframe, groups, metric_name) where groups maps
    group_name -> {"mean": col, "min": col-or-None, "max": col-or-None}."""
    df = pd.read_csv(path)
    x_col = df.columns[0]

    groups = {}
    metric_name = None
    for col in df.columns[1:]:
        m = _GROUP_RE.match(col)
        if not m:
            continue
        gname, metric = m.group(1), m.group(2)
        if metric.startswith("_step"):
            continue
        if metric.endswith("__MIN") or metric.endswith("__MAX"):
            continue
        metric_name = metric
        min_col = f"Group: {gname} - {metric}__MIN"
        max_col = f"Group: {gname} - {metric}__MAX"
        groups[gname] = {
            "mean": col,
            "min": min_col if min_col in df.columns else None,
            "max": max_col if max_col in df.columns else None,
        }
    if not groups:
        raise RuntimeError(
            f"No 'Group: <name> - <metric>' columns found in {path}."
        )
    return x_col, df, groups, metric_name


def ordered_group_names(groups):
    ordered = [g for g in GROUP_ORDER if g in groups]
    ordered += [g for g in groups if g not in ordered]
    return ordered


def default_ylabel(metric_name):
    """Fallback y-label derived from the metric column, e.g.
    'train/completion_length' -> 'completion length'."""
    if not metric_name:
        return "value"
    return metric_name.split("/")[-1].replace("_", " ")


# ===========================================================================
# Plotting
# ===========================================================================

def build_figure(x_col, df, groups, ylabel, xlim, ylim, title):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fallback_i = 0

    for gname in ordered_group_names(groups):
        cols = groups[gname]
        sub = df[[x_col, cols["mean"]]].dropna(subset=[cols["mean"]])
        x = sub[x_col].to_numpy()
        y = sub[cols["mean"]].to_numpy()

        color = COLORS.get(gname)
        if color is None:
            color = default_cycle[fallback_i % len(default_cycle)]
            fallback_i += 1
        label = LEGEND_LABELS.get(gname, gname)

        ax.plot(
            x, y,
            marker=MARKER, markersize=MARKER_SIZE, markevery=MARKER_EVERY,
            linewidth=LINE_WIDTH, color=color, label=label,
            markeredgewidth=0,
        )

        if SHOW_BAND and cols["min"] is not None and cols["max"] is not None:
            band = df[[x_col, cols["min"], cols["max"]]].dropna(
                subset=[cols["min"], cols["max"]])
            ax.fill_between(
                band[x_col].to_numpy(),
                band[cols["min"]].to_numpy(),
                band[cols["max"]].to_numpy(),
                color=color, alpha=BAND_ALPHA, linewidth=0,
            )

    ax.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS)
    if title:
        ax.set_title(title, fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FS)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc=LEGEND_LOC, fontsize=LEGEND_FS, framealpha=0.9)

    fig.tight_layout()
    return fig


def plot_one(csv_path, out_dir):
    name = csv_path.name
    x_col, df, groups, metric_name = parse_wandb_csv(csv_path)

    cfg = PER_FILE.get(name, {})
    ylabel = cfg.get("ylabel")
    if ylabel is None:
        ylabel = default_ylabel(metric_name)
        print(f"  [{name}] not in PER_FILE; using derived y-label "
              f"'{ylabel}'")
    xlim = cfg.get("xlim")
    ylim = cfg.get("ylim")
    title = cfg.get("title", "")

    fig = build_figure(x_col, df, groups, ylabel, xlim, ylim, title)

    stem = csv_path.stem
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=200)
    print(f"  [{name}] groups={list(groups.keys())} -> {pdf_path}")
    if WRITE_PNG:
        png_path = out_dir / f"{stem}.png"
        fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in_dir", default=IN_DIR,
                        help="Directory of W&B CSV exports.")
    parser.add_argument("--out_dir", default=OUT_DIR,
                        help="Directory to write PDFs/PNGs into.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(in_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files in {in_dir}/")
        return

    print(f"Plotting {len(csvs)} CSV file(s) from {in_dir}/ into {out_dir}/")
    for csv_path in csvs:
        try:
            plot_one(csv_path, out_dir)
        except Exception as e:
            print(f"  [{csv_path.name}] FAILED: {e}")

    print("Done.")


if __name__ == "__main__":
    main()