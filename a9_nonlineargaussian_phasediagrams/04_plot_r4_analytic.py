"""Figure 3: R4 1D sweep over m_1.

Top panel: Delta_2 and Delta_J vs m_1 with m_1* marker.
Bottom panel: regime call colored along the sweep.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_classifier import REGIME_COLORS, REGIME_ORDER
from sweep import load_config

# Third pass adds another +1 to each.
plt.rcParams.update({
    'font.size':      15,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})


def plot_r4_1d(df, defaults, out_path):
    df = df.sort_values('m_1').reset_index(drop=True)
    fig, (ax_delta, ax_regime) = plt.subplots(
        2, 1, figsize=(7.5, 5.2), sharex=True,
        gridspec_kw={'height_ratios': [4, 1]},
    )

    ax_delta.plot(df['m_1'], df['delta_2'], label=r'$\Delta_2$',
                  color='tab:blue', marker='o', markersize=6, linewidth=2.4)
    ax_delta.plot(df['m_1'], df['delta_J'], label=r'$\Delta J$',
                  color='tab:red', marker='s', markersize=6, linewidth=2.4)
    ax_delta.axhline(0, color='k', linewidth=0.8, alpha=0.5)

    alpha = defaults['alpha']; beta = defaults['beta']; w = defaults['w']
    rho_12 = float(df['rho_12'].iloc[0])
    rho_13 = float(df['rho_13'].iloc[0])
    gamma = float(df['gamma'].iloc[0])
    if gamma != 0:
        m_1_star = -(alpha + beta * rho_12 + w * rho_13) / (2.0 * gamma)
        if df['m_1'].min() <= m_1_star <= df['m_1'].max():
            ax_delta.axvline(m_1_star, color='k', linestyle=':', linewidth=2.0, alpha=0.85,
                             label=fr'$m_1^*={m_1_star:.3f}$')

    ax_delta.set_ylabel('Outcome value')
    ax_delta.legend(loc='best')
    # Title removed per user request.
    ax_delta.grid(True, alpha=0.3)

    # Regime strip via imshow
    regime_to_int = {label: i for i, label in enumerate(REGIME_ORDER)}
    data = np.array([[regime_to_int.get(r, len(REGIME_ORDER)) for r in df['regime']]])
    cmap_list = [REGIME_COLORS.get(lbl, '#ffffff') for lbl in REGIME_ORDER]
    cmap = mcolors.ListedColormap(cmap_list)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(cmap_list) + 1) - 0.5,
                                ncolors=len(cmap_list))
    # Padding so the strip starts/ends nicely
    if len(df) > 1:
        step = (df['m_1'].iloc[-1] - df['m_1'].iloc[0]) / (len(df) - 1)
    else:
        step = 0.1
    extent = (df['m_1'].iloc[0] - step / 2, df['m_1'].iloc[-1] + step / 2, -0.5, 0.5)
    ax_regime.imshow(data, aspect='auto', cmap=cmap, norm=norm, extent=extent, interpolation='nearest')
    ax_regime.set_yticks([])
    ax_regime.set_xlabel(r'$m_1$')

    labels_present = set(df['regime'].unique())
    patches = [mpatches.Patch(color=REGIME_COLORS[lbl], label=lbl)
               for lbl in REGIME_ORDER if lbl in labels_present]
    if patches:
        ax_regime.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.8),
                         ncol=min(len(patches), 4))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] -> {out_path}')


def main(config_path, artifacts_dir):
    cfg = load_config(config_path)
    regimes_dir = Path(artifacts_dir) / 'regimes_pred'
    figs_dir = Path(artifacts_dir) / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)

    path = regimes_dir / 'r4_1d.csv'
    if not path.exists():
        print(f'[plot] skipping r4_1d: {path} not found')
        return
    plot_r4_1d(pd.read_csv(path), cfg['defaults'], figs_dir / 'fig3_r4_1d.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--artifacts', default='artifacts')
    args = p.parse_args()
    main(args.config, args.artifacts)
