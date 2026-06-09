"""Figure 1 (headline phase diagram) and Figure 2 (gamma slice triptych)."""

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

# Baseline fonts for fig1 (+1 from previous pass). gamma_slice locally bumps a
# further +2 via rc_context so that fig2's text ends up +3 above fig1.
plt.rcParams.update({
    'font.size':      15,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

GAMMA_SLICE_FONT_BUMP = {
    'font.size':      17,
    'axes.titlesize': 18,
    'axes.labelsize': 17,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
}


def regime_to_color_grid(df_subset, rho_12_axis, rho_13_axis):
    """Build a (H, W, 3) RGB grid for the phase diagram.

    rho_12 indexes columns (x), rho_13 indexes rows (y).
    Cells not present in df_subset are colored white.
    """
    H, W = len(rho_13_axis), len(rho_12_axis)
    grid = np.ones((H, W, 3))  # white
    pivot = df_subset.pivot_table(index='rho_13', columns='rho_12',
                                  values='regime', aggfunc='first')
    pivot = pivot.reindex(index=rho_13_axis, columns=rho_12_axis)
    for i, r13 in enumerate(rho_13_axis):
        for j, r12 in enumerate(rho_12_axis):
            label = pivot.iloc[i, j]
            if pd.isna(label):
                continue
            grid[i, j] = mcolors.to_rgb(REGIME_COLORS.get(label, '#ffffff'))
    return grid


def overlay_boundaries(ax, rho_axis_lim, alpha, beta, w, gamma, m_1,
                       line_color='k', lw_scale=1.0):
    """Overlay axis lines, the g_1 = 0 line, and the unit-disk reference circle.

    `line_color` and `lw_scale` let callers darken/thicken the boundaries for
    background colormaps where black-on-dark vanishes (see plot_magnitude_panel).
    """
    ax.axvline(0.0, color=line_color, linestyle='--',
               linewidth=0.9 * lw_scale, alpha=0.75)
    ax.axhline(0.0, color=line_color, linestyle='--',
               linewidth=0.9 * lw_scale, alpha=0.75)
    # g_1 = alpha + beta * rho_12 + w * rho_13 + 2 * gamma * m_1 = 0
    # => rho_13 = -(alpha + beta * rho_12 + 2 * gamma * m_1) / w
    if w != 0:
        r12_line = np.linspace(rho_axis_lim[0], rho_axis_lim[1], 200)
        r13_line = -(alpha + beta * r12_line + 2.0 * gamma * m_1) / w
        mask = (r13_line >= rho_axis_lim[0]) & (r13_line <= rho_axis_lim[1])
        if mask.any():
            ax.plot(r12_line[mask], r13_line[mask],
                    color=line_color, linestyle='-',
                    linewidth=1.3 * lw_scale, alpha=0.9, label='$g_1=0$')
    # Unit-disk reference
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta),
            color=line_color, linewidth=0.6 * lw_scale, alpha=0.4)


def make_legend(ax, labels_present):
    patches = []
    for label in REGIME_ORDER:
        if label in labels_present:
            patches.append(mpatches.Patch(color=REGIME_COLORS[label], label=label))
    if patches:
        ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.02, 0.5))


def plot_single_phase_diagram(df_subset, params, ax):
    rho_12_axis = sorted(df_subset['rho_12'].unique())
    rho_13_axis = sorted(df_subset['rho_13'].unique())
    grid = regime_to_color_grid(df_subset, rho_12_axis, rho_13_axis)
    extent = (min(rho_12_axis), max(rho_12_axis), min(rho_13_axis), max(rho_13_axis))
    ax.imshow(grid, origin='lower', extent=extent, aspect='auto', interpolation='nearest')
    overlay_boundaries(
        ax,
        rho_axis_lim=(min(rho_12_axis), max(rho_12_axis)),
        alpha=params['alpha'], beta=params['beta'], w=params['w'],
        gamma=params['gamma'], m_1=params['m_1'],
        line_color='k', lw_scale=1.6,
    )
    ax.set_xlabel(r'$\rho_{12}$')
    ax.set_ylabel(r'$\rho_{13}$')


def headline(df, defaults, out_path):
    gamma_val = float(df['gamma'].iloc[0])
    fig, ax = plt.subplots(figsize=(7, 5))
    params = {
        'alpha': defaults['alpha'], 'beta': defaults['beta'], 'w': defaults['w'],
        'gamma': gamma_val, 'm_1': 0.0,
    }
    plot_single_phase_diagram(df, params, ax)
    # Title removed per user request.
    make_legend(ax, set(df['regime'].unique()))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] -> {out_path}')


def magnitude_grid(df_subset, rho_12_axis, rho_13_axis, value_col):
    """Return a 2D float array of df_subset[value_col] pivoted on (rho_13, rho_12).

    PSD-fail / missing cells are NaN. Callers can use np.abs and np.ma.masked_invalid.
    """
    pivot = df_subset.pivot_table(index='rho_13', columns='rho_12',
                                  values=value_col, aggfunc='first')
    pivot = pivot.reindex(index=rho_13_axis, columns=rho_12_axis)
    return pivot.to_numpy()


def plot_magnitude_panel(df_subset, params, ax, vmin, vmax, value_col='delta_J',
                         take_abs=True, cmap_name='Reds'):
    """Draw a heatmap of |df[value_col]| on (rho_12, rho_13).

    Default colormap is 'Reds' (white at vmin, dark red at vmax). White at the
    low end keeps black regime-boundary lines visible where |Delta J| is small,
    which is exactly where the boundaries live (axes and the g_1 = 0 line).
    PSD-fail / NaN cells are drawn black. Returns the AxesImage for colorbar
    attachment.
    """
    rho_12_axis = sorted(df_subset['rho_12'].unique())
    rho_13_axis = sorted(df_subset['rho_13'].unique())
    grid = magnitude_grid(df_subset, rho_12_axis, rho_13_axis, value_col)
    if take_abs:
        grid = np.abs(grid)
    grid_masked = np.ma.masked_invalid(grid)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('#000000')  # PSD-fail color

    extent = (min(rho_12_axis), max(rho_12_axis), min(rho_13_axis), max(rho_13_axis))
    im = ax.imshow(grid_masked, origin='lower', extent=extent, aspect='auto',
                   interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    overlay_boundaries(
        ax,
        rho_axis_lim=(min(rho_12_axis), max(rho_12_axis)),
        alpha=params['alpha'], beta=params['beta'], w=params['w'],
        gamma=params['gamma'], m_1=params['m_1'],
        line_color='k', lw_scale=1.6,
    )
    ax.set_xlabel(r'$\rho_{12}$')
    ax.set_ylabel(r'$\rho_{13}$')
    return im


def gamma_slice(df, defaults, out_path, value_col='delta_J'):
    """Triptych over gamma showing |delta_J| heatmaps with regime boundaries.

    The regime labels are gamma-independent at m=0 (signs of Delta_2, Delta_J are
    pinned by signs of (rho_12, rho_13, g_1)). What gamma changes is the
    magnitude of the substitution effect, visible here as the colormap intensity.
    Boundaries (rho_12=0, rho_13=0, g_1=0) are overlaid in black.

    Wrapped in plt.rc_context with GAMMA_SLICE_FONT_BUMP so this figure ends up
    +3 font units above fig1 (per user request).
    """
    with plt.rc_context(GAMMA_SLICE_FONT_BUMP):
        gammas = sorted(df['gamma'].unique())
        n = len(gammas)

        # Shared color scale across panels: same vmin/vmax for direct comparison.
        finite_vals = df[value_col].abs().to_numpy()
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        vmin = 0.0
        vmax = float(np.max(finite_vals)) if finite_vals.size else 1.0

        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.5),
                                 sharex=True, sharey=True, constrained_layout=True)
        if n == 1:
            axes = [axes]

        last_im = None
        for ax, g in zip(axes, gammas):
            sub = df[df['gamma'] == g]
            params = {
                'alpha': defaults['alpha'], 'beta': defaults['beta'], 'w': defaults['w'],
                'gamma': float(g), 'm_1': 0.0,
            }
            last_im = plot_magnitude_panel(sub, params, ax, vmin=vmin, vmax=vmax,
                                           value_col=value_col, take_abs=True)
            ax.set_title(fr'$\gamma={g:g}$')

        # Shared colorbar to the right of the rightmost panel.
        label_pretty = {'delta_J': r'$|\Delta J|$', 'delta_2': r'$|\Delta_2|$'}.get(value_col, value_col)
        cbar = fig.colorbar(last_im, ax=axes, fraction=0.04, pad=0.02, shrink=0.9)
        cbar.set_label(label_pretty)
        cbar.ax.tick_params(labelsize=plt.rcParams['ytick.labelsize'])

        # Suptitle removed per user request.
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f'[plot] -> {out_path}')


def main(config_path, artifacts_dir):
    cfg = load_config(config_path)
    defaults = cfg['defaults']
    regimes_dir = Path(artifacts_dir) / 'regimes_pred'
    figs_dir = Path(artifacts_dir) / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)

    headline_path = regimes_dir / 'headline.csv'
    if headline_path.exists():
        headline(pd.read_csv(headline_path), defaults, figs_dir / 'fig1_headline.pdf')
    else:
        print(f'[plot] skipping headline: {headline_path} not found')

    slice_path = regimes_dir / 'gamma_slice.csv'
    if slice_path.exists():
        gamma_slice(pd.read_csv(slice_path), defaults, figs_dir / 'fig2_gamma_slice.pdf')
    else:
        print(f'[plot] skipping gamma_slice: {slice_path} not found')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--artifacts', default='artifacts')
    args = p.parse_args()
    main(args.config, args.artifacts)
