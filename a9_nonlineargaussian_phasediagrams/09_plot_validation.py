"""Predicted-vs-measured scatter plots for the validation cells."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics import make_sigma, compute_outcomes
from sweep import load_config, cell_id

# Third pass adds another +1 to each.
plt.rcParams.update({
    'font.size':      15,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})


def make_predictions_for_cells(cells, defaults):
    rows = []
    alpha = float(defaults['alpha'])
    beta = float(defaults['beta'])
    w = float(defaults['w'])
    for cell in cells:
        rho_12, rho_13 = float(cell['rho_12']), float(cell['rho_13'])
        gamma = float(cell['gamma'])
        m = tuple(float(x) for x in cell['m'])
        c = float(cell['c'])
        beta_KL = float(cell.get('beta_KL_override', defaults['beta_KL']))
        Sigma_ref = make_sigma(rho_12, rho_13)
        cid = cell_id(rho_12, rho_13, gamma, m, c, beta_KL, alpha, beta, w)
        result = compute_outcomes(
            theta_lin=np.array([alpha, beta, w]),
            theta_quad=gamma,
            Sigma_ref=Sigma_ref,
            m_audit=np.array(m),
            beta_KL=beta_KL,
            c=c,
            w_true=w,
        )
        rows.append({
            'cell_name':    cell['name'],
            'cell_id':      cid,
            'pred_delta_2': result['delta_2'],
            'pred_delta_J': result['delta_J'],
            'pred_g_1':     result['g_1'],
        })
    return pd.DataFrame(rows)


def plot_validation(pred_df, emp_df, out_path):
    # Keep only emp columns we need and avoid name collision with pred.
    emp_slim = emp_df[['cell_id', 'mean_delta_2', 'se_delta_2',
                       'mean_delta_J', 'se_delta_J', 'cell_name']].copy()
    merged = pred_df.merge(emp_slim, on='cell_id', suffixes=('', '_emp'))

    fig, (ax2, axJ) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, pred_col, meas_col, se_col, label in [
        (ax2, 'pred_delta_2', 'mean_delta_2', 'se_delta_2', r'$\Delta_2$'),
        (axJ, 'pred_delta_J', 'mean_delta_J', 'se_delta_J', r'$\Delta J$'),
    ]:
        x = merged[pred_col].to_numpy()
        y = merged[meas_col].to_numpy()
        se = merged[se_col].to_numpy()
        ax.errorbar(x, y, yerr=1.96 * se, fmt='o', capsize=4,
                    markersize=9, elinewidth=1.8, capthick=1.8)
        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        if hi == lo:
            hi = lo + 0.1
        pad = 0.08 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--',
                alpha=0.5, linewidth=2.0, label='y=x')
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        for _, row in merged.iterrows():
            ax.annotate(row['cell_name'], (row[pred_col], row[meas_col]),
                        fontsize=13, alpha=0.85,
                        xytext=(4, 4), textcoords='offset points')
        ax.set_xlabel(f'Predicted {label}')
        ax.set_ylabel(f'Measured {label} (mean $\\pm$ 1.96 SE)')
        ax.set_title(f'{label}: predicted vs measured')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    # Suptitle removed per user request.
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot_validation] -> {out_path}')


def main(config_path, artifacts_dir):
    cfg = load_config(config_path)
    cells = cfg['validation_cells']
    defaults = cfg['defaults']

    pred_df = make_predictions_for_cells(cells, defaults)
    emp_df = pd.read_csv(Path(artifacts_dir) / 'regimes_emp.csv')
    figs_dir = Path(artifacts_dir) / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)
    plot_validation(pred_df, emp_df, figs_dir / 'fig_validation.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--artifacts', default='artifacts')
    args = p.parse_args()
    main(args.config, args.artifacts)
