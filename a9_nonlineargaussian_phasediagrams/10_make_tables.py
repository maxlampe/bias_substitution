"""Table 1: representative validation cells with predicted vs measured values."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import make_sigma, compute_outcomes
from sweep import load_config, cell_id


def main(config_path, artifacts_dir):
    cfg = load_config(config_path)
    cells = cfg['validation_cells']
    defaults = cfg['defaults']

    alpha = float(defaults['alpha'])
    beta = float(defaults['beta'])
    w = float(defaults['w'])

    pred_rows = []
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
        pred_rows.append({
            'cell_name': cell['name'],
            'cell_id':   cid,
            'rho_12':    rho_12, 'rho_13': rho_13, 'gamma': gamma,
            'm_1':       m[0],
            'c':         c, 'beta_KL': beta_KL,
            'pred_g_1':      result['g_1'],
            'pred_delta_2':  result['delta_2'],
            'pred_delta_J':  result['delta_J'],
            'pred_dkl':      result['dkl'],
            'pred_psd_ok':   result['psd_ok'],
        })
    pred_df = pd.DataFrame(pred_rows)

    emp_path = Path(artifacts_dir) / 'regimes_emp.csv'
    if emp_path.exists():
        emp_df = pd.read_csv(emp_path)
        emp_slim = emp_df[['cell_id', 'mean_delta_2', 'se_delta_2',
                           'mean_delta_J', 'se_delta_J', 'mean_dkl',
                           'modal_regime', 'mean_based_regime', 'seed_agreement']].copy()
        merged = pred_df.merge(emp_slim, on='cell_id', how='left')
    else:
        merged = pred_df.copy()
        for col in ['mean_delta_2', 'se_delta_2', 'mean_delta_J', 'se_delta_J',
                    'mean_dkl', 'modal_regime', 'mean_based_regime', 'seed_agreement']:
            merged[col] = np.nan

    keep_cols = ['cell_name', 'rho_12', 'rho_13', 'gamma', 'm_1', 'c',
                 'pred_g_1', 'pred_delta_2', 'mean_delta_2', 'se_delta_2',
                 'pred_delta_J', 'mean_delta_J', 'se_delta_J',
                 'modal_regime', 'mean_based_regime', 'seed_agreement']
    table = merged[keep_cols]

    out_path_csv = Path(artifacts_dir) / 'tables' / 'table1.csv'
    out_path_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path_csv, index=False, float_format='%.4f')
    print(f'[table1] -> {out_path_csv}')
    print(table.to_string(index=False, float_format=lambda v: f'{v:.4f}' if isinstance(v, float) else str(v)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--artifacts', default='artifacts')
    args = p.parse_args()
    main(args.config, args.artifacts)
