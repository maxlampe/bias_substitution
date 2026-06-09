"""For each fitted theta_hat, compute the closed-form outcomes downstream.

Writes artifacts/outcomes_emp.csv with one row per (cell, seed).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import make_sigma, compute_outcomes
from sweep import load_config, cell_id


def main(config_path, artifacts_dir):
    cfg = load_config(config_path)
    defaults = cfg['defaults']
    cells = cfg['validation_cells']

    fit_dir = Path(artifacts_dir) / 'rm_fits'
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    S = int(defaults['S'])
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

        for s in range(S):
            fit_path = fit_dir / f'{cid}_seed{s:04d}.npz'
            if not fit_path.exists():
                continue
            data = np.load(fit_path)
            theta_hat = data['theta_hat']
            theta_lin_hat = np.asarray(theta_hat[:3], dtype=float)
            theta_quad_hat = float(theta_hat[3])

            result = compute_outcomes(
                theta_lin=theta_lin_hat,
                theta_quad=theta_quad_hat,
                Sigma_ref=Sigma_ref,
                m_audit=np.array(m),
                beta_KL=beta_KL,
                c=c,
                w_true=w,
            )

            rows.append({
                'cell_name':    cell['name'],
                'cell_id':      cid,
                'seed':         s,
                'rho_12':       rho_12, 'rho_13': rho_13,
                'gamma':        gamma,
                'm_1':          m[0], 'm_2': m[1], 'm_3': m[2],
                'c':            c, 'beta_KL': beta_KL,
                'alpha':        alpha, 'beta': beta, 'w': w,
                'theta_hat_1':  float(theta_hat[0]),
                'theta_hat_2':  float(theta_hat[1]),
                'theta_hat_3':  float(theta_hat[2]),
                'theta_hat_4':  float(theta_hat[3]),
                'g_1':          result['g_1'],
                'delta_1':      result['delta_1'],
                'delta_2':      result['delta_2'],
                'delta_3':      result['delta_3'],
                'delta_J':      result['delta_J'],
                'dkl':          result['dkl'],
                'psd_ok':       bool(result['psd_ok']),
                'precision_eigmin': result['precision_eigmin'],
                'converged':    bool(data.get('converged', True)),
                'final_grad_norm': float(data.get('final_grad_norm', np.nan)),
            })

    df = pd.DataFrame(rows)
    out_path = out_dir / 'outcomes_emp.csv'
    df.to_csv(out_path, index=False)
    print(f'[outcomes_emp] wrote {len(df)} rows -> {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--artifacts', default='artifacts')
    args = p.parse_args()
    main(args.config, args.artifacts)
