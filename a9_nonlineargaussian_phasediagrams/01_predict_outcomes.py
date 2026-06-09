"""Closed-form predictions for all configured sweeps.

Writes artifacts/predictions/{sweep_name}.csv with one row per cell.
No stochastic content; pure population formulas.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import make_sigma, compute_outcomes
from sweep import load_config, iterate_sweep


def predict_sweep(sweep_cfg, defaults):
    rows = []
    for params in iterate_sweep(sweep_cfg, defaults):
        Sigma_ref = make_sigma(params['rho_12'], params['rho_13'])
        theta_lin_pop = np.array([params['alpha'], params['beta'], params['w']])
        theta_quad_pop = params['gamma']
        result = compute_outcomes(
            theta_lin=theta_lin_pop,
            theta_quad=theta_quad_pop,
            Sigma_ref=Sigma_ref,
            m_audit=np.array(params['m']),
            beta_KL=params['beta_KL'],
            c=params['c'],
            w_true=params['w'],
        )
        row = {
            'cell_id': params['cell_id'],
            'rho_12':  params['rho_12'],
            'rho_13':  params['rho_13'],
            'gamma':   params['gamma'],
            'm_1':     params['m'][0],
            'm_2':     params['m'][1],
            'm_3':     params['m'][2],
            'c':       params['c'],
            'beta_KL': params['beta_KL'],
            'alpha':   params['alpha'],
            'beta':    params['beta'],
            'w':       params['w'],
            'g_1':     result['g_1'],
            'delta_1': result['delta_1'],
            'delta_2': result['delta_2'],
            'delta_3': result['delta_3'],
            'delta_J': result['delta_J'],
            'dkl':     result['dkl'],
            'psd_ok':  result['psd_ok'],
            'precision_eigmin': result['precision_eigmin'],
            'sigma_ref_eigmin': result['sigma_ref_eigmin'],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main(config_path, artifacts_dir):
    cfg = load_config(config_path)
    out_dir = Path(artifacts_dir) / 'predictions'
    out_dir.mkdir(parents=True, exist_ok=True)

    for sweep_name, sweep_cfg in cfg['sweeps'].items():
        df = predict_sweep(sweep_cfg, cfg['defaults'])
        out_path = out_dir / f'{sweep_name}.csv'
        df.to_csv(out_path, index=False)
        n_total = len(df)
        n_psd_fail = int((~df['psd_ok']).sum())
        print(f'[predict] {sweep_name}: {n_total} rows ({n_psd_fail} PSD-fail) -> {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--artifacts', default='artifacts')
    args = p.parse_args()
    main(args.config, args.artifacts)
