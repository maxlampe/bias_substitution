"""Apply eps-banded B* to empirical outcomes. Aggregate seed-level results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from regime_classifier import classify_regime
from sweep import load_config


def main(config_path, artifacts_dir):
    cfg = load_config(config_path)
    emp_cfg = cfg['epsilon_bands']['empirical']
    factor = float(emp_cfg['factor'])
    eps_off_min = float(emp_cfg['eps_off_min'])
    eps_J_min = float(emp_cfg['eps_J_min'])

    in_path = Path(artifacts_dir) / 'outcomes_emp.csv'
    out_path = Path(artifacts_dir) / 'regimes_emp.csv'

    df = pd.read_csv(in_path)
    rows = []
    for cid, group in df.groupby('cell_id'):
        n = len(group)
        # SE = std / sqrt(n)
        s_2 = float(group['delta_2'].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        s_J = float(group['delta_J'].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        if not np.isfinite(s_2):
            s_2 = 0.0
        if not np.isfinite(s_J):
            s_J = 0.0
        eps_off = max(factor * s_2, eps_off_min)
        eps_J = max(factor * s_J, eps_J_min)

        seed_labels = []
        for _, row in group.iterrows():
            if not bool(row['psd_ok']):
                seed_labels.append('PSD_fail')
            else:
                seed_labels.append(classify_regime(row['delta_2'], row['delta_J'],
                                                  eps_off=eps_off, eps_J=eps_J))
        modes = pd.Series(seed_labels).mode()
        modal = modes.iloc[0] if len(modes) else 'unknown'
        agreement = float((pd.Series(seed_labels) == modal).mean())

        mean_delta_2 = float(group['delta_2'].mean())
        mean_delta_J = float(group['delta_J'].mean())
        if np.isnan(mean_delta_2) or np.isnan(mean_delta_J):
            mean_label = 'PSD_fail'
        else:
            mean_label = classify_regime(mean_delta_2, mean_delta_J,
                                         eps_off=eps_off, eps_J=eps_J)

        rows.append({
            'cell_id':            cid,
            'cell_name':          group['cell_name'].iloc[0],
            'rho_12':             group['rho_12'].iloc[0],
            'rho_13':             group['rho_13'].iloc[0],
            'gamma':              group['gamma'].iloc[0],
            'm_1':                group['m_1'].iloc[0],
            'c':                  group['c'].iloc[0],
            'beta_KL':            group['beta_KL'].iloc[0],
            'n_seeds':            int(n),
            'mean_delta_2':       mean_delta_2,
            'se_delta_2':         s_2,
            'mean_delta_J':       mean_delta_J,
            'se_delta_J':         s_J,
            'mean_dkl':           float(group['dkl'].mean()),
            'eps_off':            eps_off,
            'eps_J':              eps_J,
            'modal_regime':       modal,
            'seed_agreement':     agreement,
            'mean_based_regime':  mean_label,
            'mean_theta_hat_1':   float(group['theta_hat_1'].mean()),
            'mean_theta_hat_2':   float(group['theta_hat_2'].mean()),
            'mean_theta_hat_3':   float(group['theta_hat_3'].mean()),
            'mean_theta_hat_4':   float(group['theta_hat_4'].mean()),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f'[classify_emp] wrote {len(out_df)} cells -> {out_path}')
    cols = ['cell_name', 'modal_regime', 'mean_based_regime', 'seed_agreement',
            'mean_delta_2', 'mean_delta_J']
    print(out_df[cols].to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--artifacts', default='artifacts')
    args = p.parse_args()
    main(args.config, args.artifacts)
