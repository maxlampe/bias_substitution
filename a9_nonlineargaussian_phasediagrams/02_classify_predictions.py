"""Apply B* (machine-eps tolerance) to analytic predictions.

Writes artifacts/regimes_pred/{sweep}.csv with a 'regime' column added.
"""

import argparse
from pathlib import Path

import pandas as pd

from regime_classifier import classify_regime
from sweep import load_config


def classify_df(df, eps_off, eps_J):
    labels = []
    for _, row in df.iterrows():
        if not bool(row['psd_ok']):
            labels.append('PSD_fail')
        else:
            labels.append(classify_regime(row['delta_2'], row['delta_J'],
                                          eps_off=eps_off, eps_J=eps_J))
    out = df.copy()
    out['regime'] = labels
    return out


def main(config_path, artifacts_dir):
    cfg = load_config(config_path)
    eps_off = float(cfg['epsilon_bands']['analytic']['eps_off'])
    eps_J = float(cfg['epsilon_bands']['analytic']['eps_J'])

    pred_dir = Path(artifacts_dir) / 'predictions'
    out_dir = Path(artifacts_dir) / 'regimes_pred'
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(pred_dir.glob('*.csv')):
        df = pd.read_csv(path)
        labelled = classify_df(df, eps_off, eps_J)
        out_path = out_dir / path.name
        labelled.to_csv(out_path, index=False)
        counts = labelled['regime'].value_counts().to_dict()
        print(f'[classify_pred] {path.name}: {counts} -> {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--artifacts', default='artifacts')
    args = p.parse_args()
    main(args.config, args.artifacts)
