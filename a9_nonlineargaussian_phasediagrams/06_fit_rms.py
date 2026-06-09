"""Fit polynomial-feature BT-MLE to each preference file.

Features: phi(y) = (y[0], y[1], y[2], y[0]^2). BT-MLE = logistic regression on
X = phi(Y') - phi(Y) with labels in {0, 1} via Newton-Raphson (bt_mle.py).

Idempotent: existing fit files are skipped.
"""

import argparse
from pathlib import Path

import numpy as np

from bt_mle import fit_logistic_regression


def poly_features(Y):
    """phi(y) = (y[0], y[1], y[2], y[0]^2). Returns shape (N, 4)."""
    return np.column_stack([Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 0] ** 2])


def fit_one(Y, Y_prime, labels, C=1e3):
    X = poly_features(Y_prime) - poly_features(Y)
    theta_hat, info = fit_logistic_regression(X, labels, C=C, max_iter=200, tol=1e-8)
    return theta_hat, info


def main(artifacts_dir, C):
    pref_dir = Path(artifacts_dir) / 'preferences'
    fit_dir = Path(artifacts_dir) / 'rm_fits'
    fit_dir.mkdir(parents=True, exist_ok=True)

    n_fit, n_skipped, n_failed = 0, 0, 0
    for pref_path in sorted(pref_dir.glob('*.npz')):
        out_path = fit_dir / pref_path.name
        if out_path.exists():
            n_skipped += 1
            continue
        data = np.load(pref_path)
        theta_hat, info = fit_one(data['Y'], data['Y_prime'], data['labels'], C=C)
        if not info['converged']:
            n_failed += 1
            print(f'[fit] WARN not converged: {pref_path.name}  '
                  f'grad_norm={info["final_grad_norm"]:.3e}  iters={info["n_iter"]}')
        np.savez(out_path, theta_hat=theta_hat,
                 n_iter=int(info['n_iter']),
                 converged=bool(info['converged']),
                 final_grad_norm=float(info['final_grad_norm']))
        n_fit += 1
    print(f'[fit] fitted {n_fit}, skipped {n_skipped} existing, '
          f'{n_failed} did not converge to tol')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--artifacts', default='artifacts')
    p.add_argument('--C', type=float, default=1e3,
                   help='inverse L2 regularization strength (sklearn convention)')
    args = p.parse_args()
    main(args.artifacts, args.C)
