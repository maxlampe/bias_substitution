"""Generate Bradley-Terry preferences for the validation cells.

For each validation cell and each of S seeds, draw (Y, Y') ~ N(m, Sigma_ref)
iid, compute BT logits from R_anno (which includes the bilinear term), sample
binary labels, and save as a compressed .npz.

Idempotent: existing files are skipped.
"""

import argparse
from pathlib import Path

import numpy as np

from analytics import make_sigma
from sweep import load_config, cell_id, derived_seed


def generate_one(Sigma, m, alpha, beta, w, gamma, N, seed):
    rng = np.random.default_rng(seed)
    m = np.asarray(m, dtype=float)
    Y = rng.multivariate_normal(m, Sigma, size=N)
    Y_prime = rng.multivariate_normal(m, Sigma, size=N)

    def R(y):
        return (alpha * y[:, 0]
                + beta * y[:, 1]
                + w * y[:, 2]
                + gamma * y[:, 0] ** 2)

    logits = R(Y_prime) - R(Y)
    logits = np.clip(logits, -30.0, 30.0)  # avoid overflow in sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.uniform(size=N) < probs).astype(np.int8)
    return Y, Y_prime, labels


def main(config_path, artifacts_dir):
    cfg = load_config(config_path)
    defaults = cfg['defaults']
    cells = cfg['validation_cells']

    out_dir = Path(artifacts_dir) / 'preferences'
    out_dir.mkdir(parents=True, exist_ok=True)

    N = int(defaults['N'])
    S = int(defaults['S'])
    master_seed = int(defaults['master_seed'])
    alpha = float(defaults['alpha'])
    beta = float(defaults['beta'])
    w = float(defaults['w'])

    n_generated, n_skipped = 0, 0
    for cell in cells:
        rho_12, rho_13 = float(cell['rho_12']), float(cell['rho_13'])
        gamma = float(cell['gamma'])
        m = tuple(float(x) for x in cell['m'])
        c = float(cell['c'])
        beta_KL = float(cell.get('beta_KL_override', defaults['beta_KL']))
        Sigma = make_sigma(rho_12, rho_13)

        eigmin = float(np.linalg.eigvalsh(Sigma).min())
        if eigmin <= 1e-10:
            print(f'[gen] SKIP {cell["name"]}: Sigma_ref non-PSD (eigmin={eigmin:.3e})')
            continue

        cid = cell_id(rho_12, rho_13, gamma, m, c, beta_KL, alpha, beta, w)
        for s in range(S):
            seed = derived_seed(master_seed, cid, s)
            out_path = out_dir / f'{cid}_seed{s:04d}.npz'
            if out_path.exists():
                n_skipped += 1
                continue
            Y, Y_prime, labels = generate_one(Sigma, m, alpha, beta, w, gamma, N, seed)
            # np.savez (uncompressed) is ~25x faster than savez_compressed
            # at the cost of ~5% larger files. Matters at large N.
            np.savez(out_path, Y=Y, Y_prime=Y_prime, labels=labels)
            n_generated += 1
        print(f'[gen] {cell["name"]:>14s}  cell_id={cid}  ({S} seeds requested)')
    print(f'[gen] done: generated {n_generated}, skipped {n_skipped} existing')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--artifacts', default='artifacts')
    args = p.parse_args()
    main(args.config, args.artifacts)
