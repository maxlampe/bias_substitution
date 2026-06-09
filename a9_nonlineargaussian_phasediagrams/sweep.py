"""Sweep utilities: cell IDs, parameter grids, deterministic seeds, config loader."""

import hashlib
import itertools

import numpy as np


def cell_id(rho_12, rho_13, gamma, m, c, beta_KL, alpha, beta, w):
    """Deterministic 16-hex-char id for a parameter cell."""
    m1, m2, m3 = m
    parts = [
        ('r12', rho_12), ('r13', rho_13), ('g', gamma),
        ('m1', m1), ('m2', m2), ('m3', m3),
        ('c', c), ('bk', beta_KL),
        ('a', alpha), ('b', beta), ('w', w),
    ]
    s = '_'.join(f'{k}={v:.6f}' for k, v in parts)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def derived_seed(master_seed, *components):
    """Derive a per-cell uint32 seed from master seed and string/int components."""
    s = f'{master_seed}|' + '|'.join(str(c) for c in components)
    digest = hashlib.md5(s.encode()).digest()
    return int.from_bytes(digest[:4], 'big')


def expand_sweep_axis(spec):
    """Convert a sweep axis spec to a list of float values.

    Accepts either:
      - {'start': float, 'stop': float, 'num': int}  -> linspace
      - list/tuple of floats                          -> as-is
    """
    if isinstance(spec, dict):
        return np.linspace(spec['start'], spec['stop'], spec['num']).tolist()
    return [float(v) for v in spec]


def iterate_sweep(sweep_cfg, defaults):
    """Yield dicts of parameters for each cell in a sweep.

    sweep_cfg : dict possibly containing rho_12, rho_13, gamma, m_1, m_2, m_3, c
                and optional override 'beta_KL_override'.
    defaults  : dict with alpha, beta, w, beta_KL, c.

    Does NOT filter on PSD; the caller handles PSD-fail cells.
    """
    rho_12 = expand_sweep_axis(sweep_cfg.get('rho_12', [0.0]))
    rho_13 = expand_sweep_axis(sweep_cfg.get('rho_13', [0.0]))
    gamma  = expand_sweep_axis(sweep_cfg.get('gamma',  [0.0]))
    m_1    = expand_sweep_axis(sweep_cfg.get('m_1',    [0.0]))
    m_2    = expand_sweep_axis(sweep_cfg.get('m_2',    [0.0]))
    m_3    = expand_sweep_axis(sweep_cfg.get('m_3',    [0.0]))
    c_vals = expand_sweep_axis(sweep_cfg.get('c',      [defaults['c']]))

    beta_KL = float(sweep_cfg.get('beta_KL_override', defaults['beta_KL']))
    alpha   = float(defaults['alpha'])
    beta    = float(defaults['beta'])
    w       = float(defaults['w'])

    for r12, r13, g, mm1, mm2, mm3, c in itertools.product(rho_12, rho_13, gamma, m_1, m_2, m_3, c_vals):
        yield {
            'rho_12':  float(r12),
            'rho_13':  float(r13),
            'gamma':   float(g),
            'm':       (float(mm1), float(mm2), float(mm3)),
            'c':       float(c),
            'beta_KL': beta_KL,
            'alpha':   alpha, 'beta': beta, 'w': w,
            'cell_id': cell_id(r12, r13, g, (mm1, mm2, mm3), c, beta_KL, alpha, beta, w),
        }


def load_config(path):
    """Load yaml config."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
