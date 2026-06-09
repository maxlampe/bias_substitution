"""ε-banded B* classifier (Theorem 3.14 / Section A.7 of the paper)."""

import numpy as np


REGIME_LABELS = ('R0', 'R0_cont', 'R1_neutral', 'R1_harmful', 'R2', 'R3')

REGIME_COLORS = {
    'R0':         '#2ecc71',  # green
    'R0_cont':    '#f39c12',  # orange
    'R1_neutral': '#3498db',  # blue
    'R1_harmful': '#e74c3c',  # red
    'R2':         '#9b59b6',  # purple
    'R3':         '#95a5a6',  # gray
    'PSD_fail':   '#000000',  # black
    'unknown':    '#ffffff',  # white
}

REGIME_ORDER = ['R0', 'R0_cont', 'R1_neutral', 'R1_harmful', 'R2', 'R3', 'PSD_fail']


def classify_regime(delta_offtarget, delta_J, eps_off=1e-8, eps_J=1e-8):
    """Apply B*_eps to a single observation.

    Parameters
    ----------
    delta_offtarget : array-like or scalar
        Delta_j values on the off-target spurious axes Phi_sp \\ {phi_i}.
        In our setup this is a single scalar (delta_2).
    delta_J : float
        Change in true reward.
    eps_off, eps_J : float
        Tolerance bands; default near machine eps for analytic predictions.

    Returns
    -------
    label : str, one of REGIME_LABELS.
    """
    arr = np.atleast_1d(np.asarray(delta_offtarget, dtype=float))
    rotation = bool(np.any(np.abs(arr) > eps_off))
    if delta_J > eps_J:
        return 'R0_cont' if rotation else 'R0'
    elif delta_J < -eps_J:
        return 'R1_harmful' if rotation else 'R2'
    else:
        return 'R1_neutral' if rotation else 'R3'
