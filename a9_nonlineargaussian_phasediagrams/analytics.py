"""Closed-form math for the phase-diagram experiment.

Setup
-----
- Response y in R^3.
- Spurious features Phi_sp = {phi_1 = y[0], phi_2 = y[1]}; structural phi_3 = y[2].
  Target index i = 0 throughout (TARGET_IDX).
- True reward: R(y) = w * y[2].
- Annotator reward (preference-generating):
    R_anno(y) = alpha * y[0] + beta * y[1] + w * y[2] + gamma * y[0]^2.
- Reward model has polynomial-in-y_0 basis:
    R_tilde(y) = theta_lin . y + theta_quad * y[0]^2,
  with theta_lin in R^3 and theta_quad scalar. In the population BT-MLE
  small-logit limit (and exactly under model-correctness), theta_lin = (alpha,
  beta, w) and theta_quad = gamma.
- Reference policy: pi_ref = N(0, Sigma_ref) with Sigma_ref structured by
  (rho_12, rho_13):
    Sigma_ref = [[1,      rho_12, rho_13],
                 [rho_12, 1,      0     ],
                 [rho_13, 0,      1     ]].
- Audit distribution: mu_diag = N(m, Sigma_diag); default Sigma_diag = Sigma_ref.
- Single-axis mitigation M_1 with strength c:
    g_1 = Cov_mu(phi_1, R_tilde) / Var_mu(phi_1)
        = alpha + beta * rho_12 + w * rho_13 + 2 * gamma * m[0]
      in the canonical case where Sigma_diag = Sigma_ref and Sigma_diag[0,0] = 1.
    R_tilde' = R_tilde - c * g_1 * phi_1, i.e., theta_lin[0] -= c * g_1.

Policy
------
With R_tilde(y) = a . y + (1/2) y^T H y, where a = theta_lin and
H = diag(2 * theta_quad, 0, 0):
  log pi*(y) ~ -1/2 y^T (Sigma_ref^{-1} - H/beta_KL) y + (a/beta_KL) . y,
so pi* = N(mu*, Sigma*) with
  Sigma*^{-1} = Sigma_ref^{-1} - H/beta_KL,
  mu*         = Sigma* . a / beta_KL.

Outcomes
--------
  Delta_j(pi, pi') = E_{pi'}[phi_j] - E_pi[phi_j] = mu_post[j] - mu_pre[j]
  Delta_J          = w * Delta_3
"""

import numpy as np


TARGET_IDX = 0
OFFTARGET_IDX = 1
STRUCTURAL_IDX = 2


def make_sigma(rho_12, rho_13):
    """Build Sigma_ref with Sigma_ii = 1, Sigma_12 = rho_12, Sigma_13 = rho_13, Sigma_23 = 0."""
    return np.array([
        [1.0, rho_12, rho_13],
        [rho_12, 1.0, 0.0],
        [rho_13, 0.0, 1.0],
    ])


def hessian(theta_quad):
    """H = diag(2 * theta_quad, 0, 0); R_tilde = a . y + 0.5 * y^T H y."""
    H = np.zeros((3, 3))
    H[TARGET_IDX, TARGET_IDX] = 2.0 * theta_quad
    return H


def policy_optimum(theta_lin, theta_quad, Sigma_ref, beta_KL, psd_tol=1e-10):
    """KL-regularized optimum.

    Returns
    -------
    dict with keys:
      'mu':                 mean of pi*  (None if PSD-fail or Sigma_ref non-PSD)
      'Sigma':              covariance of pi* (None if PSD-fail)
      'precision_eigmin':   min eigenvalue of Sigma*^{-1} (NaN if Sigma_ref non-PSD)
      'psd_ok':             True iff Sigma_ref PSD and Sigma*^{-1} PSD
    """
    # Sigma_ref PSD check first
    sigma_ref_eigmin = np.linalg.eigvalsh(Sigma_ref).min()
    if sigma_ref_eigmin <= psd_tol:
        return {'mu': None, 'Sigma': None,
                'precision_eigmin': float('nan'),
                'sigma_ref_eigmin': float(sigma_ref_eigmin),
                'psd_ok': False}
    Sigma_ref_inv = np.linalg.inv(Sigma_ref)
    Sigma_star_inv = Sigma_ref_inv - hessian(theta_quad) / beta_KL
    eigmin = np.linalg.eigvalsh(Sigma_star_inv).min()
    if eigmin <= psd_tol:
        return {'mu': None, 'Sigma': None,
                'precision_eigmin': float(eigmin),
                'sigma_ref_eigmin': float(sigma_ref_eigmin),
                'psd_ok': False}
    Sigma_star = np.linalg.inv(Sigma_star_inv)
    mu_star = Sigma_star @ np.asarray(theta_lin, dtype=float) / beta_KL
    return {'mu': mu_star, 'Sigma': Sigma_star,
            'precision_eigmin': float(eigmin),
            'sigma_ref_eigmin': float(sigma_ref_eigmin),
            'psd_ok': True}


def reliance(theta_lin, theta_quad, Sigma_diag, m):
    """g_1(R_tilde; mu_diag = N(m, Sigma_diag)).

    Population formula (with R_tilde = theta_lin . y + theta_quad * y[0]^2):
      Cov_mu(phi_1, R_tilde) = theta_lin . Sigma_diag[:, 0]
                               + theta_quad * 2 * m[0] * Sigma_diag[0, 0]
      Var_mu(phi_1)          = Sigma_diag[0, 0].
    """
    Sigma_col_target = Sigma_diag[:, TARGET_IDX]
    cov = (float(np.dot(theta_lin, Sigma_col_target))
           + 2.0 * theta_quad * m[TARGET_IDX] * Sigma_diag[TARGET_IDX, TARGET_IDX])
    var = float(Sigma_diag[TARGET_IDX, TARGET_IDX])
    return cov / var


def mitigate(theta_lin, theta_quad, g_1, c, target_idx=TARGET_IDX):
    """Single-axis mitigation: theta_lin[target_idx] -= c * g_1. Quadratic part unchanged."""
    theta_lin_mit = np.asarray(theta_lin, dtype=float).copy()
    theta_lin_mit[target_idx] = theta_lin_mit[target_idx] - c * g_1
    return theta_lin_mit, theta_quad


def dkl_gaussian_same_cov(mu1, mu2, Sigma):
    """D_KL(N(mu1, Sigma) || N(mu2, Sigma)) = 0.5 * (mu1 - mu2)^T Sigma^{-1} (mu1 - mu2)."""
    diff = np.asarray(mu1) - np.asarray(mu2)
    return 0.5 * float(diff @ np.linalg.solve(Sigma, diff))


def compute_outcomes(theta_lin, theta_quad, Sigma_ref, m_audit, beta_KL, c, w_true,
                     Sigma_diag=None, psd_tol=1e-10):
    """Full pipeline: reliance -> mitigation -> two policies -> deltas.

    Sigma_diag defaults to Sigma_ref (audit and reference share covariance).

    Returns dict with keys:
      g_1, theta_lin_mit, mu_pre, mu_post, Sigma_star,
      delta_1, delta_2, delta_3, delta_J, dkl,
      psd_ok, precision_eigmin, sigma_ref_eigmin.
    """
    if Sigma_diag is None:
        Sigma_diag = Sigma_ref

    sigma_ref_eigmin = float(np.linalg.eigvalsh(Sigma_ref).min())
    if sigma_ref_eigmin <= psd_tol:
        return {
            'g_1': float('nan'),
            'theta_lin_mit': [float('nan')] * 3,
            'mu_pre': None, 'mu_post': None, 'Sigma_star': None,
            'delta_1': float('nan'), 'delta_2': float('nan'),
            'delta_3': float('nan'), 'delta_J': float('nan'),
            'dkl': float('nan'),
            'psd_ok': False,
            'precision_eigmin': float('nan'),
            'sigma_ref_eigmin': sigma_ref_eigmin,
        }

    g_1 = reliance(theta_lin, theta_quad, Sigma_diag, np.asarray(m_audit, dtype=float))
    theta_lin_mit, theta_quad_mit = mitigate(theta_lin, theta_quad, g_1, c)

    pre = policy_optimum(theta_lin, theta_quad, Sigma_ref, beta_KL, psd_tol)
    post = policy_optimum(theta_lin_mit, theta_quad_mit, Sigma_ref, beta_KL, psd_tol)

    psd_ok = pre['psd_ok'] and post['psd_ok']
    precision_eigmin = min(pre['precision_eigmin'], post['precision_eigmin'])

    result = {
        'g_1': g_1,
        'theta_lin_mit': theta_lin_mit.tolist(),
        'psd_ok': bool(psd_ok),
        'precision_eigmin': float(precision_eigmin),
        'sigma_ref_eigmin': sigma_ref_eigmin,
    }

    if not psd_ok:
        result.update({
            'mu_pre': None, 'mu_post': None, 'Sigma_star': None,
            'delta_1': float('nan'), 'delta_2': float('nan'),
            'delta_3': float('nan'), 'delta_J': float('nan'),
            'dkl': float('nan'),
        })
        return result

    mu_pre = pre['mu']
    mu_post = post['mu']
    Sigma_star = pre['Sigma']  # H unchanged by mitigation; pre and post share Sigma*
    delta = mu_post - mu_pre
    delta_J = w_true * float(delta[STRUCTURAL_IDX])
    dkl = dkl_gaussian_same_cov(mu_post, mu_pre, Sigma_star)

    result.update({
        'mu_pre':     mu_pre.tolist(),
        'mu_post':    mu_post.tolist(),
        'Sigma_star': Sigma_star.tolist(),
        'delta_1':    float(delta[0]),
        'delta_2':    float(delta[1]),
        'delta_3':    float(delta[2]),
        'delta_J':    float(delta_J),
        'dkl':        float(dkl),
    })
    return result
