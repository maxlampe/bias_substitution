"""Pure-numpy logistic regression (BT-MLE).

Replaces sklearn.linear_model.LogisticRegression with a small Newton-Raphson /
IRLS solver. We need this because sklearn is not available in the environment.

Model: P(label=1 | x) = sigmoid(theta^T x). No intercept. L2 prior with strength
1/(2 C) acting on theta (so smaller C is stronger regularization, matching
sklearn's convention).

Newton-Raphson update with Hessian regularization:
    theta_{t+1} = theta_t + (X^T W X + I/C)^{-1} (X^T (y - p) - theta_t / C)
where p = sigmoid(X theta), W = diag(p (1 - p)).
"""

import numpy as np


def _sigmoid(z):
    # Numerically stable sigmoid
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_logistic_regression(X, y, C=1e3, max_iter=200, tol=1e-8, verbose=False):
    """Fit logistic regression with L2 regularization (no intercept).

    Parameters
    ----------
    X : (N, d) array
    y : (N,) array of {0, 1}
    C : float, inverse regularization strength (sklearn convention)
    max_iter : int
    tol : float, stopping tolerance on gradient infinity norm

    Returns
    -------
    theta : (d,) array
    info  : dict with 'n_iter', 'converged', 'final_grad_norm'
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N, d = X.shape
    theta = np.zeros(d)
    reg = 1.0 / C  # L2 strength
    converged = False
    grad_norm = np.inf
    for it in range(max_iter):
        z = X @ theta
        p = _sigmoid(z)
        grad = X.T @ (p - y) + reg * theta
        grad_norm = float(np.max(np.abs(grad)))
        if grad_norm < tol:
            converged = True
            break
        w = p * (1.0 - p)
        # Hessian: X^T diag(w) X + reg I
        XtwX = (X * w[:, None]).T @ X
        H = XtwX + reg * np.eye(d)
        # Damped Newton step. Solve H step = grad.
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        # Simple backtracking line search on log loss.
        new_theta = theta - step
        if not np.all(np.isfinite(new_theta)):
            # Fall back to a small gradient step.
            new_theta = theta - 1e-3 * grad
        theta = new_theta
        if verbose:
            print(f'  iter {it}: grad_norm={grad_norm:.3e}')
    return theta, {'n_iter': it + 1, 'converged': converged, 'final_grad_norm': grad_norm}
