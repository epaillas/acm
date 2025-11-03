import numpy as np



def gk_mad_covariance(residuals, eps=1e-12):
    """
    Calculate the emulator covariance via Gnanadesikan–Kettenring
    (pairwise MAD). Fast but may not be strictly positive-definite.
    Section 4.2 of https://arxiv.org/abs/1810.02467

    Parameters
    ----------
    residuals : np.array with emulator residuals.

    Returns
    -------
    C : np.ndarray
        Covariance matrix of the residuals.
    """
    def _mad_1d(x, axis=None, keepdims=False):
        """Median absolute deviation with Gaussian-consistent scaling."""
        med = np.median(x, axis=axis, keepdims=True)
        mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
        mad = 1.4826 * mad
        if not keepdims:
            mad = np.squeeze(mad, axis=axis)
        return mad

    X = np.asarray(residuals)
    n_bins = X.shape[1]

    # debias the residuals
    X -= np.median(X, axis=0, keepdims=True)

    # Robust variances on each bin
    s = _mad_1d(X, axis=0)
    s2 = np.maximum(s**2, eps)

    # Pairwise robust covariance via GK:
    # cov(X,Y) ≈ (1/4)[ Var(X+Y) - Var(X-Y) ], with Var estimated by MAD^2.
    C = np.empty((n_bins, n_bins), dtype=X.dtype)
    for j in range(n_bins):
        C[j, j] = s2[j]
        for k in range(j+1, n_bins):
            sp = _mad_1d(X[:, j] + X[:, k])**2
            sm = _mad_1d(X[:, j] - X[:, k])**2
            cov_jk = 0.25 * (sp - sm)
            C[j, k] = C[k, j] = cov_jk
    return C

def orthogonal_gk_mad_covariance(residuals, eps=1e-12):
    """
    Emulator covariance through the Orthogonalized Gnanadesikan-Kettenring
    estimator, which is usually positive-definite and better conditioned
    than plain GK. Section 4.4 of https://arxiv.org/abs/1810.02467

    Parameters
    ----------
    residuals : np.array with the emulator residuals.

    Returns
    -------
    C : np.ndarray
        Covariance matrix of the residuals.
    """
    def _mad_1d(x, axis=None, keepdims=False):
        """Median absolute deviation with Gaussian-consistent scaling."""
        med = np.median(x, axis=axis, keepdims=True)
        mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
        mad = 1.4826 * mad
        if not keepdims:
            mad = np.squeeze(mad, axis=axis)
        return mad

    X = np.asarray(residuals)
    n_bins = X.shape[1]

    # Debias the residuals
    X -= np.median(X, axis=0, keepdims=True)

    # Robust scales per bin
    s = _mad_1d(X, axis=0)
    s = np.where(s <= 0, np.sqrt(eps), s)
    S_inv = 1.0 / s

    # Standardize
    Z = X * S_inv  # broadcasting

    # Robust correlation via GK on standardized variables
    R = gk_mad_covariance(Z)
    # Numerical cleanup
    R = 0.5 * (R + R.T)
    # Ensure diagonals = 1 (guard numeric drift)
    d = np.sqrt(np.clip(np.diag(R), eps, None))
    R = (R / d).T / d  # R = D^{-1} R D^{-1}

    # Eigendecomposition
    evals, evecs = np.linalg.eigh(R)

    # Rotate standardized data
    U = Z @ evecs

    # Robust variances along orthogonal directions
    tau = _mad_1d(U, axis=0)**2
    tau = np.clip(tau, eps, None)

    # Assemble covariance in original units: C = S Q diag(tau) Q^T S
    C = (evecs * tau) @ evecs.T           # Q diag(tau) Q^T
    C = (C * s).T * s                     # S (...) S
    C = 0.5 * (C + C.T)                   # symmetrize
    return C