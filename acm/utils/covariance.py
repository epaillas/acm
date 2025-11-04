import numpy as np
from scipy.stats import norm


def get_covariance_correction(n_s, n_d, n_theta=None, method='percival'):
    """
    Correction factor to debias de inverse covariance matrix.

    Args:
        n_s (int): Number of simulations.
        n_d (int): Number of bins of the data vector.
        n_theta (int): Number of free parameters.
        method (str): Method to compute the correction factor.

    Returns:
        float: Correction factor
    """
    if method == 'percival':
        B = (n_s - n_d - 2) / ((n_s - n_d - 1)*(n_s - n_d - 4))
        return (n_s - 1)*(1 + B*(n_d - n_theta))/(n_s - n_d + n_theta - 1)
    elif method == 'percival-fisher':
        return (n_s - 1)/(n_s - n_d + n_theta - 1)
    elif method == 'hartlap':
        return (n_s - 1)/(n_s - n_d - 2)
    else:
        raise ValueError(f"Unknown method: {method}. Available methods are: 'percival', 'percival-fisher', 'hartlap'.")

def correlation_from_covariance(covariance):
    """
    Compute the correlation matrix from the covariance matrix.

    Parameters
    ----------
    covariance : array_like
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Correlation matrix.
    """
    
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def mad_1d(x, axis=None, keepdims=False):
    """Median absolute deviation with Gaussian-consistent scaling."""
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    mad = mad * 1/norm.ppf(3/4)
    if not keepdims:
        mad = np.squeeze(mad, axis=axis)
    return mad

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

    X = np.asarray(residuals)
    n_bins = X.shape[1]

    # debias the residuals
    X -= np.median(X, axis=0, keepdims=True)

    # Robust variances on each bin
    s = mad_1d(X, axis=0)
    s2 = np.maximum(s**2, eps)

    # Pairwise robust covariance via GK:
    # cov(X,Y) ≈ (1/4)[ Var(X+Y) - Var(X-Y) ], with Var estimated by MAD^2.
    C = np.empty((n_bins, n_bins), dtype=X.dtype)
    for j in range(n_bins):
        C[j, j] = s2[j]
        for k in range(j+1, n_bins):
            sp = mad_1d(X[:, j] + X[:, k])**2
            sm = mad_1d(X[:, j] - X[:, k])**2
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
    X = np.asarray(residuals)
    n_bins = X.shape[1]

    # Debias the residuals
    X -= np.median(X, axis=0, keepdims=True)

    # Robust scales per bin
    s = mad_1d(X, axis=0)
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
    tau = mad_1d(U, axis=0)**2
    tau = np.clip(tau, eps, None)

    # Assemble covariance in original units: C = S Q diag(tau) Q^T S
    C = (evecs * tau) @ evecs.T           # Q diag(tau) Q^T
    C = (C * s).T * s                     # S (...) S
    C = 0.5 * (C + C.T)                   # symmetrize
    return C