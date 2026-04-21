"""Utility functions for covariance matrix & sanity checks."""

import warnings

import numpy as np
from scipy.stats import norm


# %% Covariance utils methods
def get_covariance_correction(
    n_s: int,
    n_d: int,
    n_theta: int | None = None,
    method: str = "percival",
) -> float:
    """
    Correction factor to debias the inverse covariance matrix.

    Parameters
    ----------
    n_s: int
        Number of simulations.
    n_d: int
        Number of bins of the data vector.
    n_theta: int, optional
        Number of free parameters.
    method: str, optional
        Method to compute the correction factor. Default to "percival".
        Available methods are: "percival", "percival-fisher", "hartlap".

    Returns
    -------
    float
        Correction factor
    """
    if method == "percival" and n_theta is not None:
        B = (n_s - n_d - 2) / ((n_s - n_d - 1) * (n_s - n_d - 4))
        return (n_s - 1) * (1 + B * (n_d - n_theta)) / (n_s - n_d + n_theta - 1)
    if method == "percival-fisher" and n_theta is not None:
        return (n_s - 1) / (n_s - n_d + n_theta - 1)
    if method == "hartlap":
        return (n_s - 1) / (n_s - n_d - 2)
    raise ValueError(
        f"Unknown method: {method}. Available methods are: 'percival', 'percival-fisher', 'hartlap'."
    )


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
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


def mad_1d(
    x: np.ndarray, axis: int | None = None, keepdims: bool = False
) -> np.ndarray:
    """Median absolute deviation with Gaussian-consistent scaling."""
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    mad = mad * 1 / norm.ppf(3 / 4)
    if not keepdims:
        mad = np.squeeze(mad, axis=axis)
    return mad


def gk_mad_covariance(residuals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Calculate the emulator covariance via Gnanadesikan-Kettenring (pairwise MAD).

    Fast but may not be strictly positive-definite.
    Section 4.2 of https://arxiv.org/abs/1810.02467

    Parameters
    ----------
    residuals : np.array
        Emulator residuals.
    eps : float, optional
        Precision parameter to avoid division by zero. Default is 1e-12.

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
        for k in range(j + 1, n_bins):
            sp = mad_1d(X[:, j] + X[:, k]) ** 2
            sm = mad_1d(X[:, j] - X[:, k]) ** 2
            cov_jk = 0.25 * (sp - sm)
            C[j, k] = C[k, j] = cov_jk
    return C


def orthogonal_gk_mad_covariance(
    residuals: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """
    Emulator covariance through the Orthogonalized Gnanadesikan-Kettenring estimator.

    It is usually positive-definite and better conditioned
    than plain GK. Section 4.4 of https://arxiv.org/abs/1810.02467

    Parameters
    ----------
    residuals : np.array
        Emulator residuals.
    eps : float, optional
        Precision parameter to avoid division by zero. Default is 1e-12.

    Returns
    -------
    C : np.ndarray
        Covariance matrix of the residuals.
    """
    X = np.asarray(residuals)
    # n_bins = X.shape[1]

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
    _, evecs = np.linalg.eigh(R)

    # Rotate standardized data
    U = Z @ evecs

    # Robust variances along orthogonal directions
    tau = mad_1d(U, axis=0) ** 2
    tau = np.clip(tau, eps, None)

    # Assemble covariance in original units: C = S Q diag(tau) Q^T S
    C = (evecs * tau) @ evecs.T  # Q diag(tau) Q^T
    C = (C * s).T * s  # S (...) S
    C = 0.5 * (C + C.T)  # symmetrize
    return C


# %% Covariance sanity checks methods
def check_symmetric(matrix: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Check if a matrix is symmetric.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to check for symmetry.
    rtol : float, optional
        Relative tolerance for the symmetry check. Default is 1e-5.
    atol : float, optional
        Absolute tolerance for the symmetry check. Default is 1e-8.

    Returns
    -------
    bool
        True if the matrix is symmetric, False otherwise.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)


def check_positive_definite(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is positive-definite by attempting Cholesky decomposition.

    A matrix is positive-definite if all its eigenvalues are positive.
    This is equivalent to checking if the Cholesky decomposition exists.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to check for positive-definiteness.

    Returns
    -------
    bool
        True if the matrix is positive-definite, False otherwise.
    """
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    else:
        return True


def check_condition_number(matrix: np.ndarray, precision_threshold: float = 10) -> int:
    """
    Compute the condition number of the matrix to check its inversibility.

    Parameters
    ----------
    matrix : array-like
        The matrix to check.
    precision_threshold : float, optional
        The threshold for the number of significant digits below which the matrix is considered ill-conditioned. Default is 10.

    Returns
    -------
    int
        0 if the matrix is singular,
        1 if the matrix is well-conditioned,
        2 if the matrix is ill-conditioned (number of significant digits < precision_threshold).

    Notes
    -----
    For a condition number with an order of magnitude of 10^k, the precision is roughly reduced by k digits.
    1. If the condition number is very large (>= 1/eps), the matrix is singular: No significant digits can be trusted.
    2. If the number of significant digits is less than precision_threshold, the matrix is ill-conditioned.
    3. Otherwise, the matrix is well-conditioned.
    """
    cond = np.linalg.cond(matrix)
    eps = np.finfo(float).eps
    cond *= eps
    digits = -np.log10(cond)  # Number of significant digits that can be trusted
    # print(f"Condition number: {cond}, Significant digits: {digits}")
    if cond >= 1:  # Can't be trusted at all
        return 0
    if digits < precision_threshold:  # Ill-conditioned
        return 2
    # Well-conditioned
    return 1


def check_covariance_matrix(
    matrix: np.ndarray,
    name: str = "covariance",
    rtol: float = 1e-5,
    atol: float = 1e-8,
    precision_threshold: float = 10,
    raise_warnings: bool = True,
) -> bool:
    """
    Perform sanity checks on a covariance matrix and raise warnings if checks fail.

    This function checks that the matrix is:
    1. 2-dimensional
    2. Square
    3. Symmetric
    4. Positive-definite

    Parameters
    ----------
    matrix : np.ndarray
        The covariance matrix to check.
    name : str, optional
        Name of the matrix for warning messages. Default is "covariance".
    rtol : float, optional
        Relative tolerance for the symmetry check. Default is 1e-5.
    atol : float, optional
        Absolute tolerance for the symmetry check. Default is 1e-8.
    precision_threshold : float, optional
        Threshold for the number of significant digits below which the matrix is considered ill-conditioned.
        Default is 10.

    Returns
    -------
    bool
        True if all checks pass, False otherwise.

    Warnings
    --------
    UserWarning
        If any of the checks fail, a warning is raised with details.
    """
    all_passed = True

    # Check if matrix is 2D
    if matrix.ndim != 2:
        if raise_warnings:
            warnings.warn(
                f"{name} matrix is not 2-dimensional (shape: {matrix.shape}). "
                "This may cause issues in likelihood analysis.",
                UserWarning,
                stacklevel=2,
            )
        return False  # Can't proceed with other checks

    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        if raise_warnings:
            warnings.warn(
                f"{name} matrix is not square (shape: {matrix.shape}). "
                "Covariance matrices must be square.",
                UserWarning,
                stacklevel=2,
            )
        return False  # Can't proceed with other checks

    # Check if matrix is symmetric
    if not check_symmetric(matrix, rtol=rtol, atol=atol):
        if raise_warnings:
            warnings.warn(
                f"{name} matrix is not symmetric. "
                "Covariance matrices should be symmetric. "
                "This may indicate numerical issues or incorrect computation.",
                UserWarning,
                stacklevel=2,
            )
        all_passed = False

    # Check if matrix is positive-definite
    if not check_positive_definite(matrix):
        # Get eigenvalues for more detailed diagnostics
        eigenvalues = np.linalg.eigvalsh(matrix)
        n_negative = np.sum(eigenvalues < 0)
        min_eigenvalue = np.min(eigenvalues)

        if raise_warnings:
            warnings.warn(
                f"{name} matrix is not positive-definite. "
                f"Found {n_negative} negative eigenvalue(s), minimum eigenvalue: {min_eigenvalue:.6e}. "
                "This will cause issues when inverting the matrix in likelihood analysis. "
                "Consider checking the mock realizations or increasing the number of samples.",
                UserWarning,
                stacklevel=2,
            )
        all_passed = False

    # Check condition number
    cond_status = check_condition_number(
        matrix, precision_threshold=precision_threshold
    )
    if cond_status == 0:
        if raise_warnings:
            warnings.warn(
                f"{name} matrix is singular (ill-conditioned). "
                "This will cause issues when inverting the matrix in likelihood analysis. "
                "Using the diagonal covariance only may be a temporary workaround.",
                UserWarning,
                stacklevel=2,
            )
        all_passed = False
    elif cond_status == 2:
        if raise_warnings:
            warnings.warn(
                f"{name} matrix is ill-conditioned. "
                "This may lead to unreliable results when inverting the matrix in likelihood analysis. "
                "Using the diagonal covariance only may be a temporary workaround.",
                UserWarning,
                stacklevel=2,
            )
        all_passed = False

    return all_passed
