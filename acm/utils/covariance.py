"""
Utility functions for covariance matrix sanity checks.
"""
import numpy as np
import warnings


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
        return True
    except np.linalg.LinAlgError:
        return False


def check_condition_number(matrix, precision_threshold: float = 10):
    """
    Computes the condition number of the matrix to check its inversibility.

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
    digits = -np.log10(cond) # Number of significant digits that can be trusted
    # print(f"Condition number: {cond}, Significant digits: {digits}")
    if cond >= 1: # Can't be trusted at all
        return 0
    elif digits < precision_threshold: # Ill-conditioned
        return 2
    else: # Well-conditioned
        return 1


def check_covariance_matrix(
    matrix: np.ndarray,
    name: str = "covariance",
    rtol: float = 1e-5,
    atol: float = 1e-8,
    precision_threshold: float = 10,
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
        warnings.warn(
            f"{name} matrix is not 2-dimensional (shape: {matrix.shape}). "
            "This may cause issues in likelihood analysis.",
            UserWarning,
            stacklevel=2
        )
        return False  # Can't proceed with other checks
    
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        warnings.warn(
            f"{name} matrix is not square (shape: {matrix.shape}). "
            "Covariance matrices must be square.",
            UserWarning,
            stacklevel=2
        )
        return False  # Can't proceed with other checks
    
    # Check if matrix is symmetric
    if not check_symmetric(matrix, rtol=rtol, atol=atol):
        warnings.warn(
            f"{name} matrix is not symmetric. "
            "Covariance matrices should be symmetric. "
            "This may indicate numerical issues or incorrect computation.",
            UserWarning,
            stacklevel=2
        )
        all_passed = False
    
    # Check if matrix is positive-definite
    if not check_positive_definite(matrix):
        # Get eigenvalues for more detailed diagnostics
        eigenvalues = np.linalg.eigvalsh(matrix)
        n_negative = np.sum(eigenvalues < 0)
        min_eigenvalue = np.min(eigenvalues)
        
        warnings.warn(
            f"{name} matrix is not positive-definite. "
            f"Found {n_negative} negative eigenvalue(s), minimum eigenvalue: {min_eigenvalue:.6e}. "
            "This will cause issues when inverting the matrix in likelihood analysis. "
            "Consider checking the mock realizations or increasing the number of samples.",
            UserWarning,
            stacklevel=2
        )
        all_passed = False
    
    # Check condition number
    cond_status = check_condition_number(matrix, precision_threshold=precision_threshold)
    if cond_status == 0:
        warnings.warn(
            f"{name} matrix is singular (ill-conditioned). "
            "This will cause issues when inverting the matrix in likelihood analysis. "
            "Using the diagonal covariance only may be a temporary workaround.",
            UserWarning,
            stacklevel=2
        )
        all_passed = False
    elif cond_status == 2:
        warnings.warn(
            f"{name} matrix is ill-conditioned. "
            "This may lead to unreliable results when inverting the matrix in likelihood analysis. "
            "Using the diagonal covariance only may be a temporary workaround.",
            UserWarning,
            stacklevel=2
        )
        all_passed = False
    
    return all_passed
