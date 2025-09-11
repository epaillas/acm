import numpy as np

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