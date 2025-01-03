import jax
import jax.numpy as jnp
from typing import Tuple

def generalized_eigh(A, B):
    L = jnp.linalg.cholesky(B)
    L_inv = jnp.linalg.inv(L)
    C = L_inv @ A @ L_inv.T
    eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C)
    eigenvectors_original = L_inv.T @ eigenvectors_transformed
    return eigenvalues, eigenvectors_original

# Equation 18 from arXiv:2409.02102
def compute_cca_compression(
    data: jnp.ndarray,
    params: jnp.ndarray,
    data_covariance: jnp.ndarray,
):
    """Compute CCA compression assuming parameter-independent covariance.
    
    Maximizes mutual information between parameters and compressed data by solving:
        max_v (v^T C_t v) / (v^T (C_t - C_l)v)
    
    where:
    - C_t is the fixed covariance matrix from parameter-fixed simulations
    - C_l = C_tp C_p^{-1} C_pt is projected parameter covariance
    - C_tp = E[(t - t̄)(p - p̄)^T] is data-parameter cross-covariance
    - C_p = E[(p - p̄)(p - p̄)^T] is parameter covariance
    
    This leads to the generalized eigenvalue problem:
        C_t v_i = ρ_i(C_t - C_l)v_i
    
    Args:
        data: Clean data vectors evaluated at parameter points (n_samples, n_features)
        params: Parameter values (n_samples, n_params)
        fixed_cov: Covariance matrix from fixed-parameter sims (n_features, n_features)

    Returns:
        compression_matrix: Matrix whose columns are compression vectors (n_features, n_features)
        eigenvalues: Eigenvalues sorted in descending order (n_features,)

    """
    data_mean = jnp.mean(data, axis=0)
    param_mean = jnp.mean(params, axis=0)
    data_centered = data - data_mean
    param_centered = params - param_mean
    n_samples = data.shape[0]

    Ct = data_covariance
    Cp = (param_centered.T @ param_centered) / n_samples
    # Note we assume that the covariance of the data is independent of the parameters
    Ctp = (data_centered.T @ param_centered) / n_samples
    Cl = Ctp @ jnp.linalg.inv(Cp) @ Ctp.T
    eigenvals, eigenvecs = generalized_eigh(Ct, Ct - Cl)
    idx = jnp.argsort(eigenvals)[::-1]
    
    return eigenvecs[:, idx], eigenvals[idx]

def apply_compression(data: jnp.ndarray, compression_matrix: jnp.ndarray, n_components: int = None) -> jnp.ndarray:
    """Apply CCA compression to data vectors.
    
    Args:
        data: Data vectors to compress (n_samples, n_features)
        compression_matrix: Compression vectors from compute_cca_compression (n_features, n_features)
        n_components: Number of components to keep (default: all)
    
    Returns:
        Compressed data vectors (n_samples, n_components)
    """
    if n_components is None:
        n_components = compression_matrix.shape[1]
    return data @ compression_matrix[:, :n_components]
