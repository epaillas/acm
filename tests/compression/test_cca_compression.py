import jax
import jax.numpy as jnp

from acm.compression.cca import compute_cca_compression, apply_compression

def test_simple_linear_case():
    """Test CCA with data t = ap + n where:
    - a is a known direction in feature space
    - p is a single parameter
    - n is Gaussian noise
    
    The optimal compression should recover direction a.
    """
    # Set random seed
    key = jax.random.PRNGKey(0)
    
    # Setup
    n_samples = 1000
    n_features = 3 
    
    # True direction in feature space
    a = jnp.array([0.1, 0.6,  0.3,]) 
    
    # Generate parameters
    key, subkey = jax.random.split(key)
    params = jax.random.normal(subkey, (n_samples, 1))
    
    # Generate clean data
    data = params @ a[None, :]
    
    # Fixed noise covariance (diagonal for simplicity)
    noise_scale = 1.
    fixed_cov = jnp.eye(n_features) * noise_scale**2
    
    # Compute compression
    compression_matrix, eigenvals = compute_cca_compression(data, params, fixed_cov)
    
    # First eigenvector should align with true direction
    v1 = compression_matrix[:, 0]
    alignment = jnp.abs(jnp.dot(v1, a)/jnp.linalg.norm(v1)/jnp.linalg.norm(a))
    assert alignment > 0.99, "First component should align with true direction"
    
