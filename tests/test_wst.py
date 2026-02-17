"""Unit tests for wavelet scattering transform using mockfactory lognormal mocks."""
import numpy as np  # type: ignore
import numpy.typing as npt  # type: ignore
import pytest  # type: ignore

from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform  # type: ignore


def create_lognormal_mock(
    boxsize: float = 500.0,
    nbar: float = 1e-3,
    nmesh: int = 128,
    bias: float = 2.0,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float64], float]:
    """
    Create a lognormal mock catalog using mockfactory.
    
    Parameters
    ----------
    boxsize : float, optional
        Box size in Mpc/h. Default is 500.0.
    nbar : float, optional
        Mean number density in (Mpc/h)^-3. Default is 1e-3.
    nmesh : int, optional
        Mesh size for the density field. Default is 128.
    bias : float, optional
        Galaxy bias. Default is 2.0.
    seed : int, optional
        Random seed. Default is 42.
        
    Returns
    -------
    positions : ndarray
        Galaxy positions with shape (N, 3).
    boxsize : float
        Box size in Mpc/h.
    """
    try:
        from mockfactory import LagrangianLinearMock, RandomBoxCatalog  # type: ignore
        from cosmoprimo.fiducial import DESI  # type: ignore
    except ImportError as e:
        pytest.skip(f"mockfactory or cosmoprimo not available: {e}")
    
    # Set up cosmology
    z = 0.5
    cosmo = DESI()
    power = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
    dist = cosmo.comoving_radial_distance(z)
    
    # Create mock with boxcenter at origin for simplicity
    boxcenter = [0.0, 0.0, 0.0]
    
    # Create Lagrangian (lognormal) mock
    mock = LagrangianLinearMock(
        power,
        nmesh=nmesh,
        boxsize=boxsize,
        boxcenter=boxcenter,
        seed=seed,
        unitary_amplitude=False
    )
    
    # Set the density field with Lagrangian bias (Eulerian bias - 1)
    mock.set_real_delta_field(bias=bias - 1)
    
    # Set the selection function
    mock.set_analytic_selection_function(nbar=nbar)
    
    # Sample galaxies from the density field
    mock.poisson_sample(seed=seed + 1)
    
    # Convert to catalog and extract positions
    data = mock.to_catalog()
    positions = np.array(data['Position'])
    
    return positions, boxsize


@pytest.mark.parametrize("backend", ["jaxpower", "pypower"])
def test_wst(backend: str) -> None:
    """
    Test the wavelet scattering transform with mockfactory lognormal mocks.
    
    This test creates a lognormal mock catalog on the fly using mockfactory
    and computes the WST coefficients. We verify that:
    1. The coefficients are computed without errors
    2. The coefficients are finite
    3. The number of coefficients matches expectations
    """
    # Create lognormal mock
    positions, boxsize = create_lognormal_mock(
        boxsize=500.0,
        nbar=1e-3,
        nmesh=128,
        bias=2.0,
        seed=42
    )
    
    # Set up WST parameters
    cellsize = 100.0  # Grid cell size in Mpc/h
    meshsize = (np.array([boxsize, boxsize, boxsize]) / cellsize).astype(int)
    
    # Initialize WaveletScatteringTransform
    wst = WaveletScatteringTransform(
        data_positions=positions,
        boxsize=boxsize,
        boxcenter=0.0,
        meshsize=meshsize,
        backend=backend,
        J=4,  # Number of scales
        L=4,  # Number of angles
        q=0.8,  # Power for the zeroth order coefficient
        sigma=0.8  # Wavelet scale
    )
    
    # Compute density contrast
    wst.set_density_contrast()
    
    # Run WST
    coeffs = wst.run()
    
    # Verify results
    assert coeffs is not None, "WST coefficients should not be None"
    assert np.all(np.isfinite(coeffs)), "All WST coefficients should be finite"
    assert len(coeffs) > 0, "WST should return at least one coefficient"
    
    # The number of coefficients depends on J and L
    # For J=4, L=4, max_order=2, we expect:
    # 1 (s0) + J*L (order 1) + J*(J-1)/2*L^2 (order 2) = 1 + 16 + 96 = 113
    expected_num_coeffs = 1 + 4 * 4 + 4 * 3 // 2 * 4 * 4
    assert len(coeffs) == expected_num_coeffs, \
        f"Expected {expected_num_coeffs} coefficients, got {len(coeffs)}"


def test_wst_consistency() -> None:
    """
    Test that WST produces consistent results across different random seeds.
    
    Different random seeds should produce different galaxy catalogs and thus
    different WST coefficients, but the general magnitude should be similar.
    """
    backend = "jaxpower"
    cellsize = 100.0
    boxsize = 500.0
    meshsize = (np.array([boxsize, boxsize, boxsize]) / cellsize).astype(int)
    
    coeffs_list = []
    for seed in [42, 43, 44]:
        positions, _ = create_lognormal_mock(
            boxsize=boxsize,
            nbar=1e-3,
            nmesh=128,
            bias=2.0,
            seed=seed
        )
        
        wst = WaveletScatteringTransform(
            data_positions=positions,
            boxsize=boxsize,
            boxcenter=0.0,
            meshsize=meshsize,
            backend=backend,
            J=4,
            L=4,
            q=0.8,
            sigma=0.8
        )
        
        wst.set_density_contrast()
        coeffs = wst.run()
        coeffs_list.append(coeffs)
    
    # Compute standard deviation across realizations
    coeffs_array = np.array(coeffs_list)
    mean_coeffs = np.mean(coeffs_array, axis=0)
    std_coeffs = np.std(coeffs_array, axis=0)
    
    # Check that there is some variation (cosmic variance)
    assert np.any(std_coeffs > 0), "WST should vary across different realizations"
    
    # Check that the variation is not too large (sanity check)
    # Coefficient of variation should be reasonable
    cv = std_coeffs / (np.abs(mean_coeffs) + 1e-10)
    assert np.median(cv) < 1.0, "Coefficient of variation should be reasonable"


def test_wst_small_box() -> None:
    """
    Test WST on a smaller box for faster execution.
    
    This test uses a smaller box to ensure the test suite runs quickly.
    """
    backend = "jaxpower"
    
    # Create a smaller mock
    positions, boxsize = create_lognormal_mock(
        boxsize=250.0,
        nbar=2e-3,
        nmesh=64,
        bias=1.5,
        seed=100
    )
    
    # Use coarser grid
    cellsize = 100.0
    meshsize = (np.array([boxsize, boxsize, boxsize]) / cellsize).astype(int)
    
    wst = WaveletScatteringTransform(
        data_positions=positions,
        boxsize=boxsize,
        boxcenter=0.0,
        meshsize=meshsize,
        backend=backend,
        J=3,  # Fewer scales for faster computation
        L=3,
        q=0.8,
        sigma=0.8
    )
    
    wst.set_density_contrast()
    coeffs = wst.run()
    
    # Basic sanity checks
    assert len(coeffs) > 0
    assert np.all(np.isfinite(coeffs))
    
    # Expected number of coefficients for J=3, L=3
    # 1 + J*L + J*(J-1)/2*L^2 = 1 + 9 + 27 = 37
    expected_num_coeffs = 1 + 3 * 3 + 3 * 2 // 2 * 3 * 3
    assert len(coeffs) == expected_num_coeffs


def test_wst_different_bias() -> None:
    """
    Test that WST coefficients change with different bias values.
    
    Higher bias should generally lead to stronger clustering and thus
    different WST coefficients.
    """
    backend = "jaxpower"
    cellsize = 100.0
    boxsize = 500.0
    meshsize = (np.array([boxsize, boxsize, boxsize]) / cellsize).astype(int)
    
    coeffs_low_bias = None
    coeffs_high_bias = None
    
    for bias in [1.5, 3.0]:
        positions, _ = create_lognormal_mock(
            boxsize=boxsize,
            nbar=1e-3,
            nmesh=128,
            bias=bias,
            seed=42  # Same seed for fair comparison
        )
        
        wst = WaveletScatteringTransform(
            data_positions=positions,
            boxsize=boxsize,
            boxcenter=0.0,
            meshsize=meshsize,
            backend=backend,
            J=4,
            L=4,
            q=0.8,
            sigma=0.8
        )
        
        wst.set_density_contrast()
        coeffs = wst.run()
        
        if bias == 1.5:
            coeffs_low_bias = coeffs
        else:
            coeffs_high_bias = coeffs
    
    # Coefficients should be different for different bias values
    assert not np.allclose(coeffs_low_bias, coeffs_high_bias), \
        "WST coefficients should differ for different bias values"
    
    # Higher bias should generally lead to larger coefficients
    # (though this is not guaranteed for all coefficients)
    mean_low = np.mean(np.abs(coeffs_low_bias))
    mean_high = np.mean(np.abs(coeffs_high_bias))
    assert mean_high > mean_low, \
        "Higher bias should generally lead to larger WST coefficients"


if __name__ == "__main__":
    # Run a simple test
    print("Running basic WST test with mockfactory...")
    positions, boxsize = create_lognormal_mock()
    print(f"Created mock catalog with {len(positions)} galaxies in {boxsize} Mpc/h box")
    
    cellsize = 100.0
    meshsize = (np.array([boxsize, boxsize, boxsize]) / cellsize).astype(int)
    
    wst = WaveletScatteringTransform(
        data_positions=positions,
        boxsize=boxsize,
        boxcenter=0.0,
        meshsize=meshsize,
        backend="jaxpower",
        J=4,
        L=4,
        q=0.8,
        sigma=0.8
    )
    
    wst.set_density_contrast()
    coeffs = wst.run()
    print(f"Computed {len(coeffs)} WST coefficients")
    print(f"First 10 coefficients: {coeffs[:10]}")
    print("Test passed!")
