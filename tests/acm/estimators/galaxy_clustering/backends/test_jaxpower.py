from unittest.mock import MagicMock

import numpy as np
import pytest

from acm.estimators.galaxy_clustering.backends.jaxpower import (
    JaxpowerBackend,
)

# ruff: noqa: ANN001, ANN201, D101, D102, D103, INP001, S101

#%% Fixtures
N, M = 20, 30

@pytest.fixture
def data_pos():
    return np.random.default_rng(0).uniform(0, 100, (N, 3))

@pytest.fixture
def rand_pos():
    return np.random.default_rng(1).uniform(0, 100, (M, 3))

@pytest.fixture
def data_w(data_pos):
    return np.ones(len(data_pos))

@pytest.fixture
def rand_w(rand_pos):
    return np.ones(len(rand_pos))

@pytest.fixture
def backend(data_pos):
    b = JaxpowerBackend(data_pos)
    return b

@pytest.fixture
def backend_with_randoms(data_pos, rand_pos, data_w, rand_w):
    b = JaxpowerBackend(data_pos, rand_pos, data_w, rand_w)
    return b


#%% test classes

class TestEstimatorBackendValidation:
    def test_invalid_data_positions_shape(self):
        with pytest.raises(ValueError, match="data_positions"):
            JaxpowerBackend(np.ones((10, 2)))

    def test_invalid_randoms_positions_shape(self, data_pos):
        with pytest.raises(ValueError, match="randoms_positions"):
            JaxpowerBackend(data_pos, randoms_positions=np.ones((10, 2)))

    def test_invalid_data_weights_length(self, data_pos):
        with pytest.raises(ValueError, match="data_weights"):
            JaxpowerBackend(data_pos, data_weights=np.ones(N + 1))

    def test_randoms_weights_without_randoms_raises(self, data_pos, rand_w):
        with pytest.raises(ValueError, match="randoms_weights requires"):
            JaxpowerBackend(data_pos, randoms_weights=rand_w)

    def test_invalid_randoms_weights_length(self, data_pos, rand_pos):
        with pytest.raises(ValueError, match="randoms_weights"):
            JaxpowerBackend(data_pos, rand_pos, randoms_weights=np.ones(M + 1))

    def test_size_data(self, backend, data_pos):
        assert backend.size_data == len(data_pos)

    def test_size_randoms(self, backend_with_randoms, rand_pos):
        assert backend_with_randoms.size_randoms == len(rand_pos)

    def test_size_randoms_raises_when_not_set(self, backend):
        with pytest.raises(ValueError, match="Randoms have not been set"):
            _ = backend.size_randoms

class TestJaxpowerBackendProperties:
    def test_boxsize_delegates_to_mattrs(self, backend):
        assert backend.boxsize is backend.mattrs.boxsize

    def test_boxcenter_delegates_to_mattrs(self, backend):
        assert backend.boxcenter is backend.mattrs.boxcenter

    def test_meshsize_delegates_to_mattrs(self, backend):
        assert backend.meshsize is backend.mattrs.meshsize

    def test_cellsize_delegates_to_mattrs(self, backend):
        assert backend.cellsize is backend.mattrs.cellsize

class TestDensityContrast:
    def test_read_before_set_raises(self, backend, data_pos):
        with pytest.raises(AttributeError, match="set_density_contrast"):
            backend.read_density_contrast(data_pos)

    def test_set_stores_density_contrast(self, backend):
        backend.set_density_contrast()
        assert backend._density_contrast is not None

    def test_read_after_set_calls_read(self, backend, data_pos):
        backend.set_density_contrast()
        backend._density_contrast = MagicMock()
        backend.read_density_contrast(data_pos, resampler="cic")
        backend._density_contrast.read.assert_called_once_with(data_pos, resampler="cic")

    def test_set_with_randoms(self, backend_with_randoms):
        """FKP path (data + randoms) should complete without error."""
        backend_with_randoms.set_density_contrast()
        assert backend_with_randoms._density_contrast is not None

    def test_set_with_smoothing(self, backend):
        backend.set_density_contrast(smoothing_radius=5.0)
        assert backend._density_contrast is not None

class TestGaussianKernel:
    def test_returns_jax_array(self, backend):
        """gaussian_kernel should call jax.numpy.exp and return its result."""
        s = 5.0
        x = np.linspace(0, 10, 10)
        kernel = JaxpowerBackend.gaussian_kernel(backend.mattrs, smoothing_radius=s)
        result = np.exp(-0.5 * sum(x * s**2))
        assert kernel == result

class TestGetFieldThreshold:
    def test_noise_method(self):
        field = MagicMock()
        field.weights = np.array([0.1, 0.2, 0.3]) # Dummy array to avoid jax error
        JaxpowerBackend._get_field_threshold(field, threshold=0.01, method="noise")

    def test_mean_method(self):
        field = MagicMock()
        JaxpowerBackend._get_field_threshold(field, threshold=0.01, method="mean")

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            JaxpowerBackend._get_field_threshold(MagicMock(), method="invalid")

class TestGetQueryPositions:
    def test_randoms_method_shape(self, backend):
        coords = backend.get_query_positions(method="randoms", nquery=50, seed=0)
        assert coords.shape == (50, 3)

    def test_randoms_default_nquery(self, backend):
        coords = backend.get_query_positions(method="randoms")
        assert coords.shape == (5 * N, 3)

    def test_lattice_method(self, backend):
        coords = backend.get_query_positions(method="lattice")
        assert coords is not None

    def test_invalid_method_raises(self, backend):
        with pytest.raises(ValueError, match="method"):
            backend.get_query_positions(method="invalid")

    def test_seed_reproducibility(self, backend):
        c1 = backend.get_query_positions(method="randoms", nquery=20, seed=7)
        c2 = backend.get_query_positions(method="randoms", nquery=20, seed=7)
        np.testing.assert_array_equal(c1, c2)
