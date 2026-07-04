from unittest.mock import MagicMock

import numpy as np
import pytest

from acm.estimators.galaxy_clustering.backends.filters import (
    GaussianFilter,
    NoFilter,
    TopHatFilter,
)
from acm.estimators.galaxy_clustering.backends.pypower import (
    PypowerBackend,
)

# ruff: noqa: ANN001, ANN201, ARG001, D101, D102, D103, INP001, S101

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
    return np.ones(N)

@pytest.fixture
def rand_w(rand_pos):
    return np.ones(M)

@pytest.fixture
def backend(data_pos):
    return PypowerBackend(data_pos)

@pytest.fixture
def backend_with_randoms(data_pos, rand_pos, data_w, rand_w):
    return PypowerBackend(data_pos, rand_pos, data_w, rand_w)

#%% Test classes

class TestEstimatorBackendValidation:
    def test_invalid_data_positions_shape(self):
        with pytest.raises(ValueError, match=r"Positions must be of shape (N, 3)"):
            PypowerBackend(np.ones((10, 2)))

    def test_invalid_randoms_positions_shape(self, data_pos):
        with pytest.raises(ValueError, match=r"Positions must be of shape (N, 3)"):
            PypowerBackend(data_pos, randoms_positions=np.ones((10, 2)))
    
    def test_invalid_data_weight_shape(self, data_pos):
        with pytest.raises(ValueError, match="Weights must be 1D"):
            PypowerBackend(data_pos, data_weights=np.ones((N, 1)))

    def test_invalid_data_weights_length(self, data_pos):
        with pytest.raises(ValueError, match="Weights must have the same length as positions"):
            PypowerBackend(data_pos, data_weights=np.ones(N + 1))

    def test_randoms_weights_without_randoms_raises(self, data_pos, rand_w):
        with pytest.raises(ValueError, match="randoms_weights requires"):
            PypowerBackend(data_pos, randoms_weights=rand_w)
    
    def test_invalid_randoms_weights_shape(self, data_pos, rand_pos):
        with pytest.raises(ValueError, match="Weights must be 1D"):
            PypowerBackend(data_pos, rand_pos, randoms_weights=np.ones((M, 1)))

    def test_invalid_randoms_weights_length(self, data_pos, rand_pos):
        with pytest.raises(ValueError, match="Weights must have the same length as positions"):
            PypowerBackend(data_pos, rand_pos, randoms_weights=np.ones(M + 1))

    def test_size_data(self, backend, data_pos):
        assert backend.size_data == len(data_pos)

    def test_size_randoms(self, backend_with_randoms, rand_pos):
        assert backend_with_randoms.size_randoms == len(rand_pos)

    def test_size_randoms_raises_when_not_set(self, backend):
        with pytest.raises(ValueError, match="Randoms have not been set"):
            _ = backend.size_randoms

class TestPypowerBackendProperties:
    def test_boxsize(self, backend):
        np.testing.assert_array_equal(backend.boxsize, [100.0, 100.0, 100.0])

    def test_boxcenter(self, backend):
        np.testing.assert_array_equal(backend.boxcenter, [50.0, 50.0, 50.0])

    def test_meshsize(self, backend):
        np.testing.assert_array_equal(backend.meshsize, [32, 32, 32])

    def test_cellsize(self, backend):
        np.testing.assert_array_almost_equal(backend.cellsize, [100 / 32] * 3)

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
        backend._density_contrast.readout.assert_called_once_with(data_pos, resampler="cic")

    def test_set_with_randoms(self, backend_with_randoms):
        backend_with_randoms.set_density_contrast()
        assert backend_with_randoms._density_contrast is not None

    def test_set_with_smoothing(self, backend):
        backend.set_density_contrast(smoothing_radius=10.0, filter_shape="Gaussian")
        assert backend._density_contrast is not None

class TestGetKernel:
    def test_gaussian(self):
        assert isinstance(PypowerBackend._get_kernel("Gaussian", 5.0), GaussianFilter)

    def test_tophat(self):
        assert isinstance(PypowerBackend._get_kernel("TopHat", 5.0), TopHatFilter)

    def test_nofilter(self):
        assert isinstance(PypowerBackend._get_kernel("NoFilter", 0.0), NoFilter)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            PypowerBackend._get_kernel("Invalid", 5.0)

class TestGetQueryPositions:
    def test_randoms_shape(self, backend):
        coords = backend.get_query_positions(method="randoms", nquery=50)
        assert coords.shape == (50, 3)

    def test_randoms_default_nquery(self, backend):
        coords = backend.get_query_positions(method="randoms")
        assert coords.shape == (5 * N, 3)

    def test_lattice_shape(self, backend):
        coords = backend.get_query_positions(method="lattice")
        assert coords.shape == (np.prod(backend.meshsize), 3)

    def test_seed_reproducibility(self, backend):
        c1 = backend.get_query_positions(method="randoms", nquery=20, seed=7)
        c2 = backend.get_query_positions(method="randoms", nquery=20, seed=7)
        np.testing.assert_array_equal(c1, c2)

    def test_invalid_method_raises(self, backend):
        with pytest.raises(ValueError, match="method"):
            backend.get_query_positions(method="invalid")
