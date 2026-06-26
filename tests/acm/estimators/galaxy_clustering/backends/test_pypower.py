import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# ruff: noqa: ANN001, ANN201, ANN204, ARG001, ARG002, D101, D102, D103, D105, E402, INP001, S101

#%% Mock modules-level imports

pypower_mock = MagicMock()
class MeshField:
    """Minimal mesh field sentinel supporting the r2c/apply/c2r chain and arithmetic."""

    def __init__(self, value=None):
        self.value = value if value is not None else np.ones((32, 32, 32))

    def r2c(self): return self
    def c2r(self): return self
    def apply(self, kernel): return self
    def readout(self, positions, resampler="cic"): return np.zeros(len(positions))

    def __sub__(self, other): return MeshField(self.value - (other.value if isinstance(other, MeshField) else other))
    def __truediv__(self, other): return MeshField(self.value / (other.value if isinstance(other, MeshField) else other))
    def __mul__(self, other): return MeshField(self.value * (other.value if isinstance(other, MeshField) else other))
    def __rmul__(self, other): return MeshField(self.value * other)
    def __gt__(self, other): return self.value > other
    def __invert__(self): return ~(self.value > 0)
    def __setitem__(self, key, value): self.value[key] = value
    def __getitem__(self, key): return self.value[key]


class CatalogMesh:
    """Minimal CatalogMesh sentinel."""

    def __init__(self, data_positions, data_weights=None, randoms_positions=None, randoms_weights=None, **kwargs):
        self.data_positions = data_positions
        self.data_weights = data_weights if data_weights is not None else np.ones(len(data_positions))
        self.boxsize = np.array([100.0, 100.0, 100.0])
        self.boxcenter = np.array([50.0, 50.0, 50.0])
        self.nmesh = np.array([32, 32, 32])
        self.with_randoms = randoms_positions is not None

    def to_mesh(self, **kwargs):
        return MeshField()

pypower_mock.CatalogMesh = CatalogMesh
sys.modules["pypower"] = pypower_mock

from acm.estimators.galaxy_clustering.backends.filters import (
    GaussianFilter,
    NoFilter,
    TopHatFilter,
)
from acm.estimators.galaxy_clustering.backends.pypower import (
    PypowerBackend,
)

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
        with pytest.raises(ValueError, match="data_positions"):
            PypowerBackend(np.ones((10, 2)))

    def test_invalid_randoms_positions_shape(self, data_pos):
        with pytest.raises(ValueError, match="randoms_positions"):
            PypowerBackend(data_pos, randoms_positions=np.ones((10, 2)))

    def test_invalid_data_weights_length(self, data_pos):
        with pytest.raises(ValueError, match="data_weights"):
            PypowerBackend(data_pos, data_weights=np.ones(N + 1))

    def test_randoms_weights_without_randoms_raises(self, data_pos, rand_w):
        with pytest.raises(ValueError, match="randoms_weights requires"):
            PypowerBackend(data_pos, randoms_weights=rand_w)

    def test_invalid_randoms_weights_length(self, data_pos, rand_pos):
        with pytest.raises(ValueError, match="randoms_weights"):
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

class TestFilters:
    K = (np.array([0.0, 0.1, 0.5]),) * 3
    V = np.array([1.0, 2.0, 3.0])

    def test_gaussian_zero_k_unchanged(self):
        """At k=0 the Gaussian kernel equals 1, so v is returned unchanged."""
        f = GaussianFilter(r=5.0)
        np.testing.assert_almost_equal(f((np.array([0.0]),) * 3, np.array([1.0])), [1.0])

    def test_gaussian_attenuates_high_k(self):
        """Gaussian filter should attenuate high-k modes more than low-k."""
        f = GaussianFilter(r=5.0)
        assert f((np.array([0.01]),) * 3, np.array([1.0])) > f((np.array([1.0]),) * 3, np.array([1.0]))

    def test_tophat_zero_k_unchanged(self):
        """At k=0 the top-hat kernel equals 1, so v is returned unchanged."""
        f = TopHatFilter(r=5.0)
        np.testing.assert_almost_equal(f((np.array([0.0]),) * 3, np.array([1.0])), [1.0])

    def test_nofilter_returns_v(self):
        np.testing.assert_array_equal(NoFilter(r=0.0)(self.K, self.V), self.V)

    def test_filter_radius_stored(self):
        for cls in [GaussianFilter, TopHatFilter, NoFilter]:
            assert cls(r=7.0).r == 7.0

class TestGetQueryPositions:
    def test_randoms_shape(self, backend):
        coords = backend.get_query_positions(method="randoms", nquery=50)
        assert coords.shape == (50, 3)

    def test_randoms_default_nquery(self, backend):
        coords = backend.get_query_positions(method="randoms")
        assert coords.shape == (5 * N, 3)

    def test_lattice_shape(self, backend):
        coords = backend.get_query_positions(method="lattice")
        assert coords.ndim == 2
        assert coords.shape[1] == 3

    def test_output_dtype_float32(self, backend):
        assert backend.get_query_positions(method="randoms", nquery=10).dtype == np.float32

    def test_seed_reproducibility(self, backend):
        c1 = backend.get_query_positions(method="randoms", nquery=20, seed=7)
        c2 = backend.get_query_positions(method="randoms", nquery=20, seed=7)
        np.testing.assert_array_equal(c1, c2)

    def test_invalid_method_raises(self, backend):
        with pytest.raises(ValueError, match="method"):
            backend.get_query_positions(method="invalid")
