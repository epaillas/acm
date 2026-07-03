from unittest.mock import MagicMock

import numpy as np
import pytest

from acm.estimators.galaxy_clustering.backends.pyrecon import PyreconBackend

# ruff: noqa: ANN001, ANN201, D101, D102, D103, INP001, S101


#%% Fixtures

N, M = 20, 30
BOXSIZE = 100.0
MESHSIZE = 32


@pytest.fixture
def data_pos():
    return np.random.default_rng(0).uniform(0, BOXSIZE, (N, 3))


@pytest.fixture
def rand_pos():
    return np.random.default_rng(1).uniform(0, BOXSIZE, (M, 3))


@pytest.fixture
def data_w():
    return np.ones(N)


@pytest.fixture
def rand_w():
    return np.ones(M)


@pytest.fixture
def backend(data_pos):
    return PyreconBackend(data_pos, boxsize=BOXSIZE, meshsize=MESHSIZE)


@pytest.fixture
def backend_with_randoms(data_pos, rand_pos, data_w, rand_w):
    return PyreconBackend(data_pos, BOXSIZE, MESHSIZE, rand_pos, data_w, rand_w)

#%% test classes
class TestPyreconBackendInit:
    def test_creates_instance(self, backend):
        assert isinstance(backend, PyreconBackend)

    def test_invalid_data_positions_shape(self):
        with pytest.raises(ValueError, match="data_positions"):
            PyreconBackend(np.ones((10, 2)), boxsize=BOXSIZE, meshsize=MESHSIZE)

    def test_invalid_data_weights_length(self, data_pos):
        with pytest.raises(ValueError, match="data_weights"):
            PyreconBackend(data_pos, BOXSIZE, MESHSIZE, data_weights=np.ones(N + 1))

    def test_randoms_weights_without_randoms_raises(self, data_pos, rand_w):
        with pytest.raises(ValueError, match="randoms_weights requires"):
            PyreconBackend(data_pos, BOXSIZE, MESHSIZE, randoms_weights=rand_w)

    def test_invalid_randoms_weights_length(self, data_pos, rand_pos):
        with pytest.raises(ValueError, match="randoms_weights"):
            PyreconBackend(data_pos, BOXSIZE, MESHSIZE, rand_pos, randoms_weights=np.ones(M + 1))

    def test_size_data(self, backend, data_pos):
        assert backend.size_data == len(data_pos)

    def test_size_randoms(self, backend_with_randoms, rand_pos):
        assert backend_with_randoms.size_randoms == len(rand_pos)

    def test_size_randoms_raises_when_not_set(self, backend):
        with pytest.raises(ValueError, match="Randoms have not been set"):
            _ = backend.size_randoms

class TestPyreconBackendProperties:
    def test_boxsize(self, backend):
        np.testing.assert_array_equal(backend.boxsize, [BOXSIZE] * 3)

    def test_boxcenter(self, backend):
        np.testing.assert_array_equal(backend.boxcenter, [0.0, 0.0, 0.0])

    def test_meshsize(self, backend):
        np.testing.assert_array_equal(backend.meshsize, [MESHSIZE] * 3)

    def test_cellsize(self, backend):
        np.testing.assert_array_almost_equal(backend.cellsize, [BOXSIZE / MESHSIZE] * 3)

class TestPaint:
    def test_paint_data_calls_assign_cic(self, backend):
        backend.data_mesh.assign_cic = MagicMock()
        extra_pos = np.random.default_rng(2).uniform(0, BOXSIZE, (5, 3))
        backend.paint(extra_pos, mesh_type="data")
        backend.data_mesh.assign_cic.assert_called_once_with(extra_pos, None, wrap=True)

    def test_paint_data_updates_size(self, backend):
        extra_pos = np.random.default_rng(2).uniform(0, BOXSIZE, (5, 3))
        backend.paint(extra_pos, mesh_type="data")
        assert backend.size_data == N + 5

    def test_paint_randoms_calls_assign_cic(self, backend_with_randoms):
        backend_with_randoms.randoms_mesh.assign_cic = MagicMock()
        extra_pos = np.random.default_rng(3).uniform(0, BOXSIZE, (5, 3))
        backend_with_randoms.paint(extra_pos, mesh_type="randoms")
        backend_with_randoms.randoms_mesh.assign_cic.assert_called_once_with(extra_pos, None, wrap=True)

    def test_paint_randoms_updates_size(self, backend_with_randoms):
        extra_pos = np.random.default_rng(3).uniform(0, BOXSIZE, (5, 3))
        backend_with_randoms.paint(extra_pos, mesh_type="randoms")
        assert backend_with_randoms.size_randoms == M + 5

    def test_paint_randoms_without_randoms_mesh_raises(self, backend):
        """Calling paint with mesh_type='randoms' on a backend with no randoms should raise."""
        extra_pos = np.random.default_rng(3).uniform(0, BOXSIZE, (5, 3))
        with pytest.raises(ValueError, match="Randoms mesh is not initialized"):
            backend.paint(extra_pos, mesh_type="randoms")

    def test_paint_invalid_mesh_type_raises(self, backend):
        extra_pos = np.random.default_rng(2).uniform(0, BOXSIZE, (5, 3))
        with pytest.raises(ValueError, match="mesh_type"):
            backend.paint(extra_pos, mesh_type="invalid")

    def test_paint_with_weights_forwarded(self, backend):
        backend.data_mesh.assign_cic = MagicMock()
        extra_pos = np.random.default_rng(2).uniform(0, BOXSIZE, (5, 3))
        weights = np.ones(5)
        backend.paint(extra_pos, weights=weights, mesh_type="data")
        backend.data_mesh.assign_cic.assert_called_once_with(extra_pos, weights, wrap=True)

class TestDensityContrast:
    def test_read_before_set_raises(self, backend, data_pos):
        with pytest.raises(AttributeError, match="set_density_contrast"):
            backend.read_density_contrast(data_pos)

    def test_unsupported_resampler_raises(self, backend, data_pos):
        backend.set_density_contrast()
        with pytest.raises(NotImplementedError, match="CIC"):
            backend.read_density_contrast(data_pos, resampler="tsc")

    def test_set_stores_density_contrast(self, backend):
        backend.set_density_contrast()
        assert backend._density_contrast is not None

    def test_read_after_set_calls_read(self, backend, data_pos):
            backend.set_density_contrast()
            backend._density_contrast = MagicMock()
            backend.read_density_contrast(data_pos)
            backend._density_contrast.read_cic.assert_called_once_with(data_pos)

    def test_set_with_randoms(self, backend_with_randoms):
        backend_with_randoms.set_density_contrast()
        assert backend_with_randoms._density_contrast is not None

    def test_set_with_smoothing(self, backend):
            backend.set_density_contrast(smoothing_radius=5.0)
            backend.data_mesh.smooth_gaussian.assert_called_once_with(5.0, method="fftw")

class TestGetQueryPositions:
    def test_randoms_shape(self, backend):
        assert backend.get_query_positions(method="randoms", nquery=50).shape == (50, 3)

    def test_randoms_default_nquery(self, backend):
        assert backend.get_query_positions(method="randoms").shape == (5 * N, 3)

    def test_lattice_shape(self, backend):
        coords = backend.get_query_positions(method="lattice")
        assert coords.shape == (np.prod(backend.meshsize), 3)

    def test_output_dtype_float32(self, backend):
        assert backend.get_query_positions(method="randoms", nquery=10).dtype == np.float32

    def test_seed_reproducibility(self, backend):
        c1 = backend.get_query_positions(method="randoms", nquery=20, seed=7)
        c2 = backend.get_query_positions(method="randoms", nquery=20, seed=7)
        np.testing.assert_array_equal(c1, c2)

    def test_invalid_method_raises(self, backend):
        with pytest.raises(ValueError, match="method"):
            backend.get_query_positions(method="invalid")
