import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from acm.estimators.galaxy_clustering.backends import EstimatorBackend
from acm.estimators.galaxy_clustering.backends.base import register_backend

# ruff: noqa: ANN001, ANN201, ARG001, ARG002, D101, D102, D103, INP001

N_DATA = 50
N_RANDOMS = 250

@register_backend("dummyestimatorbackend")
class DummyBackend(EstimatorBackend):

    @property
    def boxsize(self):
        return (100.0, 100.0, 100.0)

    @property
    def boxcenter(self):
        return (50.0, 50.0, 50.0)

    @property
    def meshsize(self):
        return (64, 64, 64)

    @property
    def cellsize(self):
        return (100.0 / 64,) * 3

    def set_density_contrast(self, **kwargs):
        self._density_contrast = np.zeros(self.meshsize)

    def read_density_contrast(self, positions, resampler="cic"):
        rng = np.random.default_rng(42)
        return rng.uniform(0, 1, size=len(positions))

    def get_query_positions(self, method="randoms", nquery=None, seed=42):
        rng = np.random.default_rng(seed)
        if method == "randoms":
            n = nquery or 100
            cout =  rng.uniform(0, 100, size=(n, 3))
        elif method == "lattice":
            n = np.prod(self.meshsize)
            cout = rng.uniform(0, 100, size=(n, 3))
        else:
            raise ValueError("method must be one of ['lattice', 'randoms']")
        return cout

@pytest.fixture
def dummy_backend(data_positions, randoms_positions):
    return DummyBackend(data_positions, randoms_positions)

@pytest.fixture
def dummy_backend_no_randoms(data_positions):
    return DummyBackend(data_positions, randoms_positions=None)

@pytest.fixture
def data_positions():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 100, size=(N_DATA, 3))


@pytest.fixture
def randoms_positions():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 100, size=(N_RANDOMS, 3))


@pytest.fixture
def make_estimator(data_positions, randoms_positions):
    """Instantiate any BaseEstimator subclass with a DummyBackend."""
    def _make(cls, **kwargs) -> EstimatorBackend:
        return cls(
            backend="dummyestimatorbackend",
            data_positions=data_positions,
            randoms_positions=randoms_positions,
            **kwargs,
        )
    return _make

#%% Mock estimator dependencies modules - more detailled mocks can be added in the test files themselves if needed
def pytest_configure(config):
    sys.modules["pycorr"] = MagicMock()
    sys.modules["kymatio"] = MagicMock()
