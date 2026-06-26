"""Mocking all backends here to avoid imports of loaded packages by mistake."""
import sys
from unittest.mock import MagicMock

import numpy as np

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, ARG002, D102, D103, D105, INP001


#%% Jax numpy methods used in JaxpowerBackend - mocked trough numpy
jax_mock = MagicMock()
jax_mock.numpy.where = np.where
jax_mock.numpy.exp = np.exp
jax_mock.numpy.sum = np.sum
jax_mock.numpy.meshgrid = np.meshgrid
jax_mock.numpy.vstack = np.vstack
jax_mock.numpy.exp = np.exp

#%% jaxpower mock
class RealMeshField:
    """Minimal real mesh sentinel."""

    def __init__(self, value): self.value = value
    def paint(self, **kwargs): return self
    def sum(self): return np.sum(self.value)
    def mean(self): return np.mean(self.value)
    def read(self, positions, resampler="cic"): return np.zeros(len(positions))
    def clone(self, value): return RealMeshField(value)
    def r2c(self): return ComplexMeshField(self.value.astype(complex))
    def __mul__(self, other): return RealMeshField(self.value * (other.value if isinstance(other, RealMeshField) else other))
    def __rmul__(self, other): return RealMeshField(self.value * (other.value if isinstance(other, RealMeshField) else other))
    def __truediv__(self, other): return RealMeshField(self.value / (other.value if isinstance(other, RealMeshField) else other))
    def __sub__(self, other): return RealMeshField(self.value - (other.value if isinstance(other, RealMeshField) else other))

class ComplexMeshField:
    """Minimal complex mesh sentinel."""

    def __init__(self, value): self.value = value
    def c2r(self): return RealMeshField(self.value.real)
    def __mul__(self, other): return ComplexMeshField(self.value * (other.value if isinstance(other, ComplexMeshField) else other))
    def __rmul__(self, other): return ComplexMeshField(self.value * (other.value if isinstance(other, ComplexMeshField) else other))

class ParticleField:
    """Minimal particle field sentinel."""

    def __init__(self, positions, weights=None, **kwargs):
        self.positions = positions
        self.weights = weights if weights is not None else np.ones(len(positions))
        self._size = len(positions)
    def paint(self, out="real", **kwargs): return RealMeshField(np.random.default_rng(0).uniform(0.5, 1.5, (32, 32, 32)))
    def sum(self): return float(np.sum(self.weights))
    @property
    def size(self): return self._size


def _make_mesh_attrs():
    """Return a MagicMock mattrs with sensible defaults for arithmetic and unpacking."""
    mattrs = MagicMock()
    mattrs.boxsize = np.array([100.0, 100.0, 100.0])
    mattrs.boxcenter = np.array([50.0, 50.0, 50.0])
    mattrs.meshsize = np.array([32, 32, 32])
    mattrs.cellsize = np.array([100.0 / 32] * 3)
    x = np.linspace(0, 100, 32)
    mattrs.rcoords.return_value = (x, x, x)
    mattrs.kcoords.return_value = np.linspace(0, 10, 10)
    return mattrs

jaxpower_mock = MagicMock()
jaxpower_mock.RealMeshField = RealMeshField
jaxpower_mock.ComplexMeshField = ComplexMeshField
jaxpower_mock.ParticleField = ParticleField
jaxpower_mock.get_mesh_attrs = lambda *args, **kwargs: _make_mesh_attrs()  # noqa: ARG005
jaxpower_mock.MeshAttrs = MagicMock

#%% pypower mock
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

pypower_mock = MagicMock()
pypower_mock.CatalogMesh = CatalogMesh


def pytest_configure(config):  # noqa: ARG001
    sys.modules["jax"] = jax_mock
    sys.modules["jax.numpy"] = jax_mock.numpy
    sys.modules["jaxpower"] = jaxpower_mock
    sys.modules["pypower"] = pypower_mock
