"""Mock cosmology for testing purposes."""
import sys
from unittest.mock import MagicMock

import pytest

# ruff: noqa: ANN001, ANN201, ARG001, D103, INP001


class DummyCosmology:
    """Minimal mock of cosmoprimo.Cosmology for testing purposes."""

    def __init__(self, efunc=0.8, add=800) -> None:
        """Initialize the dummy cosmology."""
        self._efunc = efunc
        self._add = add

    def efunc(self, z):
        """Return a dummy value for the expansion function."""
        return self._efunc + z

    def angular_diameter_distance(self, z):
        """Return a dummy value for the angular diameter distance."""
        return self._add / (1.0 + z)

    def comoving_radial_distance(self, z):
        """Return a dummy value for the comoving radial distance."""
        return 1000.0 * z

    def __getstate__(self) -> dict:
        """Return a dummy state for pickling."""
        return {'efunc': self._efunc, 'add': self._add}

    @classmethod
    def from_state(cls, state) -> 'DummyCosmology':
        """Restore a dummy state from pickling."""
        return cls(efunc=state['efunc'], add=state['add'])

cosmoprimo_module = MagicMock()
cosmoprimo_module.fiducial.DESI.return_value = DummyCosmology()
cosmoprimo_module.fiducial.AbacusSummit.return_value = DummyCosmology()
cosmoprimo_module.Cosmology = DummyCosmology

@pytest.fixture
def cosmo_mock1():
    """Fixture to provide a mock cosmology."""
    return DummyCosmology()

@pytest.fixture
def cosmo_mock2():
    """Fixture to provide another mock cosmology."""
    return DummyCosmology(efunc=1.0, add=1000)

def pytest_configure(config):
    sys.modules['cosmoprimo'] = cosmoprimo_module
    sys.modules['cosmoprimo.fiducial'] = MagicMock()
