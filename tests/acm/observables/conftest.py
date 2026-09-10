"""Shared fixtures for acm.observables tests."""
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# ruff: noqa: ARG002, ANN001, ANN201, D102, D103

class DummySunbirdModel:
    """Minimal stand-in for a sunbird.emulators.BaseModel, avoiding a real sunbird dependency."""

    def __init__(self, output: np.ndarray) -> None:
        self._output = np.asarray(output, dtype=float)

    def get_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Return the same fixed output row, repeated once per input sample."""
        n = x.shape[0]
        return torch.as_tensor(np.tile(self._output, (n, 1)))

@pytest.fixture
def dummy_sunbird_model() -> DummySunbirdModel:
    """Return a fixed (n_features,) output per input row."""
    return DummySunbirdModel(output=np.array([1.0, 2.0, 3.0, 4.0]))

@pytest.fixture
def make_dummy_model(dummy_sunbird_model):
    """Build an ObservableModel wrapping the dummy sunbird model."""
    from acm.observables.model import ObservableModel  # noqa: PLC0415

    def _make(transform=None):  # noqa: ANN202
        return ObservableModel(model=dummy_sunbird_model, transform=transform)

    return _make


class DummyModel:
    """A model stub matching ObservableModel's public interface, decoupled from sunbird."""

    def __init__(self, n_features: int) -> None:
        self._n_features = n_features
        self.transform = None

    def get_prediction(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.tile(np.arange(self._n_features, dtype=float), (x.shape[0], 1))

    def get_error(self, x, truth, method, **kwargs):
        pred = self.get_prediction(x)
        return np.median(np.abs(truth - pred), axis=0)

    def make_covariance(self, y, **kwargs):
        return np.cov(y, rowvar=False)


_sunbird_emulators = MagicMock()
_sunbird_emulators.BaseModel = object # For typing

def pytest_configure(config):  # noqa: ARG001
    sys.modules.setdefault("sunbird", MagicMock())
    sys.modules.setdefault("sunbird.emulators", _sunbird_emulators)
