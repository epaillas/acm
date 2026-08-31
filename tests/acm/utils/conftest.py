import sys
from unittest.mock import MagicMock

# ruff: noqa: ANN001, ANN201, D103

def pytest_configure(config):  # noqa: ARG001
    sys.modules.setdefault("jax", MagicMock())

