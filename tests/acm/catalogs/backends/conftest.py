"""Mocking all backends here to avoid imports of loaded packages by mistake."""
import sys
from unittest.mock import MagicMock

import numpy as np

# ruff: noqa: ANN001, ANN201, ARG002, D103, INP001

def make_hod_tracer_dict(n: int = 100, n_cent: int = 10) -> dict:
    """Generate a minimal AbacusHOD-style tracer output dict."""
    rng = np.random.default_rng(42)
    return {
        "x":      rng.uniform(0, 500, n).tolist(),
        "y":      rng.uniform(0, 500, n).tolist(),
        "z":      rng.uniform(0, 500, n).tolist(),
        "vx":     rng.normal(0, 100, n).tolist(),
        "vy":     rng.normal(0, 100, n).tolist(),
        "vz":     rng.normal(0, 100, n).tolist(),
        "Ncent":  n_cent,
    }

class MockAbacusHOD:
    """
    Minimal mock of abacusnbody.hod.abacus_hod.AbacusHOD.

    Mimics the interface used by AbacusHODBackend.
    """

    def __init__(self, sim_params: dict, hod_params: dict) -> None:
        self.sim_params = sim_params
        self.hod_params = hod_params
        # Mirrors AbacusHOD.tracers: maps tracer name -> param dict
        self.tracers = {
            name: hod_params.get(f"{name}_params", {})
            for name, active in hod_params.get("tracer_flags", {}).items()
            if active
        }
        self.last_run_hod_tracers: dict | None = None  # captures last call for testing

    def run_hod(
        self,
        tracers: dict,
        want_rsd: bool = False,
        reseed=None,
        Nthread: int = 1,  # noqa: N803
        **kwargs,
    ) -> dict:
        """Return a minimal galaxy dict for each requested tracer."""
        self.last_run_hod_tracers = tracers  # store for inspection
        return {name: make_hod_tracer_dict() for name in tracers}

mock_abacus_module = MagicMock()
mock_abacus_module.AbacusHOD = MockAbacusHOD

def pytest_configure(config):  # noqa: ARG001
    sys.modules["abacusnbody"] = MagicMock()
    sys.modules["abacusnbody.hod"] = MagicMock()
    sys.modules["abacusnbody.hod.abacus_hod"] = mock_abacus_module
