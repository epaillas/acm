"""Mocking all backends here to avoid imports of loaded packages by mistake."""

import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import yaml

from acm.utils.default import recursive_update

# ruff: noqa: ANN001, ANN201, ARG002, D103, N803


def make_hod_tracer_dict(n: int = 100, n_cent: int = 10) -> dict:
    """Generate a minimal AbacusHOD-style tracer output dict."""
    rng = np.random.default_rng(42)
    return {
        "x": rng.uniform(0, 500, n).tolist(),
        "y": rng.uniform(0, 500, n).tolist(),
        "z": rng.uniform(0, 500, n).tolist(),
        "vx": rng.normal(0, 100, n).tolist(),
        "vy": rng.normal(0, 100, n).tolist(),
        "vz": rng.normal(0, 100, n).tolist(),
        "Ncent": n_cent,
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
        Nthread: int = 1,
        **kwargs,
    ) -> dict:
        """Return a minimal galaxy dict for each requested tracer."""
        self.last_run_hod_tracers = tracers  # store for inspection
        return {name: make_hod_tracer_dict() for name in tracers}


class FakeCatalog:
    """Minimal stand-in for a HODDIES/mpytools Catalog."""

    def __init__(self, data) -> None:
        self._data = {key: np.asarray(value) for key, value in data.items()}

    def columns(self):
        """Return the column names."""
        return list(self._data)

    def __getitem__(self, key) -> object:
        """Return one column."""
        return self._data[key]


class MockMPTCatalog:
    """Mock of mpytools.Catalog that only supports reading registered files."""

    files: ClassVar[dict] = {}

    @classmethod
    def read(cls, path) -> FakeCatalog:
        """Return the catalog registered for path."""
        return FakeCatalog(cls.files[str(path)])


class MockHOD:
    """Minimal stand-in for HODDIES.HOD."""

    def __init__(
        self,
        param_file=None,
        args=None,
        hcat_file=None,
        path_to_abacus_sim=None,
        read_pinnochio=None,
        read_Uuchuu=None,
        subsample=None,
        **kwargs,
    ) -> None:
        config = {}
        if param_file is not None:
            with Path(param_file).open() as file:
                config = yaml.safe_load(file) or {}
        self.args = recursive_update(dict(config), kwargs)
        tracers = self.args.get("tracers", ["LRG"])
        self.args["tracers"] = [tracers] if isinstance(tracers, str) else list(tracers)
        self.hcat_file = hcat_file
        self.part_subsamples = subsample
        self.boxsize = self.args.get("hcat", {}).get("boxsize")
        self.make_mock_cat_calls = []

    def make_mock_cat(self, tracers=None, fix_seed=None, verbose=True):
        """Return a fake multi-tracer catalog and record the call."""
        tracers = tracers or self.args["tracers"]
        self.make_mock_cat_calls.append(
            {
                "tracers": list(tracers),
                "fix_seed": fix_seed,
                "verbose": verbose,
                "args": deepcopy(self.args),
            }
        )
        n_gal = 4
        chunks = []
        for index, tracer_name in enumerate(tracers):
            chunks.append(
                {
                    "x": np.arange(n_gal, dtype=float) + 10.0 * index,
                    "y": np.arange(n_gal, dtype=float) + 1.0,
                    "z": np.arange(n_gal, dtype=float) + 2.0,
                    "vx": np.zeros(n_gal),
                    "vy": np.zeros(n_gal),
                    "vz": np.zeros(n_gal),
                    "Mh": np.full(n_gal, 1e13 + index),
                    "halo_id": np.arange(n_gal) + 100 * index,
                    "Central": np.array([1, 1, 0, 0]),
                    "TRACER": np.full(n_gal, tracer_name),
                }
            )
        data = {
            key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]
        }
        return FakeCatalog(data)


mock_abacus_module = MagicMock()
mock_abacus_module.AbacusHOD = MockAbacusHOD

mock_hoddies_module = types.ModuleType("HODDIES")
mock_hoddies_module.__file__ = "/nonexistent/HODDIES/__init__.py"
mock_hoddies_module.HOD = MockHOD

mock_mpytools_module = types.ModuleType("mpytools")
mock_mpytools_module.Catalog = MockMPTCatalog


def pytest_configure(config):  # noqa: ARG001
    sys.modules["abacusnbody"] = MagicMock()
    sys.modules["abacusnbody.hod"] = MagicMock()
    sys.modules["abacusnbody.hod.abacus_hod"] = mock_abacus_module
    sys.modules["HODDIES"] = mock_hoddies_module
    sys.modules["mpytools"] = mock_mpytools_module
