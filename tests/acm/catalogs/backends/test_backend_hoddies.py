"""Tests for the HODDIES catalog backend.

HODDIES and mpytools are mocked in ``conftest.py`` so that the unit tests do
not require either package to be installed.
"""

import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from acm.catalogs.backends import _LAZY_BACKENDS, load_backend
from acm.catalogs.backends.base import DarkMatterBackend
from acm.catalogs.backends.hoddies import HODDIESBackend, _Catalog
from acm.catalogs.dataclasses import Tracer

from .conftest import FakeCatalog, MockHOD, MockMPTCatalog, mock_hoddies_module

# ruff: noqa: ANN001, ANN201, D102, S101

# %% Fixtures


@pytest.fixture
def config_dict():
    """Return a minimal HODDIES configuration."""
    return {
        "tracers": ["LRG", "ELG"],
        "LRG": {
            "Ac": 1.0,
            "log_Mcent": 12.75,
            "sigma_M": 0.5,
            "M_0": 12.5,
            "M_1": 13.5,
            "alpha": 1.0,
        },
        "ELG": {
            "Ac": 0.05,
            "log_Mcent": 11.63,
            "sigma_M": 0.12,
            "M_0": 11.63,
            "M_1": 11.7,
            "alpha": 0.6,
        },
        "hcat": {"boxsize": 500, "z_simu": 0.95},
        "use_particles": False,
    }


@pytest.fixture
def config_file(tmp_path, config_dict):
    """Write the minimal configuration to a temporary YAML file."""
    path = tmp_path / "hoddies.yaml"
    with path.open("w") as file:
        yaml.dump(config_dict, file)
    return path


@pytest.fixture
def backend(config_file):
    """Return a configured backend."""
    return HODDIESBackend(config_file=config_file)


@pytest.fixture
def halo_data():
    """Return a small HODDIES-compatible halo catalog."""
    rng = np.random.default_rng(1)
    n_halo = 8
    return {
        "x": rng.uniform(0, 500, n_halo),
        "y": rng.uniform(0, 500, n_halo),
        "z": rng.uniform(0, 500, n_halo),
        "vx": rng.normal(0, 100, n_halo),
        "vy": rng.normal(0, 100, n_halo),
        "vz": rng.normal(0, 100, n_halo),
        "Mh": rng.uniform(1e12, 1e14, n_halo),
        "Rh": rng.uniform(100, 500, n_halo),
        "c": rng.uniform(3, 10, n_halo),
        "halo_id": np.arange(n_halo),
    }


@pytest.fixture
def particle_data():
    """Return a small HODDIES-compatible particle catalog."""
    rng = np.random.default_rng(2)
    n_part = 16
    return {
        "x": rng.uniform(0, 500, n_part),
        "y": rng.uniform(0, 500, n_part),
        "z": rng.uniform(0, 500, n_part),
        "vx": rng.normal(0, 100, n_part),
        "vy": rng.normal(0, 100, n_part),
        "vz": rng.normal(0, 100, n_part),
        "halo_id": np.repeat(np.arange(4), 4),
    }


@pytest.fixture
def dm_catalog(backend, halo_data):
    """Load and return the dark matter catalog at z=0.5."""
    backend.load_dark_matter_catalog(redshift=0.5, hcat=halo_data)
    return backend.get_dark_matter_catalog(redshift=0.5)


# %% Initialization


class TestInit:
    """Tests for HODDIESBackend.__init__."""

    def test_missing_config_raises(self):
        with pytest.raises(ValueError, match="not found"):
            HODDIESBackend(config_file="/nonexistent/hoddies.yaml")

    def test_missing_boxsize_raises(self, tmp_path, config_dict):
        config_dict["hcat"]["boxsize"] = None
        path = tmp_path / "no_boxsize.yaml"
        with path.open("w") as file:
            yaml.dump(config_dict, file)
        with pytest.raises(ValueError, match="box size"):
            HODDIESBackend(config_file=path)

    def test_boxsize_from_config(self, backend):
        assert backend.boxsize == 500

    def test_boxsize_kwargs_override(self, config_file):
        backend = HODDIESBackend(config_file=config_file, hcat={"boxsize": 1000})
        assert backend.boxsize == 1000

    def test_default_config(self, tmp_path, config_dict, monkeypatch):
        package_dir = tmp_path / "HODDIES"
        package_dir.mkdir()
        with (package_dir / "default_HOD_parameters.yaml").open("w") as file:
            yaml.dump(config_dict, file)
        monkeypatch.setattr(
            mock_hoddies_module, "__file__", str(package_dir / "__init__.py")
        )
        backend = HODDIESBackend(hcat={"boxsize": 750})
        assert backend.boxsize == 750

    def test_registered(self, config_file):
        backend = load_backend(
            "HODDIES", config_file=config_file, hcat={"boxsize": 500}
        )
        assert isinstance(backend, HODDIESBackend)

    def test_load_backend_instance_passthrough(self, backend):
        assert load_backend(backend) is backend

    def test_load_backend_imports_lazily(self, tmp_path, monkeypatch):
        module_name = "lazy_test_backend"
        (tmp_path / f"{module_name}.py").write_text(
            "from acm.catalogs.backends.base import DarkMatterBackend, register_backend\n"
            "@register_backend('LazyTest')\n"
            "class LazyTestBackend(DarkMatterBackend):\n"
            "    def load_dark_matter_catalog(self, *args, **kwargs):\n"
            "        return None\n"
            "    def get_dark_matter_catalog(self, *args, **kwargs):\n"
            "        return None\n"
            "    def make_galaxy_catalog(self, dm_catalog, tracers, **kwargs):\n"
            "        return {}\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setitem(_LAZY_BACKENDS, "LazyTest", module_name)
        monkeypatch.delitem(sys.modules, module_name, raising=False)

        backend = load_backend("LazyTest")

        assert module_name in sys.modules
        assert isinstance(backend, DarkMatterBackend)


# %% Loading


class TestLoadDarkMatterCatalog:
    """Tests for HODDIESBackend.load_dark_matter_catalog."""

    def test_returns_none(self, backend, halo_data):
        result = backend.load_dark_matter_catalog(redshift=0.5, hcat=halo_data)
        assert result is None

    def test_caches_hod(self, backend, halo_data):
        backend.load_dark_matter_catalog(redshift=0.5, hcat=halo_data)
        assert isinstance(backend.get_dark_matter_catalog(0.5), MockHOD)

    def test_get_before_load_raises(self, backend):
        with pytest.raises(KeyError, match="has not been loaded"):
            backend.get_dark_matter_catalog(0.5)

    def test_redshift_overrides_z_simu(self, backend, halo_data):
        backend.load_dark_matter_catalog(redshift=0.8, hcat=halo_data)
        hod = backend.get_dark_matter_catalog(0.8)
        assert hod.args["hcat"]["z_simu"] == 0.8

    def test_batch_loading(self, backend, halo_data):
        backend.load_dark_matter_catalog(hcat={0.5: halo_data, 1.0: halo_data})
        assert set(backend._cache) == {0.5, 1.0}

    def test_batch_requires_mapping(self, backend, halo_data):
        # A column dictionary passed without a redshift is rejected because its
        # keys cannot be interpreted as redshifts.
        with pytest.raises(TypeError, match="Batch keys must be redshifts"):
            backend.load_dark_matter_catalog(hcat=halo_data)

    def test_batch_particle_keys_must_match(self, backend, halo_data, particle_data):
        with pytest.raises(ValueError, match="must match"):
            backend.load_dark_matter_catalog(
                hcat={0.5: halo_data, 1.0: halo_data},
                particles={0.5: particle_data},
            )

    def test_missing_halo_columns_raise(self, backend, halo_data):
        del halo_data["halo_id"]
        with pytest.raises(ValueError, match="halo_id"):
            backend.load_dark_matter_catalog(redshift=0.5, hcat=halo_data)

    def test_missing_particle_columns_raise(
        self, config_file, halo_data, particle_data
    ):
        del particle_data["halo_id"]
        backend = HODDIESBackend(config_file=config_file, use_particles=True)
        with pytest.raises(ValueError, match="particle catalog"):
            backend.load_dark_matter_catalog(
                redshift=0.5, hcat=halo_data, particles=particle_data
            )

    def test_particles_require_use_particles(self, backend, halo_data, particle_data):
        with pytest.raises(ValueError, match="use_particles"):
            backend.load_dark_matter_catalog(
                redshift=0.5, hcat=halo_data, particles=particle_data
            )

    def test_particles_pass_through(self, config_file, halo_data, particle_data):
        backend = HODDIESBackend(config_file=config_file, use_particles=True)
        backend.load_dark_matter_catalog(
            redshift=0.5, hcat=halo_data, particles=particle_data
        )
        hod = backend.get_dark_matter_catalog(0.5)
        assert hod.part_subsamples is not None
        assert hod.args["use_particles"] is True

    def test_dataframe_input(self, backend, halo_data):
        backend.load_dark_matter_catalog(redshift=0.5, hcat=pd.DataFrame(halo_data))
        hod = backend.get_dark_matter_catalog(0.5)
        assert isinstance(hod.hcat_file, dict)

    def test_structured_array_input(self, backend, halo_data):
        array = np.array(
            list(zip(*halo_data.values(), strict=True)),
            dtype=[(name, "f8") for name in halo_data],
        )
        backend.load_dark_matter_catalog(redshift=0.5, hcat=array)
        hod = backend.get_dark_matter_catalog(0.5)
        assert set(hod.hcat_file) == set(halo_data)

    def test_path_input(self, backend, halo_data, tmp_path):
        path = tmp_path / "halos.hdf5"
        path.write_bytes(b"")
        MockMPTCatalog.files[str(path)] = halo_data
        try:
            backend.load_dark_matter_catalog(redshift=0.5, hcat=path)
        finally:
            MockMPTCatalog.files.pop(str(path), None)
        hod = backend.get_dark_matter_catalog(0.5)
        assert isinstance(hod.hcat_file, FakeCatalog)


# %% Galaxy catalogs


class TestMakeGalaxyCatalog:
    """Tests for HODDIESBackend.make_galaxy_catalog."""

    def test_output_columns(self, backend, dm_catalog):
        tracer = Tracer(name="LRG")
        catalog = backend.make_galaxy_catalog(dm_catalog, [tracer])
        data = catalog[tracer]
        assert list(data.columns) == [
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "mass",
            "id",
            "is_cent",
        ]
        assert data["is_cent"].dtype == bool
        assert set(data["id"]) == {0, 1, 2, 3}
        assert set(data["mass"]) == {1e13}

    def test_multiple_tracers(self, backend, dm_catalog):
        lrg = Tracer(name="LRG")
        elg = Tracer(name="ELG")
        catalog = backend.make_galaxy_catalog(dm_catalog, [lrg, elg])
        assert set(catalog) == {lrg, elg}
        assert len(catalog[lrg]) == 4
        assert len(catalog[elg]) == 4
        assert set(catalog[elg]["mass"]) == {1e13 + 1}

    def test_parameter_override_and_restore(self, backend, dm_catalog):
        original = dm_catalog.args["LRG"]["M_0"]
        tracer = Tracer(name="LRG", params={"M_0": 12.0})
        backend.make_galaxy_catalog(dm_catalog, [tracer])
        call = dm_catalog.make_mock_cat_calls[-1]
        assert call["args"]["LRG"]["M_0"] == 12.0
        assert dm_catalog.args["LRG"]["M_0"] == original

    def test_unknown_parameter_raises(self, backend, dm_catalog):
        tracer = Tracer(name="LRG", params={"not_a_parameter": 1.0})
        with pytest.raises(ValueError, match="invalid"):
            backend.make_galaxy_catalog(dm_catalog, [tracer])

    def test_unknown_tracer_raises(self, backend, dm_catalog):
        with pytest.raises(KeyError, match="not defined"):
            backend.make_galaxy_catalog(dm_catalog, [Tracer(name="QSO")])

    def test_tracer_alias(self, backend, dm_catalog):
        tracer = Tracer(name="BGS")
        catalog = backend.make_galaxy_catalog(dm_catalog, [tracer])
        assert list(catalog) == [tracer]
        assert dm_catalog.make_mock_cat_calls[-1]["tracers"] == ["LRG"]

    def test_alias_collision_raises(self, backend, dm_catalog):
        with pytest.raises(ValueError, match="already requested"):
            backend.make_galaxy_catalog(
                dm_catalog, [Tracer(name="LRG"), Tracer(name="BGS")]
            )

    def test_fix_seed(self, backend, dm_catalog):
        backend.make_galaxy_catalog(dm_catalog, [Tracer(name="LRG")], fix_seed=42)
        assert dm_catalog.make_mock_cat_calls[-1]["fix_seed"] == 42

    def test_unexpected_kwargs_raise(self, backend, dm_catalog):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            backend.make_galaxy_catalog(dm_catalog, [Tracer(name="LRG")], bad=1)


# %% Output conversion


def make_galaxy_catalog():
    """Return a catalog with one central and one satellite per halo."""
    return {
        "x": np.array([10.0, 20.0, 30.0, 40.0]),
        "y": np.array([11.0, 21.0, 31.0, 41.0]),
        "z": np.array([12.0, 22.0, 32.0, 42.0]),
        "vx": np.zeros(4),
        "vy": np.zeros(4),
        "vz": np.zeros(4),
        "Mh": np.ones(4),
        "halo_id": np.array([0, 0, 1, 1]),
        "Central": np.array([1, 0, 1, 0]),
        "TRACER": np.array(["LRG"] * 4),
    }


class TestOutputConversion:
    """Tests for extracting the output columns from a HODDIES catalog."""

    def test_catalog_wrapper_preserves_container(self):
        data = FakeCatalog(make_galaxy_catalog())
        catalog = _Catalog(data)
        assert catalog.data is data
        assert catalog.names == tuple(data.columns())
        np.testing.assert_array_equal(catalog["x"], data["x"])

    def test_unsupported_catalog_raises(self):
        with pytest.raises(TypeError, match="Unsupported catalog container"):
            _Catalog(object())

    def test_extract_catalog_columns(self):
        columns = _Catalog(FakeCatalog(make_galaxy_catalog())).extract(
            (
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "Mh",
                "halo_id",
                "Central",
                "TRACER",
            )
        )
        assert set(columns) == {
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "Mh",
            "halo_id",
            "Central",
            "TRACER",
        }

    def test_extract_missing_column_raises(self):
        data = make_galaxy_catalog()
        del data["TRACER"]
        with pytest.raises(KeyError, match="TRACER"):
            _Catalog(data).extract(("TRACER",))


class TestParticlePath:
    """Tests that particles are delegated to HODDIES' native path."""

    def test_particles_use_hoddies_native_path(
        self, config_file, halo_data, particle_data
    ):
        backend = HODDIESBackend(config_file=config_file, use_particles=True)
        backend.load_dark_matter_catalog(
            redshift=0.5, hcat=halo_data, particles=particle_data
        )
        dm_catalog = backend.get_dark_matter_catalog(0.5)

        backend.make_galaxy_catalog(dm_catalog, [Tracer(name="LRG")], fix_seed=42)

        call = dm_catalog.make_mock_cat_calls[-1]
        assert call["args"]["use_particles"] is True
        assert dm_catalog.part_subsamples is not None
