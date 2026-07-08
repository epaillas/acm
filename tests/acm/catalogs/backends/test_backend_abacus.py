import pandas as pd
import pytest
import yaml
from conftest import MockAbacusHOD, make_hod_tracer_dict

from acm.catalogs.backends.abacus import AbacusHODBackend
from acm.catalogs.dataclasses import Tracer

# ruff: noqa: ANN001, ANN201, ARG002, D102, D103, INP001, S101


#%% Fixtures

@pytest.fixture
def config_dict():
    return {
        "sim_params": {
            "sim_name": "base_c000_ph000",
            "z_mock": 0.5,
        },
        "HOD_params": {
            "tracer_flags": {"FOO": True, "BAR": False, "LRG": False},
            "FOO_params": {"alpha": 1.0, "sigma": 0.3, "ic": 1},
            "BAR_params": {"alpha": 0.8, "sigma": 0.2, "ic": 1},
            "LRG_params": {"alpha": 0.9, "sigma": 0.25, "ic": 1},
        }
    }

@pytest.fixture
def config_file(tmp_path, config_dict):
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:  # noqa: PTH123
        yaml.dump(config_dict, f)
    return path

@pytest.fixture
def backend(config_file):
    return AbacusHODBackend(
        cosmo_idx=0,
        phase_idx=0,
        sim_type="base",
        config_file=config_file,
    )

@pytest.fixture
def tracer_foo():
    return Tracer(name="FOO", params={"alpha": 1.2})

@pytest.fixture
def tracer_bar():
    return Tracer(name="BAR", params={"alpha": 0.9})

@pytest.fixture
def dm_catalog(backend, tracer_foo):
    """Load dark matter catalog for a single tracer."""
    return backend.load_dark_matter_catalog(redshift=0.5, tracers=[tracer_foo])


class TestInit:
    """Tests for AbacusHODBackend.__init__."""

    def test_sim_type_stored(self, config_file):
        b = AbacusHODBackend(sim_type="base", config_file=config_file)
        assert b.sim_type == "base"

    def test_invalid_sim_type_raises(self, config_file):
        with pytest.raises(ValueError, match="Unknown simulation type"):
            AbacusHODBackend(sim_type="invalid", config_file=config_file)

    def test_missing_config_raises(self):
        with pytest.raises(ValueError, match="not found"):
            AbacusHODBackend(config_file="/nonexistent/config.yaml")

    def test_sim_params_loaded(self, backend):
        assert "sim_name" in backend.sim_params

    def test_hod_params_loaded(self, backend):
        assert "FOO_params" in backend.hod_params

    def test_kwargs_override_sim_params(self, config_file):
        b = AbacusHODBackend(config_file=config_file, custom_param=42)
        assert b.sim_params["custom_param"] == 42


class TestLoadDarkMatterCatalog:
    """Tests for AbacusHODBackend.load_dark_matter_catalog."""

    def test_returns_mock_abacus_hod(self, backend, tracer_foo):
        dm = backend.load_dark_matter_catalog(redshift=0.5, tracers=[tracer_foo])
        assert isinstance(dm, MockAbacusHOD)

    def test_redshift_set_in_sim_params(self, backend, tracer_foo):
        """The redshift passed should be set in the sim_params of the returned AbacusHOD instance."""
        dm = backend.load_dark_matter_catalog(redshift=0.8, tracers=[tracer_foo])
        assert dm.sim_params["z_mock"] == 0.8

    def test_tracer_flag_enabled(self, backend, tracer_foo):
        """Requesting a tracer should set its flag to True in the returned AbacusHOD instance."""
        dm = backend.load_dark_matter_catalog(redshift=0.5, tracers=[tracer_foo])
        assert dm.hod_params["tracer_flags"]["FOO"] is True

    def test_tracer_params_overridden(self, backend):
        """Params passed in the tracer should override those in the config."""
        tracer = Tracer(name="FOO", params={"alpha": 99.0})
        dm = backend.load_dark_matter_catalog(redshift=0.5, tracers=[tracer])
        assert dm.hod_params["FOO_params"]["alpha"] == 99.0

    def test_missing_tracer_params_raises(self, config_file, tmp_path):
        """A tracer with no params in config and no params in tracer should raise."""
        config = {
            "sim_params": {},
            "HOD_params": {
                "tracer_flags": {"FOO": False},
                "FOO_params": {},  # empty
            }
        }
        path = tmp_path / "empty_config.yaml"
        path.write_text(yaml.dump(config))
        b = AbacusHODBackend(config_file=path)
        with pytest.raises(ValueError, match="HOD parameters"):
            b.load_dark_matter_catalog(redshift=0.5, tracers=[Tracer(name="FOO")])

    def test_cache_populated_after_first_load(self, backend, tracer_foo):
        """Cache should contain the redshift key after first load."""
        backend.load_dark_matter_catalog(redshift=0.5, tracers=[tracer_foo])
        assert 0.5 in backend._cache

    def test_cache_returns_same_instance(self, backend, tracer_foo):
        """Second call with same redshift should return the exact same object."""
        dm1 = backend.load_dark_matter_catalog(redshift=0.5, tracers=[tracer_foo])
        dm2 = backend.load_dark_matter_catalog(redshift=0.5, tracers=[tracer_foo])
        assert dm1 is dm2

    def test_no_cache_bypasses_cache(self, backend, tracer_foo):
        """no_cache=True should return a new instance even if redshift is cached."""
        dm1 = backend.load_dark_matter_catalog(redshift=0.5, tracers=[tracer_foo])
        dm2 = backend.load_dark_matter_catalog(redshift=0.5, tracers=[tracer_foo], no_cache=True)
        assert dm1 is not dm2

    def test_different_redshifts_cached_separately(self, backend, tracer_foo):
        """Each redshift should get its own cache entry."""
        backend.load_dark_matter_catalog(redshift=0.5, tracers=[tracer_foo])
        backend.load_dark_matter_catalog(redshift=1.0, tracers=[tracer_foo])
        assert 0.5 in backend._cache
        assert 1.0 in backend._cache

class TestMakeGalaxyCatalog:
    """Tests for AbacusHODBackend.make_galaxy_catalog."""

    def test_returns_dict_keyed_by_tracer(self, backend, dm_catalog, tracer_foo):
        """Result should be a dict with keys matching the tracer names."""
        result = backend.make_galaxy_catalog(dm_catalog, tracers=[tracer_foo])
        assert tracer_foo in result

    def test_result_is_dataframe(self, backend, dm_catalog, tracer_foo):
        """Each tracer's catalog should be a DataFrame."""
        result = backend.make_galaxy_catalog(dm_catalog, tracers=[tracer_foo])
        assert isinstance(result[tracer_foo], pd.DataFrame)

    def test_columns_are_lowercase(self, backend, dm_catalog, tracer_foo):
        """All columns in the resulting DataFrame should be lowercase."""
        result = backend.make_galaxy_catalog(dm_catalog, tracers=[tracer_foo])
        for col in result[tracer_foo].columns:
            assert col == col.lower()

    def test_is_cent_column_added(self, backend, dm_catalog, tracer_foo):
        """make_galaxy_catalog should add an is_cent column based on Ncent."""
        result = backend.make_galaxy_catalog(dm_catalog, tracers=[tracer_foo])
        assert "is_cent" in result[tracer_foo].columns

    def test_is_cent_is_boolean(self, backend, dm_catalog, tracer_foo):
        """The is_cent column should be of boolean type."""
        result = backend.make_galaxy_catalog(dm_catalog, tracers=[tracer_foo])
        assert result[tracer_foo]["is_cent"].dtype == bool

    def test_bgs_alone_accepted(self, backend):
        """BGS requested alone should not raise."""
        tracer = Tracer(name="LRG", params={"alpha": 1.0})
        dm_catalog = backend.load_dark_matter_catalog(redshift=0.5, tracers=[tracer])
        tracer = Tracer(name="BGS", params={})
        result = backend.make_galaxy_catalog(dm_catalog, tracers=[tracer])
        assert tracer in result

    def test_bgs_without_lrg_raises(self, config_file, dm_catalog):
        """Requesting BGS without LRG should raise an error."""
        backend = AbacusHODBackend(sim_type="base", config_file=config_file, HOD_params={"tracer_flags": {"LRG": False}, "LRG_params": {"alpha": 1.0}})
        tracer = Tracer(name="BGS", params={})
        with pytest.raises(KeyError, match="BGS"):
            backend.make_galaxy_catalog(dm_catalog, tracers=[tracer])

    def test_bgs_with_other_tracers_raises(self, backend, dm_catalog):
        """Requesting BGS together with other tracers should raise an error."""
        tracers = [Tracer(name="BGS"), Tracer(name="FOO")]
        with pytest.raises(ValueError, match="BGS"):
            backend.make_galaxy_catalog(dm_catalog, tracers=tracers)

    def test_invalid_hod_param_raises(self, backend, dm_catalog):
        """Passing an invalid HOD parameter in the tracer params should raise an error."""
        tracer = Tracer(name="FOO", params={"invalid_key": 1.0})
        with pytest.raises(ValueError, match="invalid keys"):
            backend.make_galaxy_catalog(dm_catalog, tracers=[tracer])

    def test_mapping_renames_params(self, backend, dm_catalog):
        """A mapping dict should rename tracer params before passing to AbacusHOD."""
        tracer = Tracer(name="FOO", params={"my_alpha": 1.5})
        result = backend.make_galaxy_catalog(
            dm_catalog, tracers=[tracer], mapping={"alpha": ["my_alpha"]}
        )
        assert tracer in result

    def test_use_logsigma_converts_sigma(self, backend, dm_catalog):
        """sigma=-0.5 with use_logsigma=True should be converted to 10**(-0.5) before run_hod."""
        tracer = Tracer(name="FOO", params={"sigma": -0.5})
        backend.make_galaxy_catalog(dm_catalog, tracers=[tracer], use_logsigma=True)
        received_sigma = dm_catalog.last_run_hod_tracers["FOO"]["sigma"]
        assert received_sigma == pytest.approx(10 ** -0.5)

    def test_logsigma_key_converted(self, backend, dm_catalog):
        """logsigma=-0.5 should be removed and replaced by sigma=10**(-0.5) before run_hod."""
        tracer = Tracer(name="FOO", params={"logsigma": -0.5})
        backend.make_galaxy_catalog(dm_catalog, tracers=[tracer])
        received = dm_catalog.last_run_hod_tracers["FOO"]
        assert "logsigma" not in received
        assert received["sigma"] == pytest.approx(10 ** -0.5)

    def test_seed_alias_accepted(self, backend, dm_catalog, tracer_foo):
        """'seed' should be accepted as alias for 'reseed'."""
        result = backend.make_galaxy_catalog(dm_catalog, tracers=[tracer_foo], seed=42)
        assert tracer_foo in result

    def test_nthreads_alias_accepted(self, backend, dm_catalog, tracer_foo):
        """'nthreads' should be accepted as alias for 'Nthread'."""
        result = backend.make_galaxy_catalog(dm_catalog, tracers=[tracer_foo], nthreads=2)
        assert tracer_foo in result


class TestResolveTracerName:
    """Tests for AbacusHODBackend._resolve_tracer_name."""

    def test_known_alias_resolved(self, backend):
        assert backend._resolve_tracer_name("BGS") == "LRG"

    def test_unknown_name_passthrough(self, backend):
        assert backend._resolve_tracer_name("FOO") == "FOO"

    def test_resolution_emits_warning(self, backend, caplog):
        with caplog.at_level("WARNING"):
            backend._resolve_tracer_name("BGS")
        assert "BGS" in caplog.text
        assert "LRG" in caplog.text


class TestAddCentrals:
    """Tests for AbacusHODBackend._add_centrals."""

    def test_ncent_removed(self):
        galaxy_dict = {"FOO": make_hod_tracer_dict(n=10, n_cent=3)}
        AbacusHODBackend._add_centrals(galaxy_dict, "FOO")
        assert "Ncent" not in galaxy_dict["FOO"]

    def test_is_cent_added(self):
        galaxy_dict = {"FOO": make_hod_tracer_dict(n=10, n_cent=3)}
        AbacusHODBackend._add_centrals(galaxy_dict, "FOO")
        assert "is_cent" in galaxy_dict["FOO"]

    def test_centrals_flagged_correctly(self):
        """The first Ncent galaxies should be centrals (is_cent=True), the rest should be satellites (is_cent=False)."""
        galaxy_dict = {"FOO": make_hod_tracer_dict(n=10, n_cent=3)}
        AbacusHODBackend._add_centrals(galaxy_dict, "FOO")
        is_cent = galaxy_dict["FOO"]["is_cent"]
        assert sum(is_cent) == 3
        assert all(is_cent[:3])
        assert not any(is_cent[3:])

    def test_missing_ncent_raises(self):
        """Passing a galaxy dict without Ncent should raise a KeyError."""
        galaxy_dict = {"FOO": {"x": [1.0], "y": [2.0]}}  # no Ncent
        with pytest.raises(KeyError, match="Ncent"):
            AbacusHODBackend._add_centrals(galaxy_dict, "FOO")


class TestUpdateDefaultTracers:
    """Tests for AbacusHODBackend.update_default_tracers."""

    def test_tracer_flag_set_true(self, backend):
        """Requesting a tracer should set its flag to True in the HOD parameters."""
        hod_params = {"tracer_flags": {"FOO": False}, "FOO_params": {"alpha": 1.0}}
        tracer = Tracer(name="FOO", params={})
        backend.update_default_tracers(hod_params, tracers=[tracer])
        assert hod_params["tracer_flags"]["FOO"] is True

    def test_tracer_params_overridden(self, backend):
        """Params passed in the tracer should override those in the HOD parameters."""
        hod_params = {"tracer_flags": {"FOO": False}, "FOO_params": {"alpha": 1.0}}
        tracer = Tracer(name="FOO", params={"alpha": 99.0})
        backend.update_default_tracers(hod_params, tracers=[tracer])
        assert hod_params["FOO_params"]["alpha"] == 99.0

    def test_empty_tracer_params_raises(self, backend):
        """Passing a tracer with no params and no params in HOD parameters should raise an error."""
        hod_params = {"tracer_flags": {"FOO": False}, "FOO_params": {}}
        tracer = Tracer(name="FOO", params={})
        with pytest.raises(ValueError, match="HOD parameters"):
            backend.update_default_tracers(hod_params, tracers=[tracer])

    def test_no_active_tracers_raises(self, backend):
        """If no tracers are passed and no tracer flags are True, should raise an error."""
        hod_params = {"tracer_flags": {"FOO": False}}
        with pytest.raises(ValueError, match="At least one tracer"):
            backend.update_default_tracers(hod_params, tracers=[])

    def test_no_tracers_passed_uses_empty_list(self, backend):
        """Calling with no tracers should not raise if flags are already set."""
        hod_params = {"tracer_flags": {"FOO": True}, "FOO_params": {"alpha": 1.0}}
        backend.update_default_tracers(hod_params)  # no tracers kwarg

    def test_new_tracer_added_to_flags(self, backend):
        """A tracer not previously in tracer_flags should be added."""
        hod_params = {"tracer_flags": {}, "NEW_params": {"alpha": 1.0}}
        tracer = Tracer(name="NEW", params={})
        backend.update_default_tracers(hod_params, tracers=[tracer])
        assert hod_params["tracer_flags"]["NEW"] is True


class TestBoxsize:
    """Tests for the boxsize property."""

    def test_boxsize_base(self, backend):
        assert backend.boxsize == 2000.0  # from BOXSIZES["base"]

    def test_boxsize_type(self, backend):
        assert isinstance(backend.boxsize, (int, float))
