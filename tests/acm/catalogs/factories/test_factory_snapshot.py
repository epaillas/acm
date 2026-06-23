import pytest
import logging

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from acm.catalogs.backends.base import DarkMatterBackend
from acm.catalogs.dataclasses import Tracer
from acm.catalogs.factories.snapshot import SnapshotCatalogFactory
from acm.catalogs.products.snapshot import SnapshotCatalog

logger = logging.getLogger(__name__)

#%% Fixtures

def make_tracer_data(n: int = 100) -> pd.DataFrame:
    """Generate minimal valid tracer data."""
    return pd.DataFrame({
        "x": np.random.uniform(0, 500, n),
        "y": np.random.uniform(0, 500, n),
        "z": np.random.uniform(0, 500, n),
        "vx": np.random.normal(0, 1, n),
        "vy": np.random.normal(0, 1, n),
        "vz": np.random.normal(0, 1, n),
    })
    
class DummyBackend(DarkMatterBackend):
    """A minimal implementation of DarkMatterBackend for testing."""
    def load_dark_matter_catalog(self, redshift: float, **kwargs):
        return MagicMock()  # Return a dummy catalog object

    def make_galaxy_catalog(self, dm_catalog, tracers: list[Tracer], **kwargs) -> None:
        return {tracer: make_tracer_data() for tracer in tracers}
    
    @property
    def boxsize(self) -> float:
        return 500.0

@pytest.fixture
def cosmo():
    m = MagicMock()
    m.efunc.return_value = 1.0
    m.angular_diameter_distance.return_value = 1000.0
    return m

@pytest.fixture
def cosmo_fid():
    m = MagicMock()
    m.efunc.return_value = 1.0
    m.angular_diameter_distance.return_value = 1000.0
    return m

@pytest.fixture
def tracer_foo():
    return Tracer(name="FOO", params={"a": 1})

@pytest.fixture
def tracer_bar():
    return Tracer(name="BAR", params={})

@pytest.fixture
def mock_backend():
    """Mock SnapshotBackend that returns minimal valid catalogs."""
    return DummyBackend()

@pytest.fixture
def magic_mock_factory():
    """A factory using MagicMock to mimics SnapshotBackend with valid return values."""
    backend = MagicMock(spec=DarkMatterBackend)
    backend.load_dark_matter_catalog.return_value = MagicMock()
    backend.make_galaxy_catalog.side_effect = lambda dm_catalog, tracers, **kwargs: {t: make_tracer_data() for t in tracers}
    backend.boxsize = 500.0
    
    factory = SnapshotCatalogFactory(
        backend=backend,
        catalog_class=SnapshotCatalog,
        cosmo=MagicMock(),
        cosmo_fid=MagicMock(),
    )
    return factory

@pytest.fixture
def factory(mock_backend, cosmo, cosmo_fid):
    return SnapshotCatalogFactory(
        backend=mock_backend,
        catalog_class=SnapshotCatalog,
        cosmo=cosmo,
        cosmo_fid=cosmo_fid,
    )

@pytest.fixture
def factory_with_catalogs(factory, mock_backend, tracer_foo):
    """Factory with catalogs already loaded at two redshifts."""
    factory.make_catalogs(redshifts=[0.5, 1.0], tracers=[tracer_foo])
    return factory


#%% Test classes 

class TestSnapshotCatalogFactoryConstruction:
    """Tests for the constructor and basic properties of SnapshotCatalogFactory."""
    
    def test_factory_stores_backend(self, factory, mock_backend):
        assert factory.backend is mock_backend

    def test_factory_stores_catalog_class(self, factory):
        assert factory.catalog_class is SnapshotCatalog

    def test_factory_stores_cosmo(self, factory, cosmo):
        assert factory.cosmo is cosmo

    def test_factory_stores_cosmo_fid(self, factory, cosmo_fid):
        assert factory.cosmo_fid is cosmo_fid

    def test_factory_starts_empty(self, factory):
        assert factory.redshifts == []
        assert factory.catalogs == {}

    def test_repr(self, factory_with_catalogs):
        r = repr(factory_with_catalogs)
        assert "SnapshotCatalogFactory" in r
        assert "SnapshotCatalog" in r

class TestMakeCatalogs:
    """Tests for the make_catalogs method of SnapshotCatalogFactory."""
    
    def test_make_catalogs_loads_all_redshifts(self, factory_with_catalogs):
        assert set(factory_with_catalogs.redshifts) == {0.5, 1.0}
        
    def test_catalogs_property_returns_copy(self, factory_with_catalogs):
        """Modifying the returned dict should not affect internal state."""
        catalogs = factory_with_catalogs.catalogs
        catalogs.clear()
        assert len(factory_with_catalogs.catalogs) == 2

    def test_make_catalogs_creates_snapshot_catalog(self, factory_with_catalogs):
        """Catalogs created should be instances of SnapshotCatalog."""
        for catalog in factory_with_catalogs.catalogs.values():
            assert isinstance(catalog, SnapshotCatalog)

    def test_make_catalogs_passes_boxsize(self, factory_with_catalogs):
        """Boxsize from the backend should be passed to the catalog."""
        for catalog in factory_with_catalogs.catalogs.values():
            np.testing.assert_array_equal(catalog._boxsize, [500., 500., 500.])

    def test_make_catalogs_passes_cosmo(self, factory_with_catalogs, cosmo):
        """Cosmology object from the factory should be passed to the catalog."""
        for catalog in factory_with_catalogs.catalogs.values():
            assert catalog.cosmo is cosmo

    def test_make_catalogs_sets_tracer_data(self, factory_with_catalogs):
        """Tracer data returned by the backend should be set in the catalog."""
        for catalog in factory_with_catalogs.catalogs.values():
            assert "FOO" in catalog.tracers

    def test_make_catalogs_per_redshift_tracers(self, factory, tracer_foo, tracer_bar):
        """Each redshift can have its own tracer list."""
        factory.make_catalogs(
            redshifts=[0.5, 1.0],
            tracers={0.5: [tracer_foo], 1.0: [tracer_bar]},
        )
        assert "FOO" in factory.get_catalog(0.5).tracers
        assert "BAR" in factory.get_catalog(1.0).tracers
        assert "FOO" not in factory.get_catalog(1.0).tracers
    
    def test_make_catalogs_forwards_dark_matter_kwargs(self, magic_mock_factory, tracer_foo):
        """Dark matter kwargs should be forwarded to the backend."""        
        dark_matter_kwargs = {"seed": 42, "cosmology_variant": "base"}
        magic_mock_factory.make_catalogs(
            redshifts=[0.5],
            tracers=[tracer_foo],
            dark_matter_kwargs=dark_matter_kwargs,
        )
        magic_mock_factory.backend.load_dark_matter_catalog.assert_called_once()
        call_kwargs = magic_mock_factory.backend.load_dark_matter_catalog.call_args[1]
        assert call_kwargs["redshift"] == 0.5
        assert call_kwargs["seed"] == 42
        assert call_kwargs["cosmology_variant"] == "base"

    
    def test_make_catalogs_per_redshift_passes_correct_tracers_to_backend(self, magic_mock_factory, tracer_foo, tracer_bar):
        """Each redshift should pass the correct tracer list to the backend."""
        magic_mock_factory.make_catalogs(
            redshifts=[0.5, 1.0],
            tracers={0.5: [tracer_foo], 1.0: [tracer_bar]},
        )
        first_call_tracers = magic_mock_factory.backend.make_galaxy_catalog.call_args_list[0][1]["tracers"]
        second_call_tracers = magic_mock_factory.backend.make_galaxy_catalog.call_args_list[1][1]["tracers"]
        assert magic_mock_factory.backend.make_galaxy_catalog.call_count == 2
        assert first_call_tracers == [tracer_foo] # First call (z=0.5) should have tracer_foo
        assert second_call_tracers == [tracer_bar] # Second call (z=1.0) should have tracer_bar

    def test_make_catalogs_forwards_kwargs_to_make_galaxy_catalog(self, magic_mock_factory, tracer_foo):
        """Additional kwargs should be forwarded to the backend's make_galaxy_catalog method."""
        extra_kwargs = {"hod_model": "zheng07", "scatter_type": "dexmag"}
        magic_mock_factory.make_catalogs(
            redshifts=[0.5],
            tracers=[tracer_foo],
            hod_model=extra_kwargs["hod_model"],
            scatter_type=extra_kwargs["scatter_type"],
        )
        magic_mock_factory.backend.make_galaxy_catalog.assert_called_once()
        call_kwargs = magic_mock_factory.backend.make_galaxy_catalog.call_args[1]
        assert call_kwargs["hod_model"] == "zheng07"
        assert call_kwargs["scatter_type"] == "dexmag"
        assert "dm_catalog" in call_kwargs
        assert "tracers" in call_kwargs



class TestGetCatalog:
    """Tests for the get_catalog method of SnapshotCatalogFactory."""
    
    def test_get_catalog_returns_correct_redshift(self, factory_with_catalogs):
        """Requesting a catalog by redshift should return a catalog with that redshift."""
        catalog = factory_with_catalogs.get_catalog(0.5)
        assert catalog.redshift == pytest.approx(0.5)

    def test_get_catalog_unknown_redshift_raises(self, factory_with_catalogs):
        """Requesting a redshift that wasn't made should raise a KeyError."""
        with pytest.raises(KeyError, match="0.8"):
            factory_with_catalogs.get_catalog(0.8)

    def test_get_catalog_error_lists_available_redshifts(self, factory_with_catalogs):
        """Error message should list available redshifts."""
        with pytest.raises(KeyError, match="0.5"):
            factory_with_catalogs.get_catalog(0.8)


class TestSerialization:
    """Tests for the save and load_catalogs methods of SnapshotCatalogFactory."""

    def test_save_creates_files(self, factory_with_catalogs, tmp_path):
        """Saving should create one file per catalog."""
        factory_with_catalogs.save(tmp_path)
        files = list(tmp_path.glob("catalog_z*.h5"))
        assert len(files) == 2

    def test_save_filenames_match_redshifts(self, factory_with_catalogs, tmp_path):
        """Saving should create files with names matching the redshifts."""
        factory_with_catalogs.save(tmp_path)
        assert (tmp_path / "catalog_z0.500.h5").exists()
        assert (tmp_path / "catalog_z1.000.h5").exists()

    def test_save_creates_directory(self, factory_with_catalogs, tmp_path):
        """Saving should create the output directory if it doesn't exist."""
        output = tmp_path / "new_dir" / "catalogs"
        factory_with_catalogs.save(output)
        assert output.exists()

    def test_load_catalogs_restores_redshifts(self, factory_with_catalogs, tmp_path, cosmo, cosmo_fid):
        """Loading should restore the same redshifts that were saved."""
        factory_with_catalogs.save(tmp_path)
        new_factory = SnapshotCatalogFactory(
            backend=factory_with_catalogs.backend,
            catalog_class=SnapshotCatalog,
            cosmo=cosmo,
            cosmo_fid=cosmo_fid,
        )
        new_factory.load_catalogs(tmp_path)
        assert set(new_factory.redshifts) == {0.5, 1.0}

    def test_load_catalogs_restores_tracer_data(self, factory_with_catalogs, tmp_path, cosmo, cosmo_fid):
        """Loading should restore the same tracer data that was saved."""
        factory_with_catalogs.save(tmp_path)
        new_factory = SnapshotCatalogFactory(
            backend=factory_with_catalogs.backend,
            catalog_class=SnapshotCatalog,
            cosmo=cosmo,
            cosmo_fid=cosmo_fid,
        )
        new_factory.load_catalogs(tmp_path)
        assert "FOO" in new_factory.get_catalog(0.5).tracers

    def test_load_catalogs_empty_directory_raises(self, factory, tmp_path):
        """Loading from an empty directory should raise a FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            factory.load_catalogs(tmp_path)

    def test_save_load_roundtrip_ngal(self, factory_with_catalogs, tmp_path, cosmo, cosmo_fid):
        """Saving and loading should preserve the number of galaxies in the catalog."""
        original_ngal = factory_with_catalogs.get_catalog(0.5).ngal
        factory_with_catalogs.save(tmp_path)
        new_factory = SnapshotCatalogFactory(
            backend=factory_with_catalogs.backend,
            catalog_class=SnapshotCatalog,
            cosmo=cosmo,
            cosmo_fid=cosmo_fid,
        )
        new_factory.load_catalogs(tmp_path)
        assert new_factory.get_catalog(0.5).ngal == original_ngal