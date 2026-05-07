import pytest
import pandas as pd
import h5py
from unittest.mock import MagicMock


from acm.catalogs.dataclasses import Tracer, Transform
from acm.catalogs.products.base import BaseGalaxyCatalog


#%% Concrete dummy subclass

class DummyCatalog(BaseGalaxyCatalog):
    """Minimal concrete subclass for testing the base class."""

    required_columns = {"x", "y"}

    def _check_data_columns(self, data: pd.DataFrame) -> bool:
        return self.required_columns.issubset(set(data.columns))

    def _save_attrs(self, f: h5py.File) -> None:
        f.attrs["dummy_attr"] = 42.0

    @classmethod
    def _from_attrs(cls, attrs, cosmo, cosmo_fid):
        return cls(cosmo=cosmo, cosmo_fid=cosmo_fid)


#%% Fixtures

@pytest.fixture
def cosmo():
    return MagicMock()

@pytest.fixture
def cosmo_fid():
    return MagicMock()

@pytest.fixture
def catalog(cosmo, cosmo_fid):
    return DummyCatalog(cosmo=cosmo, cosmo_fid=cosmo_fid)

@pytest.fixture
def tracer_foo():
    return Tracer(name="FOO", params={"a": 1})

@pytest.fixture
def tracer_bar():
    return Tracer(name="BAR", params={})

@pytest.fixture
def valid_data():
    return pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})

@pytest.fixture
def populated_catalog(catalog, tracer_foo, valid_data):
    catalog.set_tracer_data(tracer_foo, valid_data)
    return catalog


#%% Testing BaseGalaxyCatalog 

class TestTracers:

    def test_register_tracer(self, catalog, tracer_foo):
        catalog.register_tracer(tracer_foo)
        assert "FOO" in catalog.tracers

    def test_register_tracer_overwrite_warns(self, catalog, tracer_foo, caplog):
        catalog.register_tracer(tracer_foo)
        with caplog.at_level("WARNING"):
            catalog.register_tracer(tracer_foo)
        assert "FOO" in caplog.text

    def test_set_tracer_data_valid(self, catalog, tracer_foo, valid_data):
        catalog.set_tracer_data(tracer_foo, valid_data)
        assert "FOO" in catalog._data

    def test_set_tracer_data_registers_tracer(self, catalog, tracer_foo, valid_data):
        catalog.set_tracer_data(tracer_foo, valid_data)
        assert "FOO" in catalog.tracers

    def test_set_tracer_data_missing_columns_raises(self, catalog, tracer_foo):
        bad_data = pd.DataFrame({"x": [1.0]})  # missing "y"
        with pytest.raises(ValueError, match="missing required columns"):
            catalog.set_tracer_data(tracer_foo, bad_data)

    def test_get_tracer_data_returns_data(self, populated_catalog, valid_data):
        """If no transforms are registered, get_tracer_data should return the raw data."""
        result = populated_catalog.get_tracer_data("FOO")
        pd.testing.assert_frame_equal(result, valid_data)

    def test_get_tracer_data_missing_raises(self, catalog):
        with pytest.raises(KeyError, match="FOO"):
            catalog.get_tracer_data("FOO")
            
    def test_get_raw_tracer_data_missing_raises(self, catalog):
        with pytest.raises(KeyError, match="FOO"):
            catalog.get_raw_tracer_data("FOO")

class TestTransforms:

    def test_get_tracer_data_applies_transforms(self, populated_catalog):
        t = Transform(name="scale", func=lambda data, f: data * f, kwargs={"f": 2.0})
        populated_catalog._add_transform(t)
        result = populated_catalog.get_tracer_data("FOO")
        assert result["x"].tolist() == pytest.approx([2.0, 4.0])

    def test_get_tracer_data_does_not_mutate_raw(self, populated_catalog):
        raw_before = populated_catalog._data["FOO"].copy()
        t = Transform(name="scale", func=lambda data, f: data * f, kwargs={"f": 2.0})
        populated_catalog._add_transform(t)
        populated_catalog.get_tracer_data("FOO")
        pd.testing.assert_frame_equal(populated_catalog._data["FOO"], raw_before)

    def test_get_raw_tracer_data_bypasses_transforms(self, populated_catalog, valid_data):
        t = Transform(name="scale", func=lambda data, f: data * f, kwargs={"f": 2.0})
        populated_catalog._add_transform(t)
        result = populated_catalog.get_raw_tracer_data("FOO")
        pd.testing.assert_frame_equal(result, valid_data)

    def test_add_transform(self, catalog):
        t = Transform(name="t1", func=lambda d: d, kwargs={})
        catalog._add_transform(t)
        assert "t1" in catalog._transforms

    def test_add_transform_replaces_existing(self, catalog, caplog):
        t1 = Transform(name="t1", func=lambda d: d, kwargs={})
        t2 = Transform(name="t1", func=lambda d: d * 2, kwargs={})
        catalog._add_transform(t1)
        with caplog.at_level("WARNING"):
            catalog._add_transform(t2)
        assert catalog._transforms["t1"] is t2

    def test_remove_transform(self, catalog):
        t = Transform(name="t1", func=lambda d: d, kwargs={})
        catalog._add_transform(t)
        catalog._remove_transform("t1")
        assert "t1" not in catalog._transforms

    def test_remove_transform_missing_raises(self, catalog):
        with pytest.raises(KeyError, match="t1"):
            catalog._remove_transform("t1")

    def test_transforms_applied_in_order(self, populated_catalog):
        """Transforms should be applied sequentially in insertion order."""
        t1 = Transform(name="add", func=lambda d, v: d + v, kwargs={"v": 1.0})
        t2 = Transform(name="scale", func=lambda d, f: d * f, kwargs={"f": 2.0})
        populated_catalog._add_transform(t1)
        populated_catalog._add_transform(t2)
        result = populated_catalog.get_tracer_data("FOO") # (original + 1) * 2
        assert result["x"].tolist() == pytest.approx([(1.0 + 1.0) * 2.0, (2.0 + 1.0) * 2.0])


class TestMagicMethods:

    def test_getitem(self, populated_catalog, valid_data):
        pd.testing.assert_frame_equal(populated_catalog["FOO"], valid_data)

    def test_len_single_tracer(self, populated_catalog):
        assert len(populated_catalog) == 2

    def test_len_multiple_tracers(self, catalog, tracer_foo, tracer_bar, valid_data):
        catalog.set_tracer_data(tracer_foo, valid_data)
        catalog.set_tracer_data(tracer_bar, valid_data)
        assert len(catalog) == 4

    def test_len_empty(self, catalog):
        assert len(catalog) == 0

    def test_repr(self, populated_catalog):
        assert "FOO" in repr(populated_catalog)
        assert "DummyCatalog" in repr(populated_catalog)


class TestSaveLoad:

    def test_save_creates_file(self, populated_catalog, tmp_path):
        path = tmp_path / "catalog.h5"
        populated_catalog.save(path)
        assert path.exists()

    def test_save_load_roundtrip(self, populated_catalog, tmp_path, cosmo, cosmo_fid, valid_data):
        path = tmp_path / "catalog.h5"
        populated_catalog.save(path)
        loaded = DummyCatalog.load(path, cosmo, cosmo_fid)
        pd.testing.assert_frame_equal(loaded.get_raw_tracer_data("FOO"), valid_data)

    def test_save_load_tracer_params(self, populated_catalog, tmp_path, cosmo, cosmo_fid):
        path = tmp_path / "catalog.h5"
        populated_catalog.save(path)
        loaded = DummyCatalog.load(path, cosmo, cosmo_fid)
        assert loaded.tracers["FOO"].params == {"a": 1}

    def test_save_load_subclass_attrs(self, populated_catalog, tmp_path, cosmo, cosmo_fid):
        """Subclass-specific attrs saved by _save_attrs should be present in the file."""
        path = tmp_path / "catalog.h5"
        populated_catalog.save(path)
        with h5py.File(path, "r") as f:
            assert f.attrs["dummy_attr"] == 42.0

    def test_save_warns_on_active_transforms(self, populated_catalog, tmp_path, caplog):
        t = Transform(name="scale", func=lambda d, f: d * f, kwargs={"f": 2.0})
        populated_catalog._add_transform(t)
        path = tmp_path / "catalog.h5"
        with caplog.at_level("WARNING"):
            populated_catalog.save(path)
        assert "scale" in caplog.text

    def test_load_transforms_not_restored(self, populated_catalog, tmp_path, cosmo, cosmo_fid):
        """Transforms should not be present after loading."""
        t = Transform(name="scale", func=lambda d, f: d * f, kwargs={"f": 2.0})
        populated_catalog._add_transform(t)
        path = tmp_path / "catalog.h5"
        populated_catalog.save(path)
        loaded = DummyCatalog.load(path, cosmo, cosmo_fid)
        assert len(loaded._transforms) == 0

    def test_save_load_multiple_tracers(self, catalog, tracer_foo, tracer_bar, valid_data, tmp_path, cosmo, cosmo_fid):
        catalog.set_tracer_data(tracer_foo, valid_data)
        catalog.set_tracer_data(tracer_bar, valid_data)
        path = tmp_path / "catalog.h5"
        catalog.save(path)
        loaded = DummyCatalog.load(path, cosmo, cosmo_fid)
        assert set(loaded.tracers.keys()) == {"FOO", "BAR"}