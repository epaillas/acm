import h5py
import pandas as pd
import pytest

from acm.catalogs.dataclasses import Tracer, Transform
from acm.catalogs.products.base import BaseGalaxyCatalog

# ruff: noqa: ANN001, ANN201, ANN206, ARG002, ARG003, D101, D102, D103, INP001, S101

#%% Concrete dummy subclass

class DummyCatalog(BaseGalaxyCatalog):
    """Minimal concrete subclass for testing the base class."""

    required_columns = ("x", "y")

    def _check_data_columns(self, data: pd.DataFrame) -> bool:
        return set(self.required_columns).issubset(set(data.columns))

    def _save_attrs(self, f: h5py.File) -> None:
        f.attrs["dummy_attr"] = 42.0

    @classmethod
    def _from_attrs(cls, attrs, cosmo, cosmo_fid):
        return cls(cosmo=cosmo, cosmo_fid=cosmo_fid)


#%% Fixtures

@pytest.fixture
def catalog(cosmo_mock1, cosmo_mock2):
    return DummyCatalog(cosmo=cosmo_mock1, cosmo_fid=cosmo_mock2)

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

@pytest.fixture
def valid_data_bar():
    """Smaller dataset for BAR tracer to allow testing different counts."""
    return pd.DataFrame({"x": [5.0], "y": [6.0]})

@pytest.fixture
def multi_tracer_catalog(catalog, tracer_foo, tracer_bar, valid_data, valid_data_bar):
    catalog.set_tracer_data(tracer_foo, valid_data)       # 2 galaxies
    catalog.set_tracer_data(tracer_bar, valid_data_bar)  # 1 galaxy
    return catalog

#%% ngal
def test_ngal(populated_catalog):
    assert populated_catalog.ngal == 2

def test_ngal_empty_raises(catalog):
    """Ngal property should raise if no tracers are registered."""
    with pytest.raises(RuntimeError, match="No tracers"):
        _ = catalog.ngal

def test_ngal_multi_tracer(multi_tracer_catalog):
    """Ngal should sum galaxies across all tracers."""
    assert multi_tracer_catalog.ngal == 3

def test_ngal_per_tracer(multi_tracer_catalog):
    """_ngal should return the correct count per tracer independently."""
    assert multi_tracer_catalog._ngal("FOO") == 2
    assert multi_tracer_catalog._ngal("BAR") == 1
    assert multi_tracer_catalog._ngal("FOO", "BAR") == 3

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

    def test_get_tracer_data_empty_raises(self, catalog):
        """If no tracer are registered, get_tracer_data should reaise a RuntimeError."""
        with pytest.raises(RuntimeError, match="No tracers loaded"):
            catalog.get_tracer_data("FOO")

    def test_get_tracer_data_missing_raises(self, populated_catalog):
        """Calling get_tracer_data without tracer name should raise a ValueError."""
        with pytest.raises(ValueError, match="At least one tracer"):
            populated_catalog.get_tracer_data()

    def test_get_tracer_data_incorrect_raises(self, populated_catalog):
        """Calling get_tracer_data with a non-existing tracer should raise a KeyError."""
        with pytest.raises(KeyError, match="BAR"):
            populated_catalog.get_tracer_data("BAR")
        with pytest.raises(KeyError, match="BAR"):
            populated_catalog.get_tracer_data("FOO", "BAR")

    def test_get_tracer_data_duplicate_raises(self, populated_catalog, valid_data):
        """Passing the same tracer twice should return the dataframe with duplicate data."""
        with pytest.raises(ValueError, match="FOO"):
            populated_catalog.get_tracer_data("FOO", "FOO")

    def test_get_tracer_data_multi_tracer(self, multi_tracer_catalog, valid_data, valid_data_bar):
        """Multi-tracer retrieval must respect the requested order, not sort or deduplicate."""
        foo_bar = multi_tracer_catalog.get_tracer_data("FOO", "BAR")
        bar_foo = multi_tracer_catalog.get_tracer_data("BAR", "FOO")
        expected_foo_bar = pd.concat([valid_data, valid_data_bar], ignore_index=True)
        expected_bar_foo = pd.concat([valid_data_bar, valid_data], ignore_index=True)
        pd.testing.assert_frame_equal(foo_bar, expected_foo_bar)
        pd.testing.assert_frame_equal(bar_foo, expected_bar_foo)


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
        result = populated_catalog.get_tracer_data("FOO", raw=True)
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

    def test_transform_pipeline_property(self, catalog):
        t1 = Transform(name="t1", func=lambda d: d, kwargs={})
        t2 = Transform(name="t2", func=lambda d: d * 2, kwargs={})
        catalog._add_transform(t1)
        catalog._add_transform(t2)
        assert catalog.transform_pipeline == ["t1", "t2"]

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

    def test_clear_transforms(self, populated_catalog):
        t = Transform(name="t1", func=lambda d: d, kwargs={})
        populated_catalog._add_transform(t)
        populated_catalog.clear_transforms()
        assert len(populated_catalog._transforms) == 0

    def test_transform_default_tracer_none(self, populated_catalog, tracer_bar, valid_data):
        """A transform with tracer=None should apply to all tracers."""
        t = Transform(name="t1", func=lambda d, f: d * f, kwargs={"f": 2.0})
        populated_catalog.set_tracer_data(tracer_bar, valid_data) # Add BAR tracer
        populated_catalog._add_transform(t)
        bar_data = populated_catalog.get_tracer_data("BAR")
        foo_data = populated_catalog.get_tracer_data("FOO")
        assert populated_catalog._transforms["t1"].tracer is None
        pd.testing.assert_frame_equal(foo_data, valid_data * 2.0)
        pd.testing.assert_frame_equal(bar_data, valid_data * 2.0)

    def test_transform_with_tracer(self, populated_catalog, tracer_bar, valid_data):
        """A transform with a specific tracer should only apply to that tracer."""
        t = Transform(name="t1", func=lambda d, f: d * f, kwargs={"f": 2.0}, tracer="FOO")
        populated_catalog.set_tracer_data(tracer_bar, valid_data) # Add BAR tracer
        populated_catalog._add_transform(t)
        bar_data = populated_catalog.get_tracer_data("BAR")
        foo_data = populated_catalog.get_tracer_data("FOO")
        assert populated_catalog._transforms["t1"].tracer == "FOO"
        assert populated_catalog._transforms["t1"].tracer != "BAR"
        pd.testing.assert_frame_equal(foo_data, valid_data * 2.0)
        pd.testing.assert_frame_equal(bar_data, valid_data)

class TestMagicMethods:

    def test_getitem_single(self, populated_catalog, valid_data):
        pd.testing.assert_frame_equal(populated_catalog["FOO"], valid_data)

    def test_getitem_multiple(self, multi_tracer_catalog, valid_data, valid_data_bar):
        result = multi_tracer_catalog["FOO", "BAR"]
        expected = pd.concat([valid_data, valid_data_bar], ignore_index=True)
        pd.testing.assert_frame_equal(result, expected)

    def test_getitem_unknown_tracer_raises(self, populated_catalog):
        """Indexing an unknown tracer should propagate KeyError from get_tracer_data."""
        with pytest.raises(KeyError):
            _ = populated_catalog["UNKNOWN"]

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

    def test_save_load_roundtrip(self, populated_catalog, tmp_path, valid_data):
        path = tmp_path / "catalog.h5"
        populated_catalog.save(path)
        loaded = DummyCatalog.load(path)
        pd.testing.assert_frame_equal(loaded.get_tracer_data("FOO", raw=True), valid_data)

    def test_save_load_tracer_params(self, populated_catalog, tmp_path):
        path = tmp_path / "catalog.h5"
        populated_catalog.save(path)
        loaded = DummyCatalog.load(path)
        assert loaded.tracers["FOO"].params == {"a": 1}

    def test_save_load_subclass_attrs(self, populated_catalog, tmp_path):
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

    def test_load_transforms_not_restored(self, populated_catalog, tmp_path):
        """Transforms should not be present after loading."""
        t = Transform(name="scale", func=lambda d, f: d * f, kwargs={"f": 2.0})
        populated_catalog._add_transform(t)
        path = tmp_path / "catalog.h5"
        populated_catalog.save(path)
        loaded = DummyCatalog.load(path)
        assert len(loaded._transforms) == 0

    def test_save_load_multiple_tracers(self, catalog, tracer_foo, tracer_bar, valid_data, tmp_path):
        catalog.set_tracer_data(tracer_foo, valid_data)
        catalog.set_tracer_data(tracer_bar, valid_data)
        path = tmp_path / "catalog.h5"
        catalog.save(path)
        loaded = DummyCatalog.load(path)
        assert set(loaded.tracers.keys()) == {"FOO", "BAR"}

def test_save_load_preserves_cosmo(populated_catalog, tmp_path):
    """Cosmology values should be preserved through a save/load roundtrip."""
    path = tmp_path / "snapshot.h5"
    populated_catalog.save(path)
    loaded = DummyCatalog.load(path)
    assert loaded.cosmo != populated_catalog.cosmo  # Different instances
    assert loaded.cosmo.__class__ == populated_catalog.cosmo.__class__  # Same class
    for attr in ("_efunc", "_add"): # Same attributes should be equal
        assert getattr(loaded.cosmo, attr) == getattr(populated_catalog.cosmo, attr)
