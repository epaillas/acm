# tests/observables/test_xarray.py
"""Tests for acm.observables.xarray."""
import logging
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from acm.observables.xarray import (
    XarrayObservable,
    _is_valid_dataset,
    _load_dataset,
    _stack_on,
    format_like,
)

from .conftest import DummyModel

# ruff: noqa: ANN001, ANN201, D102, D103, S101


#%% Builders for minimal xr.Dataset structures used in tests
def make_dataset(n_samples=3, n_test=1, n_cov=4, n_g=2, n_c=4, seed=0) -> xr.Dataset:
    """Build a minimal dataset with x, y, covariance_y, x_test, y_test.

    x_test/y_test are built as a .sel() subset of x/y (mirroring helpers.py), so combining
    them into one Dataset triggers real xarray NaN-padding on the shared "i" dim.
    """
    rng = np.random.default_rng(seed)
    g = list(range(n_g))
    c = np.linspace(0.0, 1.0, n_c)

    y = xr.DataArray(
        rng.normal(size=(n_samples, n_g, n_c)),
        dims=["i", "g", "c"],
        coords={"i": np.arange(n_samples), "g": g, "c": c},
        attrs={"sample": ["i"], "features": ["g", "c"]},
        name="y",
    )
    x = xr.DataArray(
        rng.normal(size=(n_samples, 2)),
        dims=["i", "parameters"],
        coords={"i": np.arange(n_samples), "parameters": ["p0", "p1"]},
        attrs={"sample": ["i"], "features": ["parameters"]},
        name="x",
    )
    covariance_y = xr.DataArray(
        rng.normal(size=(n_cov, n_g, n_c)),
        dims=["j", "g", "c"],
        coords={"j": np.arange(n_cov), "g": g, "c": c},
        attrs={"sample": ["j"], "features": ["g", "c"]},
        name="covariance_y",
    )
    test_idx = list(range(n_test))
    x_test = x.sel(i=test_idx).copy()
    y_test = y.sel(i=test_idx).copy()
    x_test.attrs["nan_dims"] = ["i"]
    y_test.attrs["nan_dims"] = ["i"]
    x_test.name = "x_test"
    y_test.name = "y_test"

    return xr.Dataset({
        "x": x, "y": y, "covariance_y": covariance_y,
        "x_test": x_test, "y_test": y_test,
    })

#%% Fixtures
@pytest.fixture
def dataset() -> xr.Dataset:
    return make_dataset()

@pytest.fixture
def obs(dataset) -> XarrayObservable:
    return XarrayObservable(data=dataset)

@pytest.fixture
def obs_with_model(dataset) -> XarrayObservable:
    n_features = dataset["y"].sizes["g"] * dataset["y"].sizes["c"]
    return XarrayObservable(data=dataset, model=DummyModel(n_features=n_features))  # ty: ignore[invalid-argument-type]

#%% Tests
class TestLoadDataset:
    """Tests for the _load_dataset utility function."""

    def test_loads_h5_file(self, dataset, tmp_path):
        fn = tmp_path / "obs.h5"
        dataset.to_netcdf(fn)
        loaded = _load_dataset(fn)
        assert set(loaded.data_vars) == set(dataset.data_vars)

    def test_loads_legacy_npy_file(self, tmp_path):
        """Checks the .npy branch calls dataset_from_dict, without depending on its real impl."""
        fn = tmp_path / "legacy.npy"
        vals = np.array({'x': [1, 2, 3]})
        np.save(fn, vals, allow_pickle=True)
        with patch("acm.observables.xarray.dataset_from_dict", return_value=xr.Dataset({"x": xr.DataArray([1, 2, 3])})) as func:
            result = _load_dataset(fn)
            func.assert_called_once_with(vals)
            assert "x" in result.data_vars


class TestIsValidDataset:
    """Tests for the _is_valid_dataset utility function."""

    def test_valid_returns_true(self, dataset):
        assert _is_valid_dataset(dataset)

    @pytest.mark.parametrize("drop", ["x", "y"])
    def test_missing_required_variable_returns_false(self, dataset, drop):
        assert not _is_valid_dataset(dataset.drop_vars(drop))

    @pytest.mark.parametrize("missing_attr", ["sample", "features"])
    def test_missing_required_attrs_returns_false(self, dataset, missing_attr):
        broken = dataset.copy()
        da = broken["y"].copy()
        del da.attrs[missing_attr]
        broken["y"] = da
        assert not _is_valid_dataset(broken)


class TestStackOn:
    """Tests for the _stack_on utility function."""

    def test_already_existing_dim_returns_unchanged(self, dataset):
        da = dataset["y"]
        result = _stack_on("i", da)
        assert result is da

    def test_no_dims_expands_new_dim(self, dataset):
        da = dataset["x_test"].isel(i=0)
        result = _stack_on("new", da)
        assert "new" in result.dims

    def test_stacks_multiple_dims(self, dataset):
        da = dataset["y"]
        result = _stack_on("features", da, "g", "c")
        assert "features" in result.dims
        assert result.sizes["features"] == da.sizes["g"] * da.sizes["c"]


class TestFormatLike:
    """Tests for the format_like utility function."""

    def test_builds_dataarray_matching_reference_shape(self, dataset):
        y = dataset["y"]
        arr = np.zeros((2, y.sizes["g"] * y.sizes["c"]))
        result = format_like(da=y, arr=arr, new="pred")
        assert result.sizes == {"pred": 2, "g": y.sizes["g"], "c": y.sizes["c"]}

    @pytest.mark.parametrize("delta", [-1, 1], ids=["too_short", "too_long"])
    def test_incompatible_array_length_raises(self, dataset, delta):
        y = dataset["y"]
        n_features = y.sizes["g"] * y.sizes["c"]
        arr = np.zeros((2, n_features + delta))
        with pytest.raises(ValueError, match="cannot reshape"):
            format_like(da=y, arr=arr, new="pred")


class TestInit:
    """Tests for the XarrayObservable constructor."""

    def test_init_storage(self, dataset):
        obs = XarrayObservable(data=dataset)
        assert obs._data is dataset
        assert obs.model is None

    def test_silent_load_suppresses_logging(self, dataset, caplog):
        with caplog.at_level(logging.INFO):
            XarrayObservable(data=dataset, silent_load=True)
        assert not any("Datasets loaded" in r.message for r in caplog.records)


class TestLoadCanLoad:
    """Tests for the XarrayObservable.load and .can_load class methods."""

    def test_load_roundtrip(self, dataset, tmp_path):
        fn = tmp_path / "obs.h5"
        dataset.to_netcdf(fn)
        loaded = XarrayObservable.load(fn)
        np.testing.assert_allclose(loaded.get_data("y"), XarrayObservable(data=dataset).get_data("y"))

    def test_load_invalid_raises(self, tmp_path):
        bad = xr.Dataset({"x": xr.DataArray([1.0], dims=["i"], attrs={"sample": ["i"], "features": []})})
        fn = tmp_path / "bad.h5"
        bad.to_netcdf(fn)
        with pytest.raises(ValueError, match="Invalid Observable structure"):
            XarrayObservable.load(fn)

    def test_can_load(self, dataset, tmp_path):
        good_fn = tmp_path / "good.h5"
        dataset.to_netcdf(good_fn)
        assert XarrayObservable.can_load(good_fn) is True

    def test_can_load_invalid_returns_false(self, tmp_path):
        bad = xr.Dataset({"x": xr.DataArray([1.0], dims=["i"], attrs={"sample": ["i"], "features": []})})
        bad_fn = tmp_path / "bad.h5"
        bad.to_netcdf(bad_fn)
        assert XarrayObservable.can_load(bad_fn) is False

    def test_can_load_missing_returns_false(self, tmp_path):
        assert XarrayObservable.can_load(tmp_path / "missing.h5") is False

    def test_can_load_false_logs(self, tmp_path, caplog):
        fn = tmp_path / "garbage.h5"
        fn.write_bytes(b"not a real hdf5 file")
        with caplog.at_level(logging.DEBUG):
            XarrayObservable.can_load(fn)
        assert any("Failed to load" in r.message for r in caplog.records)


class TestCopy:
    """Tests for the XarrayObservable._copy method."""

    @pytest.mark.parametrize("deep", [False, True])
    def test_copy_and_deepcopy_are_independent(self, obs, deep):
        cp = obs._copy(deep=deep)
        cp.set_filters(i=[0])
        assert obs.filters == {}
        assert cp.filters == {"i": [0]}

class TestXNames:
    """Tests for the XarrayObservable.x_names property."""

    def test_default_order(self, obs):
        assert obs.x_names == ["p0", "p1"]

    @pytest.mark.xfail(reason="xarray does not preserve order of selection")
    def test_filtered_order(self, obs):
        obs.set_filters(parameters=["p1", "p0"])
        assert obs.x_names == ["p0", "p1"] # Also preserved

    def test_filtered_by_parameters_filter(self, obs):
        obs.set_filters(parameters=["p1"])
        assert obs.x_names == ["p1"]


class TestGetCoordinateList:
    """Tests for the XarrayObservable.get_coordinate_list method and x_names property."""

    def test_returns_full_list(self, obs):
        assert obs.get_coordinate_list("g") == [0, 1]

    def test_respects_active_filters(self, obs):
        obs.set_filters(g=[0])
        assert obs.get_coordinate_list("g") == [0]

    def test_unknown_name_raises_keyerror(self, obs):
        with pytest.raises(KeyError):
            obs.get_coordinate_list("bogus")


class TestDropNanDims:
    """Tests for the XarrayObservable._drop_nan_dims static method."""

    def test_drops_nan_padded_entries(self, obs):
        y_test_raw = obs.get_data("y_test", raw=True)
        assert y_test_raw.sizes["i"] == 3  # padded up to full "i" range
        dropped = obs._drop_nan_dims(y_test_raw)
        assert dropped.sizes["i"] == 1  # back down to actual test count

    def test_no_nan_dims_attr_returns_unchanged(self, obs):
        y = obs.get_data("y", raw=True)
        result = obs._drop_nan_dims(y)
        assert result.sizes == y.sizes

    def test_all_dropped_logs_warning(self, caplog):
        da = xr.DataArray(
            np.full((2, 2, 2), np.nan),
            dims=["i", "g", "c"],
            attrs={"sample": ["i"], "features": ["g", "c"], "nan_dims": ["i"]},
            name="y_test",
        )
        with caplog.at_level(logging.WARNING):
            result = XarrayObservable._drop_nan_dims(da)
        assert result.size == 0
        assert any("dropped due to NaN" in r.message for r in caplog.records)


class TestApplyFilters:
    """Tests for the XarrayObservable._apply_filters method."""

    def test_no_filters_returns_input(self, obs):
        y = obs.get_data("y", raw=True)
        filtered = obs._apply_filters(y)
        xr.testing.assert_equal(filtered, y)

    def test_filters_matching_dims_applied(self, obs):
        obs.set_filters(g=[0])
        filtered = obs._apply_filters(obs.get_data("y", raw=True))
        assert list(filtered.coords["g"].to_numpy()) == [0]

    def test_filters_not_in_dims_silently_skipped(self, obs):
        obs.set_filters(bogus=[0])
        filtered = obs._apply_filters(obs.get_data("y", raw=True))
        np.testing.assert_allclose(filtered.to_numpy(), obs.get_data("y", raw=True).to_numpy())

    def test_native_slice_filter(self, obs):
        obs.set_filters(c=slice(0.0, 0.5))
        filtered = obs._apply_filters(obs.get_data("y", raw=True))
        assert filtered.coords["c"].to_numpy().max() <= 0.5


class TestToNumpy:
    """Tests for the XarrayObservable._to_numpy method."""

    def test_default_flattens(self, obs):
        y = obs.get_data("y", raw=True)
        arr = obs._to_numpy(y)
        assert arr.shape == (3, 2 * 4)  # n_samples, n_g * n_c

    def test_nested_preserves_structure(self, obs):
        y = obs.get_data("y", raw=True)
        arr = obs._to_numpy(y, nested=True)
        assert arr.shape == (3, 2, 4)

    def test_single_string_attr_edge_case(self):
        """Covers the isinstance(attr, str) branch, when sample/features are single dim names."""
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["i", "c"],
            attrs={"sample": "i", "features": "c"},
            name="y",
        )
        arr = XarrayObservable._to_numpy(da)
        assert arr.shape == (3, 4)


class TestFormatDataOverride:
    """Tests for the XarrayObservable._format_data override."""

    def test_drops_nan_before_base_formatting(self, obs):
        y_test = obs.get_data("y_test", raw=True)
        result = obs._format_data(y_test, "y_test", nested=False)
        assert result.shape[0] == 1  # nan-dropped sample count, not the padded count


class TestGetData:
    """Tests for the XarrayObservable.get_data method."""

    @pytest.mark.parametrize(("raw", "nested"), [(False, False), (False, True), (True, False)])
    def test_default_raw_nested(self, obs, raw, nested):
        result = obs.get_data("y", raw=raw, nested=nested)
        if raw:
            assert isinstance(result, xr.DataArray)
        elif nested:
            assert result.shape == (3, 2, 4)
        else:
            assert result.ndim == 2

    def test_unknown_name_raises_keyerror(self, obs):
        with pytest.raises(KeyError):
            obs.get_data("bogus")


class TestGetAttr:
    """Tests for the XarrayObservable.__getattr__ method."""

    def test_data_variable_shortcuts_to_get_data(self, obs):
        np.testing.assert_array_equal(obs.y, obs.get_data("y"))

    @pytest.mark.parametrize("name", ["_ipython_canary_method_should_not_exist_", "bogus_attr"])
    def test_unknown_attribute_raises(self, obs, name):
        with pytest.raises(AttributeError):
            getattr(obs, name)

    def test_passthrough_with_filters(self, obs):
        obs.set_filters(g=[0])
        assert obs.sizes["g"] == 1


class TestGetTestSet:
    """Tests for the XarrayObservable.get_test_set method."""

    def test_returns_2d_arrays_with_nan_dropped(self, obs):
        x_test, y_test = obs.get_test_set()
        assert x_test.shape[0] == 1 # nan-dropped
        assert y_test.shape[0] == 1
        assert x_test.ndim == 2 # 2d
        assert y_test.ndim == 2


class TestGetPrediction:
    """Tests for the XarrayObservable.get_prediction method."""

    def test_no_model_raises(self, obs):
        with pytest.raises(AttributeError):
            obs.get_prediction(np.zeros((2, 2)))

    def test_default_shape_matches_y(self, obs_with_model):
        pred = obs_with_model.get_prediction(np.zeros((2, 2)))
        y = obs_with_model.get_data("y")
        assert pred.shape == (2, y.shape[-1])

    def test_raw_returns_dataarray(self, obs_with_model):
        pred = obs_with_model.get_prediction(np.zeros((2, 2)), raw=True)
        assert isinstance(pred, xr.DataArray)

    def test_respects_filters_and_selection(self, obs_with_model):
        obs_with_model.set_filters(g=[0])
        obs_with_model.set_selection("y", indices=[0, 2])
        pred = obs_with_model.get_prediction(np.zeros((2, 2)))
        y = obs_with_model.get_data("y")
        assert pred.shape[-1] == y.shape[-1]

    def test_nested_filters_no_selection(self, obs_with_model):
        obs_with_model.set_filters(g=[0], c=slice(0.0, 0.5))
        obs_with_model.set_selection("y", indices=[0, 2])
        pred_nested = obs_with_model.get_prediction(np.zeros((2, 2)), nested=True) # skips selection
        assert pred_nested.shape == (2, 1, 2) # n_pred, n_g, n_c


class TestGetModelError:
    """Tests for the XarrayObservable.get_model_error method."""

    def test_no_model_raises(self, obs):
        with pytest.raises(AttributeError):
            obs.get_model_error(method="median")

    def test_matches_manual_computation(self, obs_with_model):
        error = obs_with_model.get_model_error(method="median")
        x_test, y_test = obs_with_model.get_test_set()
        pred = obs_with_model.model.get_prediction(x_test)
        expected = np.median(np.abs(y_test - pred), axis=0)
        np.testing.assert_allclose(error, expected)

    @pytest.mark.parametrize("method", ["median", "stdev"])
    def test_raw_returns_dataarray(self, obs_with_model, method):
        error = obs_with_model.get_model_error(method=method, raw=True)
        assert isinstance(error, xr.DataArray)

    def test_nested_filters_no_selection(self, obs_with_model):
        obs_with_model.set_filters(g=[0], c=slice(0.0, 0.5))
        obs_with_model.set_selection("y", indices=[0, 2])
        error_nested = obs_with_model.get_model_error(method="median", nested=True) # skips selection
        assert error_nested.shape == (1, 2) # n_g(selected), n_c(selected)


class TestGetModelCovariance:
    """Tests for the XarrayObservable.get_model_covariance method."""

    def test_no_model_raises(self, obs):
        with pytest.raises(AttributeError):
            obs.get_model_covariance()

    #NOTE: ignoring invalid covariance warnings because I don't want to find a good set of values ATMs
    @pytest.mark.filterwarnings("ignore:.*encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:.*Degrees of freedom:RuntimeWarning")
    def test_matches_manual_computation(self, obs_with_model):
        cov = obs_with_model.get_model_covariance(prefactor=2.0)
        x_test, y_test = obs_with_model.get_test_set()
        pred = obs_with_model.model.get_prediction(x_test)
        diff = y_test - pred
        expected = 2.0 * np.cov(diff, rowvar=False) # shape is (n_features, n_features)
        np.testing.assert_allclose(cov, expected)
