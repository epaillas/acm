"""Tests for acm.observables.lsstypes."""
import logging

import lsstypes
import numpy as np
import pytest
from lsstypes import ObservableLeaf, ObservableTree

from acm.observables.lsstypes import (
    LsstypesObservable,
    _is_valid_tree,
    format_like,
    get_filter_indexes,
)

from .conftest import DummyModel

# ruff: noqa: ANN001, ANN201, D102, D103, S101

#%% Builders for minimal ObservableTree structures used in tests
def make_leaf(c, val) -> ObservableLeaf:
    """Make a simple ObservableLeaf with a 'c' coordinate and a single value array."""
    return ObservableLeaf(coords=["c"], c=np.asarray(c, dtype=float), val=np.asarray(val, dtype=float))

def make_y_sample(seed: int, g: tuple = (0, 1), n_c: int = 4) -> ObservableTree:
    """Make a simple y ObservableTree with a 'g' label and 'c' coordinate."""
    rng = np.random.default_rng(seed)
    c = np.linspace(0.0, 1.0, n_c)
    branches = [make_leaf(c, rng.normal(size=n_c)) for _ in g]
    return ObservableTree(branches, g=list(g))

def make_x_sample(p0: float, p1: float) -> ObservableTree:
    """Make a simple x ObservableTree with two parameters."""
    leaves = [ObservableLeaf(value=np.asarray(p0)), ObservableLeaf(value=np.asarray(p1))]
    return ObservableTree(leaves, parameters=["p0", "p1"])

def make_tree(n_samples: int = 3, n_test: int = 2, n_cov: int = 4, seed: int = 0) -> ObservableTree:
    """Build a minimal top-level tree with x, y, covariance_y, x_test, y_test branches."""
    rng = np.random.default_rng(seed)

    def build(n: int, offset: int) -> tuple[ObservableTree, ObservableTree]:
        x_branches = [make_x_sample(rng.normal(), rng.normal()) for _ in range(n)]
        y_branches = [make_y_sample(seed=offset + i) for i in range(n)]
        idx = list(range(n))
        return ObservableTree(x_branches, i=idx), ObservableTree(y_branches, i=idx)

    x, y = build(n_samples, 0)
    x_test, y_test = build(n_test, 1000)
    _, covariance_y = build(n_cov, 2000)
    return ObservableTree(
        [x, y, covariance_y, x_test, y_test],
        name=["x", "y", "covariance_y", "x_test", "y_test"],
    )

#%% Fixtures for LsstypesObservable tests
@pytest.fixture
def tree() -> ObservableTree:
    return make_tree()

@pytest.fixture
def obs(tree) -> LsstypesObservable:
    return LsstypesObservable(data=tree)

@pytest.fixture
def obs_with_model(tree) -> LsstypesObservable:
    n_features = np.asarray(next(iter(tree.get(name="y"))).value()).size
    return LsstypesObservable(data=tree, model=DummyModel(n_features=n_features))  # ty: ignore[invalid-argument-type]

#%% Tests
class TestIsValidTree:
    """Tests for the _is_valid_tree utility function."""

    def test_valid_tree_returns_true(self, tree):
        assert _is_valid_tree(tree)

    def test_missing_top_name_label_returns_false(self):
        leaf = ObservableLeaf(value=np.array(1.0))
        bare = ObservableTree([leaf], other=[0]) # Top-level label is not "name"
        assert not _is_valid_tree(bare)

    @pytest.mark.parametrize("drop", ["x", "y"])
    def test_missing_required_variable_returns_false(self, drop):
        full = make_tree()
        names = [n for n in ["x", "y", "covariance_y", "x_test", "y_test"] if n != drop]
        branches = [full.get(name=n) for n in names]
        partial = ObservableTree(branches, name=names)
        assert not _is_valid_tree(partial)

    def test_mismatched_x_y_labels_returns_false(self):
        """Mismatched x and y labels should return False."""
        x_branches = [make_x_sample(0.0, 0.0)]
        x = ObservableTree(x_branches, i=[0])
        y_branches = [make_y_sample(seed=0)]
        y = ObservableTree(y_branches, j=[0])  # different label key than x
        top = ObservableTree([x, y], name=["x", "y"])
        assert not _is_valid_tree(top)


class TestFormatLike:
    """Tests for the format_like utility function."""

    def test_builds_tree_indexed_by_array_rows(self, tree):
        y = tree.get(name="y")
        sample = next(iter(y)) # Using y-element structure as a template for the new tree branches
        arr = np.stack([np.asarray(sample.value())] * 3, axis=0)
        result = format_like(tree=sample, arr=arr, new="pred")
        assert "pred" in result.labels("unflatten")
        np.testing.assert_allclose(np.asarray(result.value(concatenate=False))[0], arr[0])

    def test_builds_tree_from_independent_array(self, tree):
        """Format like with an independant array matching the tree length."""
        y = tree.get(name="y")
        sample = next(iter(y))
        n_features = np.asarray(sample.value()).size
        rng = np.random.default_rng(1)
        arr = rng.normal(size=(4, n_features))
        result = format_like(tree=sample, arr=arr, new="pred")
        assert list(result.labels("unflatten")["pred"]) == [0, 1, 2, 3]
        np.testing.assert_allclose(np.asarray(result.value(concatenate=False))[2], arr[2])

    # @pytest.mark.parametrize("delta", [-1, 1], ids=["too_short", "too_long"]) #FIXME: add this if lsstypes internal bug fixed
    def test_incompatible_array_length_raises(self, tree, delta=-1):  # noqa: PT028
        """Array's per-row length must match the (single-sample) tree's flattened size."""
        y = tree.get(name="y")
        sample = next(iter(y))
        n_features = np.asarray(sample.value()).size
        arr = np.zeros((3, n_features + delta))  # wrong length per row
        with pytest.raises(ValueError, match="cannot reshape"):
            format_like(tree=sample, arr=arr, new="pred")

    def test_new_label_already_present_raises(self, tree):
        y = tree.get(name="y")
        sample = next(iter(y))
        arr = np.stack([np.asarray(sample.value())] * 3, axis=0)
        with pytest.raises(ValueError, match="Cannot use labels with same name at different levels"):
            format_like(tree=sample, arr=arr, new="g")  # "g" already exists as a label

class TestGetFilterIndexes:
    """Tests for the get_filter_indexes utility function."""

    def test_indexes_reproduce_filtered_values(self):
        og = make_y_sample(seed=0, g=(0, 1, 2))
        target = og.get(g=[0, 2])
        idx = get_filter_indexes(og, target)
        flat_og = np.asarray(og.value())
        flat_target = np.asarray(target.value())
        np.testing.assert_allclose(flat_og[idx], flat_target)


class TestInit:
    """Tests for the LsstypesObservable constructor."""

    def test_init_storage(self, tree):
        obs = LsstypesObservable(data=tree)
        assert obs._data is tree
        assert obs.model is None
        assert obs._filters_idx == {}

    def test_silent_load_suppresses_logging(self, tree, caplog):
        with caplog.at_level(logging.INFO):
            LsstypesObservable(data=tree, silent_load=True)
        assert not any("Tree loaded" in r.message for r in caplog.records)


class TestLoadCanLoad:
    """Tests for the LsstypesObservable.load and .can_load class methods."""

    def test_load_roundtrip(self, tree, tmp_path):
        fn = tmp_path / "obs.h5"
        lsstypes.write(str(fn), tree)
        loaded = LsstypesObservable.load(fn)
        assert isinstance(loaded, LsstypesObservable)
        assert loaded._data == tree # The tree structure should be identical

    def test_load_invalid_tree_raises(self, tmp_path):
        leaf = ObservableLeaf(value=np.array(1.0))
        bad = ObservableTree([ObservableTree([leaf], parameters=["p0"])], name=["x"])
        fn = tmp_path / "bad.h5"
        lsstypes.write(str(fn), bad)
        with pytest.raises(ValueError, match="Invalid Observable structure"):
            LsstypesObservable.load(fn)

    def test_can_load(self, tree, tmp_path):
        good_fn = tmp_path / "good.h5"
        lsstypes.write(str(good_fn), tree)
        assert LsstypesObservable.can_load(good_fn) is True

    def test_can_load_invalid_returns_false(self, tmp_path):
        leaf = ObservableLeaf(value=np.array(1.0))
        bad = ObservableTree([ObservableTree([leaf], parameters=["p0"])], name=["x"])
        bad_fn = tmp_path / "bad.h5"
        lsstypes.write(str(bad_fn), bad)
        assert LsstypesObservable.can_load(bad_fn) is False

    def test_can_load_missing_file_returns_false(self, tmp_path):
        assert LsstypesObservable.can_load(tmp_path / "missing.h5") is False

    def test_can_load_false_logs(self, tmp_path, caplog):
        fn = tmp_path / "garbage.h5"
        fn.write_bytes(b"not a real hdf5 file")
        with caplog.at_level(logging.DEBUG):
            LsstypesObservable.can_load(fn)
        assert any("Failed to load" in r.message for r in caplog.records)

    def test_can_load_garbage_file_returns_false(self, tmp_path):
        fn = tmp_path / "garbage.h5"
        fn.write_bytes(b"not a real hdf5 file")
        assert LsstypesObservable.can_load(fn) is False

class TestCopy:
    """Tests for the LsstypesObservable._copy method."""

    @pytest.mark.parametrize("deep", [False, True])
    def test_copy_and_deepcopy_are_independent(self, obs, deep):
        cp = obs._copy(deep=deep)
        cp.set_filters(i=[0])
        assert obs.filters == {}
        assert cp.filters == {"i": [0]}


class TestXNames:
    """Tests for the LsstypesObservable.x_names property."""

    def test_default_order(self, obs):
        assert obs.x_names == ["p0", "p1"]

    def test_filtered_order(self, obs):
        obs.set_filters(parameters=["p1", "p0"])
        assert obs.x_names == ["p0", "p1"] # Also preserved

    def test_filtered_by_parameters_filter(self, obs):
        obs.set_filters(parameters=["p1"])
        assert obs.x_names == ["p1"]


class TestGetCoordinateList:
    """Tests for the LsstypesObservable.get_coordinate_list method."""

    def test_structural_label(self, obs):
        assert obs.get_coordinate_list("g") == [0, 1]

    def test_leaf_coordinate(self, obs):
        c = obs.get_coordinate_list("c")
        assert len(c) == 4

    def test_respects_active_filters(self, obs):
        obs.set_filters(g=[0])
        assert obs.get_coordinate_list("g") == [0]

    def test_unknown_name_raises_keyerror(self, obs):
        with pytest.raises(KeyError):
            obs.get_coordinate_list("bogus")


class TestFiltersSetter:
    """Tests for the LsstypesObservable.set_filters method."""

    def test_sets_filters_idx_only_for_affected_names(self, obs):
        obs.set_filters(g=[0]) # Only y and covariance_y have a g label
        assert "y" in obs._filters_idx
        assert "covariance_y" in obs._filters_idx
        assert "x" not in obs._filters_idx

    def test_clearing_resets_filters_idx(self, obs):
        obs.set_filters(g=[0])
        assert obs._filters_idx
        obs.clear_filters()
        assert obs._filters_idx == {}


class TestApplyFilters:
    """Tests for the LsstypesObservable._apply_filters method."""

    def test_no_filters_returns_input(self, obs):
        y = obs.get_data("y", raw=True)
        filtered = obs._apply_filters(y)
        assert filtered is y

    def test_label_filter(self, obs):
        obs.set_filters(g=[0])
        filtered = obs._apply_filters(obs.get_data("y", raw=True))
        g_values = filtered.labels("unflatten", level=None)["g"] # All filtered g values
        assert list(g_values) == [0] * len(g_values)

    def test_coordinate_filter(self, obs):
        obs.set_filters(c=(0.0, 0.5))
        filtered = obs._apply_filters(obs.get_data("y", raw=True))
        y_default = obs.get_data("y", raw=True)
        assert filtered != y_default
        for sample in filtered: # Both g=0 and g=1 branches should be filtered
            for g in [0, 1]:
                c_values = np.asarray(sample.get(g=g).c)
                assert c_values.min() >= 0.0
                assert c_values.max() <= 0.5

    def test_slice_step_is_dropped(self, obs):
        """slice(0, 0.5) and slice(0, 0.5, 5) must filter identically, since step is ignored."""
        obs.set_filters(c=slice(0.0, 0.5))
        no_step = obs._apply_filters(obs.get_data("y", raw=True))

        obs.set_filters(c=slice(0.0, 0.5, 5))
        with_step = obs._apply_filters(obs.get_data("y", raw=True))

        c_no_step = np.asarray(next(iter(no_step)).get(g=0).c)
        c_with_step = np.asarray(next(iter(with_step)).get(g=0).c)
        np.testing.assert_array_equal(c_no_step, c_with_step)

    def test_slice_on_label_raises(self, obs): # lsstypes behavior
        with pytest.raises(ValueError, match="not found"):
            obs.set_filters(g=slice(0, 1))

    @pytest.mark.xfail(reason="lsstypes bug: unknown filter name does not raise on select, but raises on get")
    def test_unknown_filter_name_raises_keyerror(self, obs):
        obs.set_filters(bogus=[0])
        with pytest.raises(KeyError):
            obs._apply_filters(obs.get_data("y", raw=True))

    def test_sample_level_filter_reduces_row_count(self, obs):
        obs.set_filters(i=[0])
        assert obs.get_data("y").shape[0] == 1
        assert obs.get_data("x").shape[0] == 1


class TestToNumpy:
    """Tests for the LsstypesObservable._to_numpy method."""

    def test_default_flattens(self, obs):
        y = obs.get_data("y", raw=True)
        arr = obs._to_numpy(y)
        assert arr.ndim == 2
        assert arr.shape == (3, 8)  # n_samples, n_features

    def test_nested_preserves_structure(self, obs):
        y = obs.get_data("y", raw=True)
        arr = obs._to_numpy(y, nested=True)
        assert np.asarray(arr).shape == (3, 2, 4)  # n_samples, n_g, n_c


class TestFilter2D:
    """Tests for the LsstypesObservable._filter_2d method."""

    def test_applies_precomputed_index(self, obs):
        obs.set_filters(g=[0])
        data = obs._to_numpy(obs.get_data("y", raw=True))
        filtered = obs._filter_2d(data, name="y")
        assert filtered.shape[-1] < data.shape[-1]

    def test_passthrough_when_no_index_registered(self, obs):
        data = obs._to_numpy(obs.get_data("y", raw=True))
        result = obs._filter_2d(data, name="y")
        np.testing.assert_array_equal(result, data)

    def test_nested_skips_selection(self, obs):
        obs.set_selection("y", indices=[0])
        data = obs._to_numpy(obs.get_data("y", raw=True))
        result = obs._filter_2d(data, name="y", nested=True)
        np.testing.assert_array_equal(result, data)


class TestGetData:
    """Tests for the LsstypesObservable.get_data method."""

    @pytest.mark.parametrize(("raw", "nested"), [(False, False), (False, True), (True, False)])
    def test_default_raw_nested(self, obs, raw, nested):
        result = obs.get_data("y", raw=raw, nested=nested)
        if raw:
            assert isinstance(result, ObservableTree)
        elif nested:
            assert np.asarray(result).shape == (3, 2, 4)  # n_samples, n_g, n_c
        else:
            assert np.asarray(result).shape == (3, 8)  # n_samples, n_features
            assert result.ndim == 2

    def test_unknown_name_raises_keyerror(self, obs):
        with pytest.raises(KeyError):
            obs.get_data("bogus")


class TestGetAttr:
    """Tests for the LsstypesObservable.__getattr__ method."""

    def test_data_variable_shortcuts_to_get_data(self, obs):
        np.testing.assert_array_equal(obs.y, obs.get_data("y"))

    @pytest.mark.parametrize("name", ["_ipython_canary_method_should_not_exist_", "bogus_attr"])
    def test_unknown_attribute_raises(self, obs, name):
        with pytest.raises(AttributeError):
            getattr(obs, name)

    def test_passthrough_with_filters(self, obs):
        obs.set_filters(g=[0])
        g_values = obs.labels("unflatten", level=None)["g"]
        assert set(g_values) == {0}


class TestGetTestSet:
    """Tests for the LsstypesObservable.get_test_set method."""

    def test_returns_2d_x_and_y(self, obs):
        x_test, y_test = obs.get_test_set()
        assert x_test.shape == (2, 2)  # n_test, n_params
        assert y_test.shape == (2, 8)  # n_test, n_features


class TestGetPrediction:
    """Tests for the LsstypesObservable.get_prediction method."""

    def test_no_model_raises(self, obs):
        with pytest.raises(AttributeError):
            obs.get_prediction(np.zeros((2, 2)))

    def test_default_shape_matches_y(self, obs_with_model):
        x = np.zeros((2, 2))
        pred = obs_with_model.get_prediction(x)
        y = obs_with_model.get_data("y")
        assert pred.shape == (2, y.shape[-1])

    def test_raw_returns_tree(self, obs_with_model):
        pred = obs_with_model.get_prediction(np.zeros((2, 2)), raw=True)
        assert isinstance(pred, ObservableTree)

    def test_filters_and_selection_like_y(self, obs_with_model):
        obs_with_model.set_filters(g=[0])
        obs_with_model.set_selection("y", indices=[0, 2])
        pred = obs_with_model.get_prediction(np.zeros((2, 2)))
        y = obs_with_model.get_data("y")
        assert pred.shape[-1] == y.shape[-1]

    def test_nested_filters_no_selection(self, obs_with_model):
        obs_with_model.set_filters(g=[0], c=slice(0.0, 0.5))
        obs_with_model.set_selection("y", indices=[0, 2])
        pred_nested = obs_with_model.get_prediction(np.zeros((2, 2)), nested=True) # skips selection
        assert pred_nested.shape == (2, 1, 2)  # n_pred, n_g(selected), n_c(selected)


class TestGetModelError:
    """Tests for the LsstypesObservable.get_model_error method."""

    def test_no_model_raises(self, obs):
        with pytest.raises(AttributeError):
            obs.get_model_error(method="median")

    def test_matches_manual_computation(self, obs_with_model):
        error = obs_with_model.get_model_error(method="median")
        x_test, y_test = obs_with_model.get_test_set()
        pred = obs_with_model.model.get_prediction(x_test)
        expected = np.median(np.abs(y_test - pred), axis=0) # shape is (n_features,)
        np.testing.assert_allclose(error, expected)

    @pytest.mark.parametrize("method", ["median", "stdev"])
    def test_raw_returns_tree(self, obs_with_model, method):
        """Test that raw=True returns an ObservableTree, and that the values match the non-raw output."""
        kwargs = {} if method == "median" else {"diag": True} # stdev calls covariance
        error = obs_with_model.get_model_error(method=method, raw=True, **kwargs)
        y_sample = next(iter(obs_with_model.get_data("y", raw=True)))
        assert isinstance(error, type(y_sample))
        np.testing.assert_allclose(
            np.asarray(error.value()),
            obs_with_model.get_model_error(method=method, **kwargs),
        ) # No filters - values should match

    def test_nested_filters_no_selection(self, obs_with_model):
        obs_with_model.set_filters(g=[0], c=slice(0.0, 0.5))
        obs_with_model.set_selection("y", indices=[0, 2])
        error_nested = obs_with_model.get_model_error(method="median", nested=True) # skips selection
        assert error_nested.shape == (1, 2)  # n_g(selected), n_c(selected)


class TestGetModelCovariance:
    """Tests for the LsstypesObservable.get_model_covariance method."""

    def test_no_model_raises(self, obs):
        with pytest.raises(AttributeError):
            obs.get_model_covariance()

    def test_matches_manual_computation(self, obs_with_model):
        cov = obs_with_model.get_model_covariance(prefactor=2.0)
        x_test, y_test = obs_with_model.get_test_set()
        pred = obs_with_model.model.get_prediction(x_test)
        diff = y_test - pred
        expected = 2.0 * np.cov(diff, rowvar=False) # shape is (n_features, n_features)
        np.testing.assert_allclose(cov, expected)


@pytest.fixture
def single_sample_obs() -> LsstypesObservable:
    return LsstypesObservable(data=make_tree(n_samples=1, n_test=1, n_cov=2))

class TestSingleSampleTree:
    """Tests for LsstypesObservable with a single-sample tree."""

    def test_get_data_shapes(self, single_sample_obs):
        assert single_sample_obs.get_data("x").shape == (1, 2)  # n_samples, n_params
        assert single_sample_obs.get_data("y").shape == (1, 8)  # n_samples, n_features

    def test_x_names_and_coordinate_list_unaffected(self, single_sample_obs):
        assert single_sample_obs.x_names == ["p0", "p1"]
        assert set(single_sample_obs.get_coordinate_list("g")) == {0, 1}

    def test_apply_filters_is_identity_with_no_filters_set(self, single_sample_obs):
        """Regression test: .get()/.select() must be skipped entirely when there's nothing to filter."""
        x_tree = single_sample_obs.get_data("x", raw=True)
        filtered = single_sample_obs._apply_filters(x_tree)
        assert filtered.labels("unflatten").keys() == x_tree.labels("unflatten").keys()
        np.testing.assert_allclose(np.asarray(filtered.value()), np.asarray(x_tree.value()))
