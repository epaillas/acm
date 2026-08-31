"""Tests for acm.observables.base."""
import copy

import numpy as np
import pytest

from acm.observables.base import BaseObservable, _format_filter_value, make_handle

# ruff: noqa: ANN001, ANN201, ARG002, D102, S101

class DummyObservable(BaseObservable[np.ndarray]):
    """Minimal concrete BaseObservable, decoupled from any real backend.

    R = plain 3D np.ndarray (n_samples, a, b). Filtering masks samples by "i";
    flattening merges the last two axes unless nested=True.
    """

    def __init__(self, data: dict[str, np.ndarray], model=None) -> None:
        self._data = data
        super().__init__(model=model)

    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        idx = self.filters.get("i")
        return data if idx is None else data[idx]

    @staticmethod
    def _to_numpy(data: np.ndarray, nested: bool = False) -> np.ndarray:
        return data if nested else data.reshape(data.shape[0], -1)

    def get_data(self, name: str, raw: bool = False, nested: bool = False):
        if name not in self._data:
            raise KeyError(name)
        data = self._data[name]
        if raw:
            return data
        return self._format_data(data, name, nested=nested)

    def get_prediction(self, x, raw: bool = False, nested: bool = False):
        raise NotImplementedError

    def get_model_error(self, method, raw=False, nested=False, **kwargs):
        raise NotImplementedError

    def get_model_covariance(self, prefactor: float = 1, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def _copy(self, deep: bool = False, **kwargs) -> "DummyObservable":
        cp = copy.deepcopy if deep else copy.copy
        return DummyObservable(data=cp(self._data), model=self.model)

    @classmethod
    def load(cls, filename, **kwargs):  # noqa: ANN206
        raise NotImplementedError

    @classmethod
    def can_load(cls, filename) -> bool:  # noqa: ARG003
        return False

    @property
    def x_names(self) -> list[str]:
        return ["p0", "p1"]


@pytest.fixture
def obs() -> DummyObservable:
    """DummyObservable with 5 samples, 2 x-parameters, 3 y-parameters, and a covariance_y."""
    rng = np.random.default_rng(0)
    return DummyObservable(data={
        "x": rng.normal(size=(5, 2, 1)),
        "y": rng.normal(size=(5, 2, 3)),
        "covariance_y": rng.normal(size=(10, 2, 3)),
    })


class TestFormatFilterValue:
    """Tests for _format_filter_value."""

    def test_scalar(self):
        assert _format_filter_value(3) == "3"

    @pytest.mark.parametrize(("value", "expected"), [
        (slice(0, 10), "0-10"),
        (slice(0, 10, 2), "0-10-2"),
    ])
    def test_slice(self, value, expected):
        assert _format_filter_value(value) == expected

    @pytest.mark.parametrize(("value", "expected"), [
        ([0, 1, 2], "0-2"),
        ([0, 2, 4], "0-4-2"),
        ([0, 3, 6], "0-6-3"),
    ])
    def test_list_arithmetic_sequence(self, value, expected):
        """Checks a constant-step numeric sequence collapses to start-stop[-step], omitting step=1."""
        assert _format_filter_value(value) == expected

    @pytest.mark.parametrize("value", [
        [0, 5],           # only 2 elements, not eligible for the arithmetic-sequence branch
        [0, 1, 4],         # irregular steps
    ])
    def test_list_non_arithmetic_or_short(self, value):
        assert _format_filter_value(value) == ",".join(str(v) for v in value)


class TestMakeHandle:
    """Tests for make_handle and BaseObservable.get_handle."""

    def test_sorted_order_independent(self):
        h1 = make_handle({"i": 0, "j": 1})
        h2 = make_handle({"j": 1, "i": 0})
        assert h1 == h2

    def test_empty_filters_returns_empty_string(self):
        assert make_handle({}) == ""

    def test_hlength_triggers_hash(self):
        filters = {"i": [0, 1, 4], "j": "a_long_value"}
        full = make_handle(filters)
        short = make_handle(filters, hlength=5)
        assert short != full
        assert len(short) <= 5
        assert make_handle(filters, hlength=len(full) + 10) == full

    def test_get_handle(self, obs):
        obs.set_filters(i=[0, 1])
        assert obs.get_handle() == make_handle(obs.filters)
        assert obs.get_handle(prefix="p") == f"p_{make_handle(obs.filters)}"
        assert obs.get_handle(hlength=2) == make_handle(obs.filters, hlength=2)

class TestFormatterFiltersAndSelection:
    """Tests for BaseObservable._format_filters and _format_selection."""

    def test_set_filters_and_get(self, obs):
        obs.set_filters(i=[0, 1])
        assert obs.filters == {"i": [0, 1]}

    def test_clear_filters_resets_everything(self, obs):
        """Checks clear_filters resets state without mutating a dict the caller still holds."""
        original = {"i": [0, 1]}
        obs.filters = original
        obs.set_selection("y", indices=[0, 1])
        obs.clear_filters()
        assert obs.filters == {}
        assert original == {"i": [0, 1]}
        assert obs._select == {}

    def test_get_handle_prefix_and_hlength_forwarded(self, obs):
        """Checks get_handle forwards prefix/hlength to make_handle without duplicating its logic."""
        obs.set_filters(i=0)
        assert obs.get_handle() == make_handle(obs.filters)
        assert obs.get_handle(prefix="p") == f"p_{make_handle(obs.filters)}"
        assert obs.get_handle(hlength=2) == make_handle(obs.filters, hlength=2)


class TestApplySelection:
    """Tests for BaseObservable._apply_selection."""

    def test_no_selection_set_returns_unchanged(self, obs):
        data = np.arange(6).reshape(2, 3)
        assert obs._apply_selection("y", data) is data

    def test_empty_selection_raises(self, obs):
        with pytest.raises(ValueError, match="cannot be empty"):
            obs.set_selection("y", indices=[])

    def test_selection_applied_on_matching_name(self, obs):
        obs.set_selection("y", indices=[0, 2])
        data = np.arange(6).reshape(2, 3)
        result = obs._apply_selection("y", data)
        np.testing.assert_array_equal(result, data[:, [0, 2]])

    def test_selection_skipped_for_unregistered_name(self, obs):
        obs.set_selection("y", indices=[0, 2])
        data = np.arange(6).reshape(2, 3)
        result = obs._apply_selection("covariance_y", data)
        assert result is data

    def test_selection_skipped_for_ndim_3_or_more(self, obs):
        obs.set_selection("y", indices=[0])
        data = np.zeros((2, 3, 4))
        assert obs._apply_selection("y", data) is data

    @pytest.mark.parametrize(("indices", "raises"), [
        ([0, 2], False),   # max index 2 == last valid index for size 3
        ([0, 3], True),    # max index 3 == size -> out of bounds
    ])
    def test_selection_bounds(self, obs, indices, raises):
        obs.set_selection("y", indices=indices)
        data = np.arange(6).reshape(2, 3)
        if raises:
            with pytest.raises(ValueError, match="exceed"):
                obs._apply_selection("y", data)
        else:
            obs._apply_selection("y", data)  # no raise

    def test_multi_name_selection_applies_to_each(self, obs):
        """One call with multiple names registers the same indices under each name."""
        obs.set_selection("y", "covariance_y", indices=[0, 1])
        assert obs._select == {"y": [0, 1], "covariance_y": [0, 1]}
        data = np.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(obs._apply_selection("y", data), data[:, [0, 1]])
        np.testing.assert_array_equal(obs._apply_selection("covariance_y", data), data[:, [0, 1]])

    def test_adding_new_name_to_existing_selection(self, obs):
        """A second call with a different name adds to, rather than replaces, the selection."""
        obs.set_selection("y", indices=[0, 1])
        obs.set_selection("covariance_y", indices=[0])
        assert obs._select == {"y": [0, 1], "covariance_y": [0]}

    def test_overwriting_existing_selection(self, obs):
        """A second call re-using an already-registered name overwrites just that entry."""
        obs.set_selection("y", indices=[0, 1])
        obs.set_selection("y", indices=[2])
        assert obs._select == {"y": [2]}


class TestFormatData:
    """Tests for BaseObservable._format_data."""

    def test_nested_false_applies_selection(self, obs):
        obs.set_selection("y", indices=[0, 1])
        result = obs._format_data(obs._data["y"], "y", nested=False)
        assert result.shape == (5, 2)

    def test_nested_true_skips_selection(self, obs):
        """Same input as the nested=False case; only the nested flag differs."""
        obs.set_selection("y", indices=[0, 1])
        result = obs._format_data(obs._data["y"], "y", nested=True)
        assert result.shape == (5, 2, 3)


class TestBaseObservableCopy:
    """Tests for BaseObservable._copy."""

    @pytest.mark.parametrize(("op", "expected_deep"), [
        (copy.copy, False),
        (copy.deepcopy, True),
    ])
    def test_copy_and_deepcopy_forward_deep_flag(self, obs, op, expected_deep, monkeypatch):
        """Checks __copy__/__deepcopy__ call _copy with the correct deep flag."""
        seen = {}
        monkeypatch.setattr(
            obs,
            "_copy",
            lambda deep=False, **kw: seen.setdefault("deep", deep)  # noqa: ARG005
        )
        op(obs)
        assert seen["deep"] is expected_deep


class TestBaseObservableRepr:
    """Tests for BaseObservable.__repr__."""

    def test_repr_includes_shapes_selection_and_model(self, obs):
        obs.set_filters(i=0)
        obs.set_selection("y", indices=[0, 1])
        r = repr(obs)
        assert "x=" in r
        assert "y=" in r
        assert "covariance_y=" in r
        assert "select=['y']" in r
        assert "has_model=False" in r

    def test_repr_omits_missing_data_gracefully(self):
        obs_no_x = DummyObservable(data={"y": np.zeros((3, 2, 2))})
        r = repr(obs_no_x)
        assert "x=" not in r # No KeyError raised
        assert "y=" in r


class TestGetCovarianceMatrix:
    """Tests for BaseObservable.get_covariance_matrix."""

    def test_covariance_matches_manual_computation(self, obs):
        cov = obs.get_covariance_matrix(volume_factor=2, prefactor=3)
        cov_y = obs.get_data("covariance_y")
        expected = 3 / 2 * np.cov(cov_y, rowvar=False)
        np.testing.assert_allclose(cov, expected)

    def test_selection_applies_only_when_covariance_y_explicitly_named(self, obs):
        """Checks a selection registered under 'y' is NOT applied to covariance_y; only an explicit 'covariance_y' registration is."""
        obs.set_selection("y", indices=[0, 1])
        full_cov = obs.get_covariance_matrix()
        obs.set_selection("covariance_y", indices=[0, 1])
        selected_cov = obs.get_covariance_matrix()
        assert full_cov.shape == (6, 6)
        assert selected_cov.shape == (2, 2)


class TestAbstractContract:
    """Tests that BaseObservable enforces its abstract contract."""

    def test_missing_abstract_method_prevents_instantiation(self):
        """Locks in correct @classmethod/@property + @abstractmethod ordering."""
        # ruff: disable[ANN202, ANN205, ANN206, ARG003, ARG004]
        class Incomplete(BaseObservable[np.ndarray]):
            def _apply_filters(self, data): return data
            @staticmethod
            def _to_numpy(data, nested=False): return data
            def get_data(self, name, raw=False, nested=False): ...
            def get_prediction(self, x, raw=False, nested=False): ...
            def get_model_error(self, method, raw=False, nested=False, **kwargs): ...
            def get_model_covariance(self, prefactor=1, **kwargs): ...
            def _copy(self, deep=False, **kwargs): ...
            @classmethod
            def load(cls, filename, **kwargs): ...
            @classmethod
            def can_load(cls, filename): return False
            # x_names intentionally omitted
        # ruff: enable[ANN202, ANN205, ANN206, ARG003, ARG004]
        with pytest.raises(TypeError):
            Incomplete() # Because x_names is not implemented, this should raise a TypeError
