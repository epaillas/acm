"""Tests for acm.observables.combined."""
import numpy as np
import pytest

from acm.observables.combined import CombinedObservable, ObservableList

# ruff: noqa: ANN001, ANN201, ARG002, D102, S101

class DummyObservable:
    """Minimal stand-in for a BaseObservable, decoupled from any real backend."""

    def __init__(self, x, x_names, y, covariance_y, model_error=None, model_cov=None) -> None:
        self._x_names = x_names
        self._data = {"x": x, "y": y, "covariance_y": covariance_y}
        self._model_error = model_error if model_error is not None else np.zeros(y.shape[1])
        self._model_cov = model_cov if model_cov is not None else np.eye(y.shape[1])

    @property
    def x_names(self):
        return self._x_names

    def get_data(self, name, raw=False, nested=False):
        return self._data[name]

    def get_prediction(self, x):
        return x @ np.ones((x.shape[1], self._data["y"].shape[1]))

    def get_model_error(self, method, raw=False, nested=False, **kwargs):
        return self._model_error

    def get_model_covariance(self, prefactor=1, **kwargs):
        return prefactor * self._model_cov

    def get_handle(self, name=None, hlength=None):
        h = "h" if hlength is None else f"h{hlength}"
        return f"{name}_{h}" if name else h


class SubDummyA(DummyObservable):
    """A distinct subtype of DummyObservable."""


class SubDummyB(DummyObservable):
    """Another distinct subtype of DummyObservable."""


def make_obs(seed: int = 0, n: int = 4, n_features: int = 3, n_cov: int = 6) -> DummyObservable:
    """Create a DummyObservable with random data for testing."""
    rng = np.random.default_rng(seed)
    return DummyObservable(
        x=rng.normal(size=(n, 2)),
        x_names=["p0", "p1"],
        y=rng.normal(size=(n, n_features)),
        covariance_y=rng.normal(size=(n_cov, n_features)),
    )


class Placeholder:
    """Trivial item for ObservableList tests, where the item type itself is irrelevant."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


class TestObservableList:
    """Tests for the ObservableList class."""

    def test_init_stores_and_orders_by_insertion(self):
        a, b = Placeholder("a"), Placeholder("b")
        lst = ObservableList(a=a, b=b)
        assert lst.order == ["a", "b"]

    def test_init_mixed_types_raises(self):
        """Checks ValueError for both unrelated types and distinct subtypes of a common base."""
        with pytest.raises(ValueError, match="same type"):
            ObservableList(a=Placeholder("a"), b=3)
        with pytest.raises(ValueError, match="same type"):
            ObservableList(a=SubDummyA(**_dummy_kwargs()), b=SubDummyB(**_dummy_kwargs()))

    def test_order_setter_updates_order(self):
        lst = ObservableList(a=Placeholder("a"), b=Placeholder("b"))
        lst.order = ["b", "a"]
        assert lst.order == ["b", "a"]

    @pytest.mark.parametrize(("bad_order", "match"), [
        (["a"], "must match"), # missing "b"
        (["a", "a", "b"], "must not contain duplicates"), # duplicate
        (["a", "b", "c"], "must match"), # unknown key
    ])
    def test_order_setter_invalid(self, bad_order, match):
        lst = ObservableList(a=Placeholder("a"), b=Placeholder("b"))
        with pytest.raises(ValueError, match=match):
            lst.order = bad_order

    def test_getitem_and_contains_by_name_and_index(self):
        a, b = Placeholder("a"), Placeholder("b")
        lst = ObservableList(a=a, b=b)
        assert lst["a"] is a
        assert lst[0] is a
        assert "a" in lst
        assert 1 in lst
        assert 2 not in lst
        assert "c" not in lst

    def test_iter_and_reversed_follow_order(self):
        a, b = Placeholder("a"), Placeholder("b")
        lst = ObservableList(a=a, b=b)
        assert list(lst) == [a, b]
        assert list(reversed(lst)) == [b, a]
        assert len(lst) == 2

    def test_items_yields_name_obs_pairs_in_order(self):
        a, b = Placeholder("a"), Placeholder("b")
        lst = ObservableList(a=a, b=b)
        assert list(lst.items()) == [("a", a), ("b", b)]

    def test_add_combines_preserving_order(self):
        lst1 = ObservableList(a=Placeholder("a"))
        lst2 = ObservableList(b=Placeholder("b"))
        combined = lst1 + lst2
        assert combined.order == ["a", "b"]

    def test_add_overlapping_names_raises(self):
        lst1 = ObservableList(a=Placeholder("a"))
        lst2 = ObservableList(a=Placeholder("a2"))
        with pytest.raises(ValueError, match="overlapping"):
            lst1 + lst2


def _dummy_kwargs() -> dict:
    return dict(x=np.zeros((2, 1)), x_names=["p0"], y=np.zeros((2, 1)), covariance_y=np.zeros((2, 1)))


class TestCombinedObservableXProperties:
    """Tests for the x and x_names properties of CombinedObservable."""

    def test_x_and_x_names_consistent(self):
        x = np.zeros((3, 2))
        a = DummyObservable(x=x, x_names=["p0", "p1"], y=np.zeros((3, 2)), covariance_y=np.zeros((4, 2)))
        b = DummyObservable(x=x, x_names=["p0", "p1"], y=np.zeros((3, 3)), covariance_y=np.zeros((4, 3)))
        combined = CombinedObservable(a=a, b=b)  # ty: ignore[invalid-argument-type]
        np.testing.assert_array_equal(combined.x, x)
        assert combined.x_names == ["p0", "p1"]

    def test_x_names_mismatch_raises(self):
        x = np.zeros((3, 2))
        a = DummyObservable(x=x, x_names=["p0", "p1"], y=np.zeros((3, 2)), covariance_y=np.zeros((4, 2)))
        b = DummyObservable(x=x, x_names=["p0", "other"], y=np.zeros((3, 2)), covariance_y=np.zeros((4, 2)))
        combined = CombinedObservable(a=a, b=b)  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="x_names"):
            _ = combined.x_names

    def test_x_mismatch_raises(self):
        a = DummyObservable(x=np.zeros((3, 2)), x_names=["p0", "p1"], y=np.zeros((3, 2)), covariance_y=np.zeros((4, 2)))
        b = DummyObservable(x=np.ones((3, 2)), x_names=["p0", "p1"], y=np.zeros((3, 2)), covariance_y=np.zeros((4, 2)))
        combined = CombinedObservable(a=a, b=b)  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="x"):
            _ = combined.x


class TestRepr:
    """Tests for the __repr__ method of CombinedObservable."""

    def test_repr_includes_names_and_shapes(self):
        a, b = make_obs(0), make_obs(1)
        b._data["x"] = a._data["x"]  # same x to not raise ValueError in repr call to x
        combined = CombinedObservable(a=a, b=b)  # ty: ignore[invalid-argument-type]
        r = repr(combined)
        assert "a" in r
        assert "b" in r
        assert "x=" in r
        assert "y=" in r
        assert "covariance_y=" in r

    def test_repr_omits_missing_data_gracefully(self):
        a = DummyObservable(x=np.zeros((3, 2)), x_names=["p0", "p1"], y=np.zeros((3, 2)), covariance_y=np.zeros((4, 2)))
        b = DummyObservable(x=np.zeros((3, 2)), x_names=["p0", "p1"], y=np.zeros((3, 2)), covariance_y=np.zeros((4, 2)))
        del b._data["x"] # remove x from b
        del b._data["covariance_y"]  # remove covariance_y from b
        combined = CombinedObservable(a=a, b=b)  # ty: ignore[invalid-argument-type]
        r = repr(combined)
        assert "x=" not in r  # KeyError captured
        assert "covariance_y=" not in r  # KeyError captured

    def test_repr_propagates_x_value_error(self):
        """Only KeyError is swallowed in __repr__; a genuine x mismatch (ValueError) must propagate."""
        a = DummyObservable(x=np.zeros((3, 2)), x_names=["p0", "p1"], y=np.zeros((3, 2)), covariance_y=np.zeros((4, 2)))
        b = DummyObservable(x=np.ones((3, 2)), x_names=["p0", "p1"], y=np.zeros((3, 2)), covariance_y=np.zeros((4, 2)))
        combined = CombinedObservable(a=a, b=b)  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="x"):
            repr(combined)


class TestGetHandle:
    """Tests for the get_handle method of CombinedObservable."""

    def test_get_handle_joins_component_handles_with_plus(self):
        a, b = make_obs(0), make_obs(1)
        combined = CombinedObservable(a=a, b=b)  # ty: ignore[invalid-argument-type]
        assert combined.get_handle() == "a_h+b_h" # Default
        assert combined.get_handle(hlength=7) == "a_h7+b_h7" # hlength


class TestTransferCall:
    """Tests for the _transfer_call method of CombinedObservable."""

    def test_transfer_call_concatenates_last_axis(self):
        a, b = make_obs(0, n_features=2), make_obs(1, n_features=3)
        combined = CombinedObservable(a=a, b=b)  # ty: ignore[invalid-argument-type]
        result = combined._transfer_call("get_data", "y")
        assert result.shape == (4, 5)
        np.testing.assert_array_equal(result[:, :2], a.get_data("y"))
        np.testing.assert_array_equal(result[:, 2:], b.get_data("y"))

    @pytest.mark.parametrize(("method", "call_args"), [
        ("get_data", ("y",)),
        ("get_prediction", (np.zeros((2, 2)),)),
    ])
    def test_get_data_and_get_prediction_forward_to_transfer_call(self, method, call_args, monkeypatch):
        """Checks the thin wrapper forwards to _transfer_call; concatenation itself isn't re-tested."""
        combined = CombinedObservable(a=make_obs(0), b=make_obs(1))  # ty: ignore[invalid-argument-type]
        seen = {}
        monkeypatch.setattr(
            combined, "_transfer_call",
            lambda name, *a, **kw: seen.update(name=name, args=a, kwargs=kw) or np.zeros((1, 1)),
        )
        getattr(combined, method)(*call_args)
        assert seen["name"] == method
        assert seen["args"] == call_args

    def test_get_model_error_forces_raw_and_nested_false(self, monkeypatch):
        """Tests that get_model_error forces raw and nested to False."""
        combined = CombinedObservable(a=make_obs(0), b=make_obs(1))  # ty: ignore[invalid-argument-type]
        seen = {}
        monkeypatch.setattr(
            combined, "_transfer_call",
            lambda name, *a, **kw: seen.update(name=name, args=a, kwargs=kw) or np.zeros((1, 1)),
        )
        combined.get_model_error(method="median", extra=1)
        assert seen["name"] == "get_model_error"
        assert seen["args"] == ("median",)
        assert seen["kwargs"] == {"raw": False, "nested": False, "extra": 1}

    def test_order_reorders_output(self):
        a, b = make_obs(0, n_features=2), make_obs(1, n_features=3)
        combined = CombinedObservable(a=a, b=b)  # ty: ignore[invalid-argument-type]
        combined.order = ["b", "a"]
        result = combined._transfer_call("get_data", "y")
        assert result.shape == (4, 5)  #Shape is unchanged
        np.testing.assert_array_equal(result[:, :3], b.get_data("y")) # First columns are b
        np.testing.assert_array_equal(result[:, 3:], a.get_data("y"))


class TestCovariance:
    """Tests for the get_covariance_matrix and get_model_covariance methods of CombinedObservable."""

    def test_get_covariance_matrix_block_true(self):
        a, b = make_obs(0, n_features=2, n_cov=6), make_obs(1, n_features=3, n_cov=6)
        combined = CombinedObservable(a=a, b=b)  # ty: ignore[invalid-argument-type]
        cov = combined.get_covariance_matrix(volume_factor=2, prefactor=3, block=True)
        expected_a = 3 / 2 * np.cov(a.get_data("covariance_y"), rowvar=False)
        expected_b = 3 / 2 * np.cov(b.get_data("covariance_y"), rowvar=False)
        assert cov.shape == (5, 5)
        np.testing.assert_allclose(cov[:2, :2], expected_a)
        np.testing.assert_allclose(cov[2:, 2:], expected_b)
        np.testing.assert_allclose(cov[:2, 2:], 0)

    def test_get_covariance_matrix_block_false(self):
        a, b = make_obs(0, n_features=2, n_cov=6), make_obs(1, n_features=3, n_cov=6)
        combined = CombinedObservable(a=a, b=b)  # ty: ignore[invalid-argument-type]
        cov = combined.get_covariance_matrix(volume_factor=2, prefactor=3, block=False)
        joint = np.concatenate([a.get_data("covariance_y"), b.get_data("covariance_y")], axis=-1)
        expected = 3 / 2 * np.cov(joint, rowvar=False)
        np.testing.assert_allclose(cov, expected)

    def test_get_model_covariance_block_true(self):
        a, b = make_obs(0, n_features=2), make_obs(1, n_features=3)
        combined = CombinedObservable(a=a, b=b)  # ty: ignore[invalid-argument-type]
        cov = combined.get_model_covariance(block=True, prefactor=2)
        assert cov.shape == (5, 5)
        np.testing.assert_allclose(cov[:2, :2], 2 * a._model_cov)
        np.testing.assert_allclose(cov[2:, 2:], 2 * b._model_cov)

    def test_get_model_covariance_block_false_raises(self):
        combined = CombinedObservable(a=make_obs(0), b=make_obs(1))  # ty: ignore[invalid-argument-type]
        with pytest.raises(NotImplementedError):
            combined.get_model_covariance(block=False)
