import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from acm.utils.h5 import (
    _h5_read_state,
    _h5_write_state,
    _prepare_for_json,
    _restore_from_json,
)

# ruff: noqa: ANN201, ARG005, D101, D102, S101


class TestPrepareForJson:

    def test_ndarray_encoded_with_type(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = _prepare_for_json(arr)
        assert result == {"__ndarray__": [1.0, 2.0], "__dtype__": "float32"}

    def test_tuple_encoded(self):
        result = _prepare_for_json((1, 2))
        assert result == {"__tuple__": [1, 2]}

    def test_nested_tuple(self):
        result = _prepare_for_json(((1, 2), 3))
        assert result == {"__tuple__": [{"__tuple__": [1, 2]}, 3]}

    def test_list_recursed(self):
        result = _prepare_for_json([1, (2, 3)])
        assert result == [1, {"__tuple__": [2, 3]}]

    def test_dict_recursed(self):
        result = _prepare_for_json({"a": (1, 2), "b": 3})
        assert result == {"a": {"__tuple__": [1, 2]}, "b": 3}

    def test_np_integer_becomes_int(self):
        result = _prepare_for_json(np.int32(7))
        assert isinstance(result, int)
        assert result == 7

    def test_np_floating_becomes_float(self):
        result = _prepare_for_json(np.float64(3.14))
        assert isinstance(result, float)

    def test_plain_types_passthrough(self):
        for val in (1, 3.14, "hello", None, True):
            assert _prepare_for_json(val) == val


class TestRestoreFromJson:

    def test_ndarray_restored(self):
        encoded = {"__ndarray__": [1.0, 2.0], "__dtype__": "float32"}
        result = _restore_from_json(encoded)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0], dtype=np.float32))
        assert result.dtype == np.float32

    def test_tuple_restored(self):
        encoded = {"__tuple__": [1, 2, 3]}
        assert _restore_from_json(encoded) == (1, 2, 3)

    def test_nested_tuple_restored(self):
        encoded = {"__tuple__": [{"__tuple__": [1, 2]}, 3]}
        assert _restore_from_json(encoded) == ((1, 2), 3)

    def test_list_recursed(self):
        encoded = [1, {"__tuple__": [2, 3]}]
        assert _restore_from_json(encoded) == [1, (2, 3)]

    def test_dict_recursed(self):
        encoded = {"a": {"__tuple__": [1, 2]}, "b": 3}
        assert _restore_from_json(encoded) == {"a": (1, 2), "b": 3}

    def test_plain_types_passthrough(self):
        for val in (1, 3.14, "hello", None):
            assert _restore_from_json(val) == val

    def test_tuple_nolist_raises(self):
        """If a tuple is encoded as a dict but not a list, it should raise an error."""
        encoded = {"__tuple__": "not a list"}
        with pytest.raises(TypeError):
            _restore_from_json(encoded)

    def test_roundtrip(self):
        """Prepare then restore must recover original Python objects."""
        original = {"arr": np.array([1, 2, 3]), "t": (4, 5), "x": 1.0}
        restored = _restore_from_json(_prepare_for_json(original))
        np.testing.assert_array_equal(restored["arr"], original["arr"])
        assert restored["t"] == (4, 5)
        assert restored["x"] == 1.0


class TestH5WriteState:

    def _make_group(self) -> MagicMock:
        """Return a minimal h5py.Group-like mock."""
        grp = MagicMock()
        grp.attrs = {}
        return grp

    def test_arrays_written_as_datasets(self):
        grp = self._make_group()
        arr = np.array([1.0, 2.0])
        _h5_write_state(grp, {"data": arr, "name": "foo"})
        grp.create_dataset.assert_called_once_with("data", data=arr)

    def test_non_arrays_stored_in_meta(self):
        grp = self._make_group()
        _h5_write_state(grp, {"x": 42, "label": "bar"})
        meta = json.loads(grp.attrs["__meta__"])
        assert meta["x"] == 42
        assert meta["label"] == "bar"

    def test_arrays_excluded_from_meta(self):
        grp = self._make_group()
        _h5_write_state(grp, {"arr": np.array([1]), "scalar": 5})
        meta = json.loads(grp.attrs["__meta__"])
        assert "arr" not in meta

    def test_empty_state_writes_empty_meta(self):
        grp = self._make_group()
        _h5_write_state(grp, {})
        meta = json.loads(grp.attrs["__meta__"])
        assert meta == {}

    def test_meta_encodes(self):
        grp = self._make_group()
        _h5_write_state(grp, {"t": (1, 2), "x": 3.14, "d": {"nested": True}})
        meta = json.loads(grp.attrs["__meta__"])
        assert meta == {"t": {"__tuple__": [1, 2]}, "x": 3.14, "d": {"nested": True}}


class TestH5ReadState:

    def _make_group(self, datasets: dict, meta: dict | None = None) -> MagicMock:
        """Return a mock h5 group with datasets and optional __meta__ attr."""
        grp = MagicMock()
        grp.__iter__ = MagicMock(return_value=iter(datasets))
        grp.__getitem__ = lambda self_, k: MagicMock(**{"__getitem__.return_value": datasets[k]})
        for k, v in datasets.items():
            grp[k][...] = v
        grp.attrs = {}
        if meta is not None:
            grp.attrs["__meta__"] = json.dumps(_prepare_for_json(meta))
        return grp

    def test_datasets_are_read(self):
        arr = np.array([1.0, 2.0])
        grp = self._make_group({"data": arr}) # NOTE: also tests the no meta case
        state = _h5_read_state(grp)
        np.testing.assert_array_equal(state["data"], arr)

    def test_meta_is_merged(self):
        grp = self._make_group({}, meta={"label": "foo", "n": 3})
        state = _h5_read_state(grp)
        assert state["label"] == "foo"
        assert state["n"] == 3

    def test_raises_when_meta_not_a_dict(self):
        """Non-dict __meta__ JSON must raise ValueError."""
        grp = self._make_group({})
        grp.attrs["__meta__"] = json.dumps([1, 2, 3])  # list, not dict
        with pytest.raises(ValueError, match="Expected dict"):
            _h5_read_state(grp)

    def test_meta_decodes(self):
        grp = self._make_group({}, meta={"t": (1, 2, 3), "x": 3.14, "d": {"nested": True}})
        state = _h5_read_state(grp)
        assert state["t"] == (1, 2, 3)
        assert state["x"] == 3.14
        assert state["d"] == {"nested": True}
