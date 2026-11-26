"""
Small tests on the default values of acm.utils.defaults.

We mainly test that those values match the expected types.
"""
from hashlib import sha256

import numpy as np
import pytest

from acm.utils.default import (
    _make_array,
    cosmo_list,
    is_nersc,
    recursive_update,
    short_hash,
)

# ruff: noqa: ANN001, ANN201, D101, D102, D103, S101


def test_cosmo_list():
    assert isinstance(cosmo_list, list)
    assert all(type(l) is int for l in cosmo_list)  # noqa: E741

def test_is_nersc():
    assert type(is_nersc) is bool

class TestMakeArray:
    def test_scalar_to_1d(self):
        result = _make_array(3.0, 5)
        np.testing.assert_array_equal(result, np.full(5, 3.0))

    def test_scalar_to_2d(self):
        result = _make_array(2.0, (3, 4))
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, 2.0)

    def test_dtype_respected(self):
        result = _make_array(1.0, 4, dtype=np.float32)
        assert result.dtype == np.float32

    def test_nan_input_raises(self):
        """NaN cannot be broadcast into a non-NaN array; should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            _make_array(np.nan, 5)

    def test_incompatible_shape_raises(self):
        """A value whose shape cannot be broadcast to the target shape should raise."""
        with pytest.raises(ValueError, match="could not broadcast"):
            _make_array([1.0, 2.0], 5)


class TestShortHash:
    """Test the short_hash function."""

    def test_short_hash_output(self):
        """Test that the short hash function returns the correct hash."""
        val = "test_string"
        result = short_hash(val, length=None)
        expected = sha256(val.encode()).hexdigest()
        assert result == expected

    @pytest.mark.parametrize("length", [4, 8, 16])
    def test_short_hash_length(self, length):
        """Test that the short hash is correctly truncated."""
        result = short_hash("test_string", length=length)
        expected = sha256(b"test_string").hexdigest()[:length]
        assert result == expected
        assert len(result) == length

    def test_short_hash_consistency(self):
        """Test that the same input produces the same hash."""
        result1 = short_hash("test_string")
        result2 = short_hash("test_string")
        assert result1 == result2

    def test_short_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        result1 = short_hash("test_string_1")
        result2 = short_hash("test_string_2")
        assert result1 != result2


class TestRecursiveUpdate:
    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        updates = {"a": {"y": 4, "z": 5}, "c": {"d": 6}}
        result = recursive_update(base, updates)
        assert result is base
        assert result == {"a": {"x": 1, "y": 4, "z": 5}, "b": 3, "c": {"d": 6}}
        assert updates["c"] is not result["c"]

    def test_preserves_nested_references_when_updating_base(self):
        base = {"a": {"x": 1}, "b": 2}
        nested = base["a"]
        updates = {"a": {"y": 3}, "c": {"d": 4}}

        result = recursive_update(base, updates)

        assert result is base
        assert base["a"] is not nested
        assert nested == {"x": 1}
        assert base == {"a": {"x": 1, "y": 3}, "b": 2, "c": {"d": 4}}
        assert updates == {"a": {"y": 3}, "c": {"d": 4}}
        assert updates["c"] is not base["c"]

    def test_isolates_deeply_nested_mutable_values(self):
        nested = {"child": {"values": [1]}, "retained": [2]}
        untouched = {"values": [3]}
        base = {"a": nested, "untouched": untouched}
        updates = {"a": {"child": {"added": [4]}}, "new": [5]}

        result = recursive_update(base, updates)

        assert result is base
        assert result == {
            "a": {"child": {"values": [1], "added": [4]}, "retained": [2]},
            "untouched": {"values": [3]},
            "new": [5],
        }
        result["a"]["child"]["values"].append(6)
        result["a"]["retained"].append(7)
        result["a"]["child"]["added"].append(8)
        result["new"].append(9)
        assert nested == {"child": {"values": [1]}, "retained": [2]}
        assert updates == {"a": {"child": {"added": [4]}}, "new": [5]}
        assert result["untouched"] is untouched
