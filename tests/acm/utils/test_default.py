"""
Small tests on the default values of acm.utils.defaults.

We mainly test that those values match the expected types.
"""
from hashlib import sha256

import numpy as np
import pytest

from acm.utils.default import _make_array, cosmo_list, is_nersc, short_hash

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
