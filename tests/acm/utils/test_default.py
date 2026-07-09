"""
Small tests on the default values of acm.utils.defaults.

We mainly test that those values match the expected types.
"""
import numpy as np
import pytest

from acm.utils.default import _make_array, cosmo_list, is_nersc

# ruff: noqa: ANN201, D101, D102, D103, INP001, S101


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
