import numpy as np
import pytest

from acm.utils.catalogs import check_catalog

# ruff: noqa: ANN001, ANN201, D101, INP001, S101

#%% Helpers
def make_positions(n=10, low=0.0, high=100.0, seed=0):
    """Create a random catalog of positions in a box."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, 3)).astype(np.float64)


class TestCheckCatalog:
    def test_valid_catalog(self):
        """Test that a valid catalog passes the checks."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        boxsize = 100.0
        check_catalog(positions, boxsize)

    def test_array_boxsize(self):
        """Test that an array boxsize is accepted."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        boxsize = np.array([100.0, 100.0, 100.0])
        check_catalog(positions, boxsize)

    def test_list_boxsize(self):
        """Test that a list boxsize is accepted."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        boxsize = [100.0, 100.0, 100.0]
        check_catalog(positions, boxsize)

    def test_center_at_zero(self):
        """Test that positions centered at zero are accepted when center_at_zero is True."""
        positions = make_positions(n=100, low=-50.0, high=50.0)
        boxsize = 100.0
        check_catalog(positions, boxsize, center_at_zero=True)

    def test_float64_precision(self):
        """Test that checks can be performed in float64 precision."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        boxsize = 100.0
        check_catalog(positions, boxsize, check_in_float32=False)

    def test_left_edge_inclusive(self):
        """Test that positions on the left edge are accepted."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        positions[0, 0] = 0.0  # Set one position to the left edge
        boxsize = 100.0
        check_catalog(positions, boxsize)

    def test_right_edge_exclusive(self):
        """Test that positions on the right edge are rejected."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        positions[0, 0] = 100.0  # Set one position to the right edge
        boxsize = 100.0
        with pytest.raises(ValueError, match="right edge"):
            check_catalog(positions, boxsize)

    def test_invalid_boxsize_shape(self):
        """Test that an invalid boxsize shape raises an error."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        boxsize = np.array([100.0, 100.0])  # Invalid shape
        with pytest.raises(ValueError, match="boxsize"):
            check_catalog(positions, boxsize)

    def test_out_of_bounds_left(self):
        """Test that positions outside the left boundary raise an error."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        positions[0, 0] = -1.0  # Set one position outside the left boundary
        boxsize = 100.0
        with pytest.raises(ValueError, match="left edge"):
            check_catalog(positions, boxsize)

    def test_out_of_bounds_right(self):
        """Test that positions outside the right boundary raise an error."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        positions[0, 0] = 101.0  # Set one position outside the right boundary
        boxsize = 100.0
        with pytest.raises(ValueError, match="right edge"):
            check_catalog(positions, boxsize)

    def test_out_of_bounds_both_edges(self):
        """Test that positions outside both boundaries raise an error."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        positions[0, 0] = -1.0  # Set one position outside the left boundary
        positions[1, 0] = 101.0  # Set another position outside the right boundary
        boxsize = 100.0
        with pytest.raises(ValueError) as exc_info:  # noqa: PT011
            check_catalog(positions, boxsize)
        assert "left edge" in str(exc_info.value)
        assert "right edge" in str(exc_info.value)

    def test_out_of_bounds_centered(self):
        """Test that positions outside the boundaries raise an error when center_at_zero is True."""
        positions = make_positions(n=100, low=-50.0, high=50.0)
        positions[0, 0] = -51.0  # Set one position outside the left boundary
        positions[1, 0] = 51.0   # Set another position outside the right boundary
        boxsize = 100.0
        with pytest.raises(ValueError) as exc_info:  # noqa: PT011
            check_catalog(positions, boxsize, center_at_zero=True)
        assert "left edge" in str(exc_info.value)
        assert "right edge" in str(exc_info.value)

    # Edge cases
    def test_single_position(self):
        """Test that a catalog with a single position is checked correctly."""
        positions = np.array([[50.0, 50.0, 50.0]])
        boxsize = 100.0
        check_catalog(positions, boxsize)

    def test_asymmetric_box(self):
        """Test that an asymmetric box size is handled correctly."""
        # Symmetric positions in the smallest box dimension
        positions = make_positions(n=100, low=0.0, high=25.0)
        boxsize = [100.0, 50.0, 25.0]
        check_catalog(positions, boxsize)

    def test_asymmetric_box_centered(self):
        """Test that an asymmetric box size with centered positions is handled correctly."""
        # Symmetric positions in the smallest box dimension
        positions = make_positions(n=100, low=-12.5, high=12.5)
        boxsize = [100.0, 50.0, 25.0]
        check_catalog(positions, boxsize, center_at_zero=True)

    def test_asymmetric_box_out_of_bounds(self):
        """Test that positions outside the boundaries of an asymmetric box raise an error."""
        # Symmetric positions in the smallest box dimension
        positions = make_positions(n=100, low=0.0, high=25.0)
        positions[0, 1] = 51.0  # Set one position outside the second dimension boundary
        boxsize = [100.0, 50.0, 25.0]
        with pytest.raises(ValueError, match="right edge"):
            check_catalog(positions, boxsize)
