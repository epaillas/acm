from unittest.mock import MagicMock, patch

import lsstypes
import matplotlib  # noqa: ICN001
import matplotlib.pyplot as plt
import numpy as np
import pytest

from acm.estimators.galaxy_clustering.cic import CountsInCells

matplotlib.use("Agg")

MODULE = "acm.estimators.galaxy_clustering.cic"

# ruff: noqa: ANN001, ANN201, D101, D102, D103, INP001, S101

@pytest.fixture
def estimator(make_estimator):
    return make_estimator(CountsInCells)

@pytest.fixture
def estimator_no_randoms(dummy_backend_no_randoms, data_positions):
    """CountsInCells instance without a randoms catalog."""
    return CountsInCells(
        backend=dummy_backend_no_randoms,
        data_positions=data_positions,
        randoms_positions=None,
    )

@pytest.fixture
def compute_result(estimator):
    """Compute a CountsInCells result for testing."""
    query_positions = np.zeros((10, 3))
    return estimator.compute(query_positions=query_positions)

@pytest.fixture
def file(tmp_path, estimator, compute_result):
    """Create a temporary saved result file for testing the load method."""
    file_path = tmp_path / "test_cic_file.h5"
    estimator.save(compute_result, str(file_path))
    return str(file_path)


class TestCompute:

    def test_raise_when_no_query_positions_and_no_randoms(self, estimator_no_randoms):
        """ValueError must be raised in non-uniform geometry without explicit query positions."""
        with pytest.raises(ValueError, match="non-uniform geometry"):
            estimator_no_randoms.compute()

    def test_use_provided_query_positions(self, estimator):
        # Override the backend's read_density_contrast method to track calls
        estimator.backend.read_density_contrast = MagicMock(return_value=np.zeros(10))
        rng = np.random.default_rng(0)
        query_pos = rng.uniform(0, 100, size=(20, 3))
        estimator.compute(query_positions=query_pos)
        estimator.backend.read_density_contrast.assert_called_once_with(query_pos, resampler="cic")

    def test_fall_back_to_backend_query_positions(self, estimator):
        """When query_positions is None, the backend's get_query_positions method is called."""
        # Override the backend's get_query_positions method to return a fixed set of positions
        estimator.backend.get_query_positions = MagicMock(return_value=np.zeros((20, 3)))
        estimator.compute(query_positions=None, nquery=50)
        estimator.backend.get_query_positions.assert_called_once_with(nquery=50)

    def test_returns_observable_leaf(self, estimator):
        """The compute method should return an ObservableLeaf with the expected attributes."""
        result = estimator.compute(query_positions=np.zeros((10, 3)))
        assert isinstance(result, lsstypes.ObservableLeaf)
        assert "density_contrast" in result.values()
        assert "index" in result.coords()
        assert np.array_equal(result.coords("index"), np.arange(10))

class TestLoad:

    def test_returns_object_directly(self, file, compute_result):
        """The load method returns an ObservableLeaf when to_hist is False."""
        loaded_obj = CountsInCells.load(file, to_hist=False)
        assert isinstance(loaded_obj, lsstypes.ObservableLeaf)
        assert "density_contrast" in loaded_obj.values()
        assert "index" in loaded_obj.coords()
        assert np.array_equal(loaded_obj.density_contrast, compute_result.density_contrast)
        assert np.array_equal(loaded_obj.coords("index"), compute_result.coords("index"))

    def test_converts_to_histogram(self, file, compute_result):
        """The load method converts to a histogram representation when to_hist is True."""
        loaded_hist = CountsInCells.load(file, to_hist=True, bins=5)
        assert isinstance(loaded_hist, lsstypes.ObservableLeaf)
        assert "hist" in loaded_hist.values()
        assert "bins" in loaded_hist.coords()
        # Check that the histogram has the expected number of bins
        hist, _ = np.histogram(compute_result.density_contrast, bins=5)
        assert np.array_equal(loaded_hist.hist, hist)

    def test_to_hist_has_left_edges(self, file, compute_result):
        """Bins stored in the leaf must be the left edges (bin_edges[:-1])."""
        loaded_hist = CountsInCells.load(file, to_hist=True, bins=5)
        _, bins = np.histogram(compute_result.density_contrast, bins=5)
        assert np.array_equal(loaded_hist.coords("bins"), bins[:-1])

    def test_skipped_hist(self, file, compute_result):
        """The load method returns the original object when to_hist is False."""
        with patch(f"{MODULE}.np.histogram", wraps=np.histogram) as mock_hist:
            loaded_obj = CountsInCells.load(file, to_hist=False)
            mock_hist.assert_not_called()
        assert isinstance(loaded_obj, lsstypes.ObservableLeaf)
        assert "density_contrast" in loaded_obj.values()
        assert "index" in loaded_obj.coords()
        assert np.array_equal(loaded_obj.density_contrast, compute_result.density_contrast)
        assert np.array_equal(loaded_obj.coords("index"), compute_result.coords("index"))


class TestPlot:

    def test_creates_fig_and_ax_when_none_provided(self, compute_result):
        fig, ax = CountsInCells.plot(compute_result)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_uses_provided_fig_and_ax(self, compute_result):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = CountsInCells.plot(compute_result, fig=fig_in, ax=ax_in)
        assert fig_out is fig_in
        assert ax_out is ax_in
        plt.close(fig_in)

    def test_plots_bar_when_obj_has_hist(self, file):
        mock_hist_leaf = CountsInCells.load(file, to_hist=True)
        fig, ax = CountsInCells.plot(mock_hist_leaf)
        assert len(ax.patches) > 0  # bar() produces patches
        plt.close(fig)

    def test_plots_hist_when_obj_has_no_hist(self, compute_result):
        fig, ax = CountsInCells.plot(compute_result)
        assert len(ax.patches) > 0  # hist() also produces patches
        plt.close(fig)
