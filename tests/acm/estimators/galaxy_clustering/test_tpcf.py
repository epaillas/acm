from unittest.mock import MagicMock, patch

import lsstypes
import matplotlib  # noqa: ICN001
import matplotlib.pyplot as plt
import numpy as np
import pytest

from acm.estimators.galaxy_clustering.tpcf import TwoPointCorrelationFunctionEstimator

matplotlib.use("Agg") # Use a non-interactive backend for testing

MODULE = "acm.estimators.galaxy_clustering.tpcf"

# ruff: noqa: ANN001, ANN201, D101, D102, D103, INP001, S101


@pytest.fixture
def estimator(make_estimator):
    return make_estimator(TwoPointCorrelationFunctionEstimator)


@pytest.fixture
def mock_tpcf_result():
    return MagicMock()


@pytest.fixture
def mock_count2corr():
    return MagicMock(spec=lsstypes.Count2Correlation)


@pytest.fixture
def mock_poles():
    """Simulate an already-projected Count2CorrelationPoles object."""
    poles = MagicMock(spec=lsstypes.Count2CorrelationPoles)
    coord_mock = MagicMock(return_value=np.linspace(1, 100, 30))
    poles.flatten.return_value = (MagicMock(coords=coord_mock),)
    poles.get.return_value.value.return_value = np.ones(30)
    return poles


class TestCompute:

    @patch(f"{MODULE}.TwoPointCorrelationFunction")
    @patch(f"{MODULE}.from_pycorr")
    def test_call_tpcf_with_correct_args(self, mock_from_pycorr, mock_tpcf, estimator, mock_tpcf_result, mock_count2corr):
        mock_from_pycorr.return_value = mock_count2corr
        mock_tpcf.return_value = mock_tpcf_result
        estimator.compute(mode="s")
        mock_tpcf.assert_called_once_with(
            data_positions1=estimator.data_positions,
            randoms_positions1=estimator.randoms_positions,
            data_weights1=estimator.data_weights,
            randoms_weights1=estimator.randoms_weights,
            boxsize=estimator.backend.boxsize,
            position_type="pos",
            mode="s",
        )

    @patch(f"{MODULE}.TwoPointCorrelationFunction")
    @patch(f"{MODULE}.from_pycorr")
    def test_return_count2correlation(self, mock_from_pycorr, mock_tpcf, estimator, mock_tpcf_result, mock_count2corr):
        mock_from_pycorr.return_value = mock_count2corr
        mock_tpcf.return_value = mock_tpcf_result
        result = estimator.compute()
        assert result is mock_count2corr

    @patch(f"{MODULE}.TwoPointCorrelationFunction")
    @patch(f"{MODULE}.from_pycorr")
    def test_pass_pycorr_result_to_from_pycorr(self, mock_from_pycorr, mock_tpcf, estimator, mock_tpcf_result, mock_count2corr):
        mock_from_pycorr.return_value = mock_count2corr
        mock_tpcf.return_value = mock_tpcf_result
        estimator.compute()
        mock_from_pycorr.assert_called_once_with(mock_tpcf_result)


class TestLoad:

    @patch(f"{MODULE}.lsstypes.read")
    def test_load_calls_lsstypes_read(self, mock_read, tmp_path):
        mock_obj = MagicMock(spec=lsstypes.Count2Correlation)
        mock_read.return_value = mock_obj
        result = TwoPointCorrelationFunctionEstimator.load(tmp_path / "result.h5")
        mock_read.assert_called_once()
        assert result is mock_obj

    @patch(f"{MODULE}.lsstypes.read")
    def test_load_with_project_calls_project(self, mock_read, tmp_path):
        mock_obj = MagicMock(spec=lsstypes.Count2Correlation)
        projected = MagicMock()
        mock_obj.project.return_value = projected
        mock_read.return_value = mock_obj
        result = TwoPointCorrelationFunctionEstimator.load(
            tmp_path / "result.h5", project=True, ells=(0, 2)
        )
        mock_obj.project.assert_called_once_with(ells=(0, 2))
        assert result is projected

    @patch(f"{MODULE}.lsstypes.read")
    def test_load_without_project_skips_projection(self, mock_read, tmp_path):
        mock_obj = MagicMock(spec=lsstypes.Count2Correlation)
        mock_read.return_value = mock_obj
        TwoPointCorrelationFunctionEstimator.load(tmp_path / "result.h5", project=False)
        mock_obj.project.assert_not_called()


class TestPlot:

    def test_create_fig_and_ax_when_none_provided(self, mock_poles):
        fig, ax = TwoPointCorrelationFunctionEstimator.plot(mock_poles)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_use_provided_fig_and_ax(self, mock_poles):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = TwoPointCorrelationFunctionEstimator.plot(mock_poles, fig=fig_in, ax=ax_in)
        assert fig_out is fig_in
        assert ax_out is ax_in
        plt.close(fig_in)

    def test_project_count2correlation_object(self, mock_count2corr, mock_poles):
        """Count2Correlation input should be projected before plotting."""
        mock_count2corr.project.return_value = mock_poles
        TwoPointCorrelationFunctionEstimator.plot(mock_count2corr, ells=(0, 2))
        mock_count2corr.project.assert_called_once_with(ells=(0, 2))
        plt.close("all")

    def test_skip_projection_for_poles_object(self, mock_poles):
        """Count2CorrelationPoles input should not be re-projected."""
        # Give mock_poles a project method to ensure it's not called
        mock_poles.project = MagicMock()
        TwoPointCorrelationFunctionEstimator.plot(mock_poles)
        mock_poles.project.assert_not_called()
        plt.close("all")

    def test_create_fig_with_custom_figsize(self, mock_poles):
        fig, _ = TwoPointCorrelationFunctionEstimator.plot(mock_poles, figsize=(12, 4))
        w, h = fig.get_size_inches()
        assert (w, h) == (12, 4)
        plt.close(fig)

    def test_draw_one_line_per_multipole(self, mock_poles):
        fig, ax = TwoPointCorrelationFunctionEstimator.plot(mock_poles, ells=(0, 2, 4))
        assert len(ax.lines) == 3
        plt.close(fig)
