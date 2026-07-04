import logging
from unittest.mock import MagicMock, patch

import lsstypes
import matplotlib  # noqa: ICN001
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from acm.estimators.galaxy_clustering.backends.jaxpower import JaxpowerBackend
from acm.estimators.galaxy_clustering.density_split import DensitySplit

matplotlib.use("Agg")

MODULE = "acm.estimators.galaxy_clustering.density_split"

# ruff: noqa: ANN001, ANN202, ANN201, ARG005, D101, D102, D103, INP001, PT019, S101

NQUANTILES = 5

@pytest.fixture
def corr_estimator(make_estimator):
    """Fixture for a DensitySplit estimator with a dummy backend."""
    est = make_estimator(DensitySplit)
    est.backend.set_density_contrast()
    est.set_quantiles(nquantiles=NQUANTILES)
    return est

@pytest.fixture
def power_estimator(data_positions, randoms_positions):
    """Fixture for a DensitySplit estimator with a JaxpowerBackend."""
    backend = JaxpowerBackend(data_positions, randoms_positions)
    # Patch a value to the mocked jaxpower output of JaxpowerBackend
    mock_density_contrast = MagicMock()
    mock_density_contrast.read = lambda positions, **kwargs: np.random.default_rng(42).uniform(0, 1, size=len(positions))
    backend._density_contrast = mock_density_contrast

    return DensitySplit(
        backend=backend,
        data_positions=data_positions,
        randoms_positions=randoms_positions,
    )

def _make_plot_obj(data_type):
    obj = MagicMock()
    obj.attrs = {"data_type": data_type}
    coords = np.linspace(0.01, 0.5, 30)
    obj.flatten.return_value = (MagicMock(coords=MagicMock(return_value=coords)),)
    leaf = obj.get.return_value # quantiles
    leaf.get.return_value.value.return_value = np.ones(30)
    leaf.project.return_value.value.return_value = np.ones(30)
    return obj

@pytest.fixture
def mock_spectrum():
    return _make_plot_obj("power")

@pytest.fixture
def mock_correlation():
    return _make_plot_obj("correlation")


class TestInit:

    def test_log_no_density_contrast(self, caplog, dummy_backend, data_positions, randoms_positions):
        """Test that a log message is emitted when the density contrast is not set."""
        with caplog.at_level(logging.INFO):
            DensitySplit(backend=dummy_backend, data_positions=data_positions, randoms_positions=randoms_positions)
        assert "Density contrast not set" in caplog.text

    def test_set_quantiles_called_when_density_contrast_set(self, dummy_backend, data_positions, randoms_positions):
        """Test that set_quantiles is called when the density contrast is set."""
        dummy_backend.set_density_contrast() # Set quantiles at initialization
        with patch.object(DensitySplit, "set_quantiles") as mock_set_quantiles:
            DensitySplit(backend=dummy_backend, data_positions=data_positions, randoms_positions=randoms_positions)
            mock_set_quantiles.assert_called_once()

class TestSetQuantiles:

    def test_raises_when_no_query_positions_and_no_randoms(self, dummy_backend_no_randoms, data_positions):
        dummy_backend_no_randoms.set_density_contrast() # Set quantiles at initialization
        with pytest.raises(ValueError, match="non-uniform geometry"):
            DensitySplit(backend=dummy_backend_no_randoms, data_positions=data_positions, randoms_positions=None)

    @patch(f"{MODULE}.qcut", wraps=pd.qcut)
    def test_uses_provided_query_positions(self, mock_qcut, dummy_backend, data_positions, randoms_positions):
        """Quantile setting should use provided query positions instead of generating new ones."""
        dummy_backend.set_density_contrast() # Set quantiles at initialization
        rng = np.random.default_rng(42)
        query = rng.uniform(0, 100, size=(10, 3))
        DensitySplit(backend=dummy_backend, data_positions=data_positions, randoms_positions=randoms_positions, query_positions=query)
        mock_qcut.assert_called_once()
        assert np.array_equal(mock_qcut.call_args[0][0], dummy_backend.read_density_contrast(query))

    def test_fall_back_to_backend_query_positions(self, dummy_backend, data_positions, randoms_positions):
        """If no query positions are provided, the backend's get_query_positions should be used."""
        dummy_backend.set_density_contrast() # Set quantiles at initialization
        with patch.object(dummy_backend, "get_query_positions", wraps=dummy_backend.get_query_positions) as mock_get_query:
            DensitySplit(backend=dummy_backend, data_positions=data_positions, randoms_positions=randoms_positions)
            mock_get_query.assert_called_once()

def test_nquantiles_property(corr_estimator):
    """Test that the nquantiles property returns the correct number of quantiles."""
    assert corr_estimator.nquantiles == NQUANTILES

def test_nquantiles_raise_when_not_set(corr_estimator):
    """Test that the nquantiles property raises an error when quantiles are not set."""
    delattr(corr_estimator, "_quantiles")
    with pytest.raises(AttributeError, match=r"Quantiles have not been set yet."):
        _ = corr_estimator.nquantiles

class TestCorrelation:

    @patch(f"{MODULE}.from_pycorr", return_value=MagicMock())
    @patch(f"{MODULE}.TwoPointCorrelationFunction", return_value=MagicMock())
    def test_cross_passes_data_positions2(self, mock_tpcf, _, corr_estimator):
        corr_estimator._correlation(cross=True)
        _, kwargs = mock_tpcf.call_args
        assert np.array_equal(kwargs["data_positions2"], corr_estimator.data_positions)

    @patch(f"{MODULE}.from_pycorr", return_value=MagicMock())
    @patch(f"{MODULE}.TwoPointCorrelationFunction", return_value=MagicMock())
    def test_auto_sets_data_positions2_to_none(self, mock_tpcf, _, corr_estimator):
        corr_estimator._correlation(cross=False)
        _, kwargs = mock_tpcf.call_args
        assert kwargs["data_positions2"] is None

    @patch(f"{MODULE}.from_pycorr", return_value=MagicMock())
    @patch(f"{MODULE}.TwoPointCorrelationFunction")
    def test_r1r2_reused_across_quantiles(self, mock_tpcf, _, corr_estimator):
        """Second and later calls should receive the R1R2 from the first call."""
        r1r2_sentinel = MagicMock()
        mock_tpcf.return_value.R1R2 = r1r2_sentinel
        corr_estimator._correlation(cross=True)
        calls = mock_tpcf.call_args_list
        assert calls[0][1]["R1R2"] is None
        for c in calls[1:]:
            assert c[1]["R1R2"] is r1r2_sentinel

    @patch(f"{MODULE}.from_pycorr", return_value=MagicMock())
    @patch(f"{MODULE}.TwoPointCorrelationFunction")
    def test_r1r2_not_reused_for_davispeebles(self, mock_tpcf, _, corr_estimator):
        mock_tpcf.return_value.R1R2 = MagicMock()
        corr_estimator._correlation(cross=True, estimator="davispeebles")
        for c in mock_tpcf.call_args_list:
            assert c[1]["R1R2"] is None

    @patch(f"{MODULE}.from_pycorr", return_value=MagicMock())
    @patch(f"{MODULE}.TwoPointCorrelationFunction", return_value=MagicMock())
    def test_returns_one_result_per_quantile(self, _, __, corr_estimator):
        result = corr_estimator._correlation(cross=True)
        assert len(result) == corr_estimator.nquantiles

class TestPower:

    def test_raises_for_non_jaxpower_backend(self, corr_estimator):
        """TypeError must be raised when backend is not a JaxpowerBackend."""
        with pytest.raises(TypeError, match="JaxpowerBackend"):
            corr_estimator._power(cross=True)

    @patch(f"{MODULE}.compute_fkp2_shotnoise")
    def test_auto_computes_shotnoise(self, mock_shotnoise, power_estimator):
        power_estimator.jit_cm2s = MagicMock()
        power_estimator._power(cross=False)
        assert mock_shotnoise.call_count == power_estimator.nquantiles

    @patch(f"{MODULE}.compute_fkp2_shotnoise")
    def test_cross_skips_shotnoise(self, mock_shotnoise, power_estimator):
        power_estimator.jit_cm2s = MagicMock()
        power_estimator._power(cross=True)
        mock_shotnoise.assert_not_called()

    def test_returns_one_spectrum_per_quantile(self, power_estimator):
        power_estimator.jit_cm2s = MagicMock()
        result = power_estimator._power(cross=True)
        assert len(result) == power_estimator.nquantiles

class TestCompute:

    def test_dispatches_to_correlation(self, corr_estimator):
        with patch.object(corr_estimator, "_correlation", return_value=[MagicMock()] * NQUANTILES) as mock_correlation:
            corr_estimator.compute(data_type="correlation", cross=True)
            mock_correlation.assert_called_once()

    def test_dispatches_to_power(self, power_estimator):
        with patch.object(power_estimator, "_power", return_value=[MagicMock()] * NQUANTILES) as mock_power:
            power_estimator.compute(data_type="power", cross=True)
            mock_power.assert_called_once()

    def test_raises_for_unknown_data_type(self, corr_estimator):
        with pytest.raises(ValueError, match="Unknown data type"):
            corr_estimator.compute(data_type="unknown")

    def test_returns_observable_tree(self, power_estimator):
        with patch.object(power_estimator, "_power", return_value=[MagicMock()] * NQUANTILES):
            result = power_estimator.compute(data_type="power")
        assert isinstance(result, lsstypes.ObservableTree)
        assert len(result.flatten(level=1)) == NQUANTILES

    @pytest.mark.parametrize("data_type", ["correlation", "power"])
    @pytest.mark.parametrize("cross", [True, False])
    def test_observable_tree_attributes(self, data_type, cross, power_estimator):
        with patch.object(power_estimator, f"_{data_type}", return_value=[MagicMock()] * NQUANTILES):
            result = power_estimator.compute(data_type=data_type, cross=cross)
        assert result.attrs["name"] == "DensitySplit"
        assert result.attrs["data_type"] == data_type
        assert result.attrs["cross"] is cross


class TestLoad:

    def test_returns_observable_tree(self, tmp_path):
        mock_obj = MagicMock()
        with patch(f"{MODULE}.lsstypes.read", return_value=mock_obj):
            result = DensitySplit.load(tmp_path / "result.h5")
        assert result is mock_obj

    def test_project_each_leaf(self, tmp_path):
        n = 3

        mock_leaf = MagicMock()
        mock_leaf.project = MagicMock()
        mock_obj = MagicMock()
        mock_obj.quantiles = list(range(n))
        mock_obj.flatten.return_value = [mock_leaf for _ in range(n)]

        with patch(f"{MODULE}.lsstypes.read", return_value=mock_obj):
            DensitySplit.load(tmp_path / "result.h5", project=True, somearg=3)
        assert mock_leaf.project.call_count == n
        assert mock_leaf.project.call_args_list[0][1] == {"somearg": 3}

    def test_no_project_skips(self, tmp_path):
        mock_leaf = MagicMock()
        mock_leaf.project = MagicMock()
        mock_obj = MagicMock()
        mock_obj.quantiles = list(range(3))
        mock_obj.flatten.return_value = [mock_leaf for _ in range(3)]

        with patch(f"{MODULE}.lsstypes.read", return_value=mock_obj):
            DensitySplit.load(tmp_path / "result.h5", project=False)
        assert mock_leaf.project.call_count == 0

class TestPlot:

    def test_raises_when_data_type_missing(self):
        obj = MagicMock()
        obj.attrs = {}
        with pytest.raises(ValueError, match="data_type"):
            DensitySplit.plot(obj)

    def test_raises_for_unknown_data_type(self):
        obj = MagicMock()
        obj.attrs = {"data_type": "unknown"}
        with pytest.raises(ValueError, match="Unknown data type"):
            DensitySplit.plot(obj)

    def test_create_fig_and_ax_when_none_provided(self, mock_spectrum):
        fig, ax = DensitySplit.plot(mock_spectrum)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_use_provided_fig_and_ax(self, mock_spectrum):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = DensitySplit.plot(mock_spectrum, fig=fig_in, ax=ax_in)
        assert fig_out is fig_in
        assert ax_out is ax_in
        plt.close("all")

    def test_power_uses_get(self, mock_spectrum):
        DensitySplit.plot(mock_spectrum, quantiles=[0], ell=0)
        mock_spectrum.get.return_value.get.assert_called_once()
        plt.close("all")

    def test_correlation_uses_project(self, mock_correlation):
        DensitySplit.plot(mock_correlation, quantiles=[0], ell=0)
        mock_correlation.get.return_value.project.assert_called_once()
        plt.close("all")

    def test_one_line_per_quantile(self, mock_spectrum):
        q = list(range(3))
        _, ax = DensitySplit.plot(mock_spectrum, quantiles=q, ell=0)
        assert len(ax.lines) == 3
        plt.close("all")

class TestPlotQuantiles:

    @pytest.fixture
    def quantile_data(self):
        rng = np.random.default_rng(0)
        nquantiles = 5
        delta_query = rng.uniform(-1, 5, size=200)
        quantiles_idx = np.repeat(np.arange(nquantiles), 40)
        return nquantiles, delta_query, quantiles_idx

    def test_creates_fig_and_ax_when_none_provided(self, quantile_data):
        fig, ax = DensitySplit.plot_quantiles(*quantile_data)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_uses_provided_fig_and_ax(self, quantile_data):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = DensitySplit.plot_quantiles(*quantile_data, fig=fig_in, ax=ax_in)
        assert fig_out is fig_in and ax_out is ax_in
        plt.close(fig_in)

    def test_one_color_per_quantile(self, quantile_data):
        fig, ax = DensitySplit.plot_quantiles(*quantile_data)
        colors = [p.get_facecolor() for p in ax.patches]
        unique_colors = {tuple(c) for c in colors}
        assert len(unique_colors) == quantile_data[0]  # nquantiles
        plt.close(fig)

    def test_custom_colormap_is_applied(self, quantile_data):
        """Patches with 'viridis' must differ from the default 'coolwarm' coloring."""
        fig1, ax1 = DensitySplit.plot_quantiles(*quantile_data, bins=30, colormap="coolwarm")
        fig2, ax2 = DensitySplit.plot_quantiles(*quantile_data, bins=30, colormap="viridis")
        colors1 = [p.get_facecolor() for p in ax1.patches]
        colors2 = [p.get_facecolor() for p in ax2.patches]
        assert colors1 != colors2
        plt.close("all")

    def test_legend_has_one_handle_per_quantile(self, quantile_data):
        nquantiles, *rest = quantile_data
        fig, ax = DensitySplit.plot_quantiles(nquantiles, *rest)
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.legend_handles) == nquantiles
        plt.close(fig)

    def test_single_quantile(self):
        """Edge case: nquantiles=1 must produce a valid plot without errors."""
        delta = np.linspace(-1, 1, 50)
        idx = np.zeros(50, dtype=int)
        fig, ax = DensitySplit.plot_quantiles(1, delta, idx)
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.legend_handles) == 1
        plt.close(fig)
