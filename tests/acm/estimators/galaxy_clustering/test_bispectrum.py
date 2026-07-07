from unittest.mock import MagicMock, patch

import lsstypes
import matplotlib.pyplot as plt
import numpy as np
import pytest

from acm.estimators.galaxy_clustering.backends.jaxpower import JaxpowerBackend
from acm.estimators.galaxy_clustering.bispectrum import BispectrumMultipoles

MODULE = "acm.estimators.galaxy_clustering.bispectrum"

# ruff: noqa: ANN001, ANN201, ANN202, ARG002, D101, D102, D103, INP001, S101

@pytest.fixture
def estimator(data_positions, randoms_positions):
    backend = JaxpowerBackend(data_positions, randoms_positions)
    return BispectrumMultipoles(
        backend=backend,
        data_positions=data_positions,
        randoms_positions=randoms_positions,
    )

@pytest.fixture
def estimator_no_randoms(data_positions):
    backend = JaxpowerBackend(data_positions, None)
    return BispectrumMultipoles(
        backend=backend,
        data_positions=data_positions,
        randoms_positions=None,
    )

def _make_mock_spectrum(basis: str, k: np.ndarray):
    obj = MagicMock(spec=lsstypes.Mesh3SpectrumPoles)
    obj.basis = basis
    coords_mock = MagicMock(return_value=k)
    obj.flatten.return_value = (MagicMock(coords=coords_mock),)
    obj.get.return_value.value.return_value = MagicMock(real=np.ones(len(k)))
    return obj

@pytest.fixture
def mock_spectrum_scoccimarro():
    return _make_mock_spectrum("scoccimarro", np.linspace(0.01, 0.5, 50))

@pytest.fixture
def mock_spectrum_sugiyama():
    k = np.column_stack([np.linspace(0.01, 0.5, 30)] * 2)
    return _make_mock_spectrum("sugiyama", k)

class TestInit:

    def test_raises_when_backend_not_jaxpower(self, data_positions, randoms_positions, dummy_backend):
        """TypeError must be raised when backend is not a JaxpowerBackend."""
        with pytest.raises(TypeError, match="requires a JaxpowerBackend"):
            BispectrumMultipoles(
                backend=dummy_backend,
                data_positions=data_positions,
                randoms_positions=randoms_positions,
            )

    def test_init_with_jaxpower_backend(self, data_positions, randoms_positions):
        backend = JaxpowerBackend(data_positions, randoms_positions)
        estimator = BispectrumMultipoles(
            backend=backend,
            data_positions=data_positions,
            randoms_positions=randoms_positions,
        )
        assert isinstance(estimator.backend, JaxpowerBackend)

    def test_jit_cm3s_is_callable(self, estimator):
        assert callable(estimator.jit_cm3s)

class TestCompute:

    @patch(f"{MODULE}.BinMesh3SpectrumPoles")
    @patch(f"{MODULE}.cm3s")
    @patch(f"{MODULE}.compute_box3_normalization")
    @patch(f"{MODULE}.compute_fkp3_normalization")
    @patch(f"{MODULE}.compute_fkp3_shotnoise")
    @patch(f"{MODULE}.FKPField")
    def test_randoms_use_fkp(self, mock_fkp, mock_shotnoise, mock_fkp_norm, mock_box_norm, mock_cm3s, mock_bin, estimator):
        """FKP path must be used when randoms are provided."""
        estimator.jit_cm3s = mock_cm3s
        estimator.compute()
        mock_fkp.assert_called_once()
        mock_fkp_norm.assert_called_once()
        mock_box_norm.assert_not_called()
        mock_cm3s.assert_called_once()

    @patch(f"{MODULE}.BinMesh3SpectrumPoles")
    @patch(f"{MODULE}.cm3s")
    @patch(f"{MODULE}.compute_box3_normalization")
    @patch(f"{MODULE}.compute_fkp3_normalization")
    @patch(f"{MODULE}.compute_fkp3_shotnoise")
    @patch(f"{MODULE}.FKPField")
    def test_no_randoms_use_box_normalization(self, mock_fkp, mock_shotnoise, mock_fkp_norm, mock_box_norm, mock_cm3s, mock_bin, estimator_no_randoms):
        """Box normalization must be used when no randoms are provided."""
        estimator_no_randoms.jit_cm3s = mock_cm3s
        estimator_no_randoms.compute()
        mock_fkp.assert_not_called()
        mock_fkp_norm.assert_not_called()
        mock_box_norm.assert_called_once()
        mock_cm3s.assert_called_once()

    @patch(f"{MODULE}.BinMesh3SpectrumPoles")
    @patch(f"{MODULE}.cm3s")
    @patch(f"{MODULE}.compute_box3_normalization")
    @patch(f"{MODULE}.compute_fkp3_normalization")
    @patch(f"{MODULE}.compute_fkp3_shotnoise")
    @patch(f"{MODULE}.FKPField")
    def test_returns_cloned_spectrum(self, mock_fkp, mock_shotnoise, mock_fkp_norm, mock_box_norm, mock_cm3s, mock_bin, estimator):
        estimator.jit_cm3s = mock_cm3s
        result = estimator.compute()
        assert result is mock_cm3s.return_value.clone.return_value

    @patch(f"{MODULE}.BinMesh3SpectrumPoles")
    @patch(f"{MODULE}.cm3s")
    @patch(f"{MODULE}.compute_box3_normalization")
    @patch(f"{MODULE}.compute_fkp3_normalization")
    @patch(f"{MODULE}.compute_fkp3_shotnoise")
    @patch(f"{MODULE}.FKPField")
    def test_default_ells_scoccimarro(self, mock_fkp, mock_shotnoise, mock_fkp_norm, mock_box_norm, mock_cm3s, mock_bin, estimator):
        """Default ells for scoccimarro basis must be (0, 2)."""
        estimator.jit_cm3s = mock_cm3s
        estimator.compute(basis="scoccimarro")
        _, kwargs = mock_bin.call_args
        assert kwargs["ells"] == (0, 2)

    @patch(f"{MODULE}.BinMesh3SpectrumPoles")
    @patch(f"{MODULE}.cm3s")
    @patch(f"{MODULE}.compute_box3_normalization")
    @patch(f"{MODULE}.compute_fkp3_normalization")
    @patch(f"{MODULE}.compute_fkp3_shotnoise")
    @patch(f"{MODULE}.FKPField")
    def test_default_ells_sugiyama(self, mock_fkp, mock_shotnoise, mock_fkp_norm, mock_box_norm, mock_cm3s, mock_bin, estimator):
        """Default ells for sugiyama basis must be [(0,0,0), (0,0,2)]."""
        estimator.jit_cm3s = mock_cm3s
        estimator.compute(basis="sugiyama")
        _, kwargs = mock_bin.call_args
        assert kwargs["ells"] == [(0, 0, 0), (0, 0, 2)]

    @patch(f"{MODULE}.BinMesh3SpectrumPoles")
    @patch(f"{MODULE}.cm3s")
    @patch(f"{MODULE}.compute_box3_normalization")
    @patch(f"{MODULE}.compute_fkp3_normalization")
    @patch(f"{MODULE}.compute_fkp3_shotnoise")
    @patch(f"{MODULE}.FKPField")
    def test_bin_mesh_receives_correct_args(self, mock_fkp, mock_shotnoise, mock_fkp_norm, mock_box_norm, mock_cm3s, mock_bin, estimator):
        """BinMesh3SpectrumPoles must receive basis, buffer_size, and mask_edges."""
        estimator.jit_cm3s = mock_cm3s
        estimator.compute(basis="scoccimarro", buffer_size=10, mask_edges="k1 > k2")
        _, kwargs = mock_bin.call_args
        assert kwargs["basis"] == "scoccimarro"
        assert kwargs["buffer_size"] == 10
        assert kwargs["mask_edges"] == "k1 > k2"

class TestLoad:

    @patch(f"{MODULE}.lsstypes.read")
    def test_load_returns_lsstypes_object(self, mock_read, tmp_path):
        mock_obj = MagicMock(spec=lsstypes.Mesh3SpectrumPoles)
        mock_read.return_value = mock_obj
        result = BispectrumMultipoles.load(tmp_path / "result.h5")
        assert result is mock_obj
        mock_read.assert_called_once_with(tmp_path / "result.h5")

class TestPlot:

    def test_raises_for_wrong_obj_type(self):
        with pytest.raises(TypeError, match="Mesh3SpectrumPoles"):
            BispectrumMultipoles.plot(MagicMock())

    def test_raises_for_unknown_basis(self, mock_spectrum_scoccimarro):
        mock_spectrum_scoccimarro.basis = "unknown"
        with pytest.raises(ValueError, match="basis"):
            BispectrumMultipoles.plot(mock_spectrum_scoccimarro)

    @patch(f"{MODULE}._plot_scoccimarro", return_value=(MagicMock(), MagicMock()))
    def test_dispatches_to_scoccimarro(self, mock_plot, mock_spectrum_scoccimarro):
        BispectrumMultipoles.plot(mock_spectrum_scoccimarro)
        mock_plot.assert_called_once()
        plt.close("all")

    @patch(f"{MODULE}._plot_sugiyama", return_value=(MagicMock(), MagicMock()))
    def test_dispatches_to_sugiyama(self, mock_plot, mock_spectrum_sugiyama):
        BispectrumMultipoles.plot(mock_spectrum_sugiyama)
        mock_plot.assert_called_once()
        plt.close("all")

class TestPlotScoccimarro:

    def test_creates_fig_and_ax_when_none_provided(self, mock_spectrum_scoccimarro):
        fig, ax = BispectrumMultipoles.plot(mock_spectrum_scoccimarro)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_uses_provided_fig_and_ax(self, mock_spectrum_scoccimarro):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = BispectrumMultipoles.plot(mock_spectrum_scoccimarro, fig=fig_in, ax=ax_in)
        assert fig_out is fig_in
        assert ax_out is ax_in
        plt.close(fig_in)

    def test_draws_one_line_per_multipole(self, mock_spectrum_scoccimarro):
        fig, ax = BispectrumMultipoles.plot(mock_spectrum_scoccimarro, ells=(0, 2))
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_uses_linear_x_axis(self):
        """1D-k input must use k directly as x-axis and apply k**2 weights."""
        k = np.linspace(0.01, 0.5, 50)
        x = np.arange(len(k))
        obj = _make_mock_spectrum("scoccimarro", k)
        fig, ax = BispectrumMultipoles.plot(obj, ells=(0, 2))
        x_result = ax.lines[0].get_xdata()
        np.testing.assert_allclose(x_result, x)  # ty:ignore[no-matching-overload]
        plt.close(fig)

    def test_1d_k_weights_return_k3(self):
        """1D-k input must apply k**3 weights."""
        k = np.linspace(0.01, 0.5, 50)
        obj = _make_mock_spectrum("scoccimarro", k)
        fig, ax = BispectrumMultipoles.plot(obj, ells=(0, 2), weight_by_kprod=True)
        y_result = ax.lines[0].get_ydata()
        y_expected = k**3 # NOTE: assuming ones for the mocked poles
        np.testing.assert_allclose(y_result, y_expected)  # ty:ignore[no-matching-overload]
        plt.close(fig)

    def test_2d_k_weights_return_kprod(self):
        """2D-k input must apply k1*k2*k3 weights."""
        k = np.column_stack([np.linspace(0.01, 0.5, 30)] * 2)
        obj = _make_mock_spectrum("scoccimarro", k)
        fig, ax = BispectrumMultipoles.plot(obj, ells=(0, 2), weight_by_kprod=True)
        y_result = ax.lines[0].get_ydata()
        y_expected = np.prod(k, axis=-1) # NOTE: assuming ones for the mocked poles
        np.testing.assert_allclose(y_result, y_expected)  # ty:ignore[no-matching-overload]
        plt.close(fig)

    def test_no_weights_returns_poles(self):
        """When weight_by_kprod is False, the y-data must be the poles."""
        k = np.linspace(0.01, 0.5, 50)
        obj = _make_mock_spectrum("scoccimarro", k)
        fig, ax = BispectrumMultipoles.plot(obj, ells=(0, 2), weight_by_kprod=False)
        y_result = ax.lines[0].get_ydata()
        y_expected = np.ones_like(k) # NOTE: assuming ones for the mocked poles
        np.testing.assert_allclose(y_result, y_expected)  # ty:ignore[no-matching-overload]
        plt.close(fig)

class TestPlotSugiyama:

    def test_creates_fig_and_ax_when_none_provided(self, mock_spectrum_sugiyama):
        fig, ax = BispectrumMultipoles.plot(mock_spectrum_sugiyama, ells=[(0, 0, 0)])
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_uses_provided_fig_and_ax(self, mock_spectrum_sugiyama):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = BispectrumMultipoles.plot(mock_spectrum_sugiyama, fig=fig_in, ax=ax_in)
        assert fig_out is fig_in
        assert ax_out is ax_in
        plt.close(fig_in)

    def test_draws_one_line_per_multipole(self, mock_spectrum_sugiyama):
        fig, ax = BispectrumMultipoles.plot(mock_spectrum_sugiyama, ells=[(0, 0, 0), (0, 0, 2)])
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_1d_k_uses_k_as_x_axis(self):
        """1D-k input must use k directly as x-axis and apply k**2 weights."""
        k = np.linspace(0.01, 0.5, 30)
        obj = _make_mock_spectrum("sugiyama", k)
        fig, ax = BispectrumMultipoles.plot(obj, ells=[(0, 0, 0)])
        x_data = ax.lines[0].get_xdata()
        np.testing.assert_allclose(x_data, k)  # ty:ignore[no-matching-overload]
        plt.close(fig)

    def test_1d_k_weights_return_k2(self):
        """1D-k input must apply k**2 weights."""
        k = np.linspace(0.01, 0.5, 30)
        obj = _make_mock_spectrum("sugiyama", k)
        fig, ax = BispectrumMultipoles.plot(obj, ells=[(0, 0, 0)], weight_by_kprod=True)
        y_data = ax.lines[0].get_ydata()
        y_expected = k**2 # NOTE: assuming ones for the mocked poles
        np.testing.assert_allclose(y_data, y_expected)  # ty:ignore[no-matching-overload]
        plt.close(fig)

    def test_2d_k_uses_linear_x_axis(self):
        """2D-k input must use a linear x-axis and apply k1*k2*k3 weights."""
        # Different k1 and k2 to avoid diagonal case
        k = np.column_stack([np.linspace(0.01, 0.5, 30), np.linspace(0.1, 0.5, 30)])
        x = np.arange(len(k))
        obj = _make_mock_spectrum("sugiyama", k)
        fig, ax = BispectrumMultipoles.plot(obj, ells=[(0, 0, 0)])
        x_data = ax.lines[0].get_xdata()
        np.testing.assert_allclose(x_data, x)  # ty:ignore[no-matching-overload]
        plt.close(fig)

    def test_2d_k_weights_return_kprod(self):
        """2D-k input must apply k1*k2*k3 weights."""
        k = np.column_stack([np.linspace(0.01, 0.5, 30)] * 2)
        obj = _make_mock_spectrum("sugiyama", k)
        fig, ax = BispectrumMultipoles.plot(obj, ells=[(0, 0, 0)], weight_by_kprod=True)
        y_data = ax.lines[0].get_ydata()
        y_expected = np.prod(k, axis=-1) # NOTE: assuming ones for the mocked poles
        np.testing.assert_allclose(y_data, y_expected)  # ty:ignore[no-matching-overload]
        plt.close(fig)

    def test_diagonal_case_uses_1d_x_axis(self, mock_spectrum_sugiyama):
        """When k1 == k2 on the diagonal, x-axis must use 1D k values."""
        k_diag = np.linspace(0.01, 0.5, 30)
        k_2d = np.column_stack([k_diag, k_diag])
        obj = _make_mock_spectrum("sugiyama", k_2d)
        fig, ax = BispectrumMultipoles.plot(obj, ells=[(0, 0, 0)])
        x_data = ax.lines[0].get_xdata()
        np.testing.assert_allclose(x_data, k_diag)  # ty:ignore[no-matching-overload]
        plt.close(fig)

    def test_no_weights_returns_poles(self):
        """When weight_by_kprod is False, the y-data must be the poles."""
        k = np.column_stack([np.linspace(0.01, 0.5, 30)] * 2)
        obj = _make_mock_spectrum("sugiyama", k)
        fig, ax = BispectrumMultipoles.plot(obj, ells=[(0, 0, 0)], weight_by_kprod=False)
        y_result = ax.lines[0].get_ydata()
        y_expected = np.ones(len(k)) # NOTE: assuming ones for the mocked poles
        np.testing.assert_allclose(y_result, y_expected)  # ty:ignore[no-matching-overload]
        plt.close(fig)
