from unittest.mock import MagicMock, patch

import matplotlib  # noqa: ICN001
import matplotlib.pyplot as plt
import numpy as np
import pytest

from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform

matplotlib.use("Agg")

MODULE = "acm.estimators.galaxy_clustering.wst"

# ruff: noqa: ANN001, ANN202, ANN201, ARG002, D101, D102, D103, INP001, S101

def _make_mock_kymatio_object(J=4, L=4, sigma_0=0.8, max_order=2, integral_powers=(0.8,)):  # noqa: N803
    S = MagicMock()
    S.backend = "torch"
    S.J = J
    S.L = L
    S.sigma_0 = sigma_0
    S.max_order = max_order
    S.integral_powers = integral_powers
    return S

@pytest.fixture
def mock_kymatio_object():
    return _make_mock_kymatio_object()

@pytest.fixture
def estimator(make_estimator, mock_kymatio_object):
    return make_estimator(WaveletScatteringTransform, kymatio_object=mock_kymatio_object)


class TestInit:

    def test_raise_for_multiple_integral_powers(self, dummy_backend, data_positions):
        with pytest.raises(ValueError, match="single integral power"):
            WaveletScatteringTransform(backend=dummy_backend, data_positions=data_positions, integral_powers=(0.8, 1.0))

    def test_use_preloaded_object(self, dummy_backend, data_positions, mock_kymatio_object):
        with patch.object(WaveletScatteringTransform, "initialize_kymatio") as mock_init:
            estimator = WaveletScatteringTransform(backend=dummy_backend, data_positions=data_positions, kymatio_object=mock_kymatio_object)
        assert estimator._S is mock_kymatio_object
        mock_init.assert_not_called()

    def test_initialize_kymatio_object(self, dummy_backend, data_positions):
        with patch.object(WaveletScatteringTransform, "initialize_kymatio") as mock_init:
            WaveletScatteringTransform(
                backend=dummy_backend,
                data_positions=data_positions,
                J = 4,
                L = 4,
                sigma_0 = 0.8,
                integral_powers=(0.8,)
            )
        mock_init.assert_called_once()

    @patch(f"{MODULE}.HarmonicScattering3D")
    def test_initialize_kymatio_object_with_kwargs(self, mock_s3d, dummy_backend, data_positions):
        args = {
            "J": 4,
            "L": 4,
            "sigma_0": 0.8,
            "integral_powers": (0.8,),
        }
        est = WaveletScatteringTransform(
            backend=dummy_backend,
            data_positions=data_positions,
            **args,  # ty:ignore[invalid-argument-type]
        )
        shape = est.backend.meshsize
        assert est._S is mock_s3d.return_value
        mock_s3d.assert_called_once_with(**args, shape=shape, frontend="torch")

class TestCompute:

    def test_raises_for_unsupported_backend(self, estimator):
        """ValueError must be raised when _S.backend has no matching method."""
        estimator._S.backend = "unsupported"
        with pytest.raises(ValueError, match="Unsupported Kymatio backend"):
            estimator.compute()

    @pytest.mark.parametrize("method", ["torch", "jax"])
    def test_dispatches_to_correct_backend_method(self, method, estimator):
        """compute() must call the method matching _S.backend."""
        estimator._S.backend = method
        coefficients = np.ones(10)
        with patch.object(estimator, f"_{method}", return_value=coefficients) as mock_method:
            estimator.compute()
        mock_method.assert_called_once()

    @patch.object(WaveletScatteringTransform, "_torch", return_value=np.ones(10))
    def test_returns_lsstypes_observable_leaf(self, mock_torch_method, estimator):
        """compute() must return an ObservableLeaf with the correct coefficients."""
        leaf = estimator.compute()
        assert hasattr(leaf, "coefficients")
        np.testing.assert_array_equal(leaf.coefficients, mock_torch_method.return_value)
        np.testing.assert_array_equal(leaf.coords("index"), np.arange(len(mock_torch_method.return_value)))

    @patch.object(WaveletScatteringTransform, "_torch", return_value=np.ones(10))
    def test_leaf_attrs_have_correct_values(self, mock_torch_method, estimator):
        """The ObservableLeaf returned by compute() must have the correct attributes."""
        leaf = estimator.compute()
        assert leaf.attrs["J"] == estimator._S.J
        assert leaf.attrs["L"] == estimator._S.L
        assert leaf.attrs["sigma_0"] == estimator._S.sigma_0
        assert leaf.attrs["integral_powers"] == estimator._S.integral_powers
        assert leaf.attrs["frontend"] == estimator._S.backend
        assert leaf.attrs["boxsize"] == list(estimator.backend.boxsize)
        assert leaf.attrs["boxcenter"] == list(estimator.backend.boxcenter)
        assert leaf.attrs["meshsize"] == list(estimator.backend.meshsize)

class TestLoad:

    def test_load_returns_lsstypes_object(self, tmp_path):
        mock_obj = MagicMock()
        with patch(f"{MODULE}.lsstypes.read", return_value=mock_obj) as mock_read:
            result = WaveletScatteringTransform.load(tmp_path / "result.h5")
        mock_read.assert_called_once()
        assert result is mock_obj

class TestPlot:

    @pytest.fixture
    def mock_leaf(self):
        rng = np.random.default_rng(42)
        leaf = MagicMock()
        leaf.index = np.arange(10)
        leaf.coefficients = rng.uniform(0, 1, size=10)
        return leaf

    def test_creates_fig_and_ax_when_none_provided(self, mock_leaf):
        fig, ax = WaveletScatteringTransform.plot(mock_leaf)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_uses_provided_fig_and_ax(self, mock_leaf):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = WaveletScatteringTransform.plot(mock_leaf, fig=fig_in, ax=ax_in)
        assert fig_out is fig_in
        assert ax_out is ax_in
        plt.close(fig_in)

    def test_draws_one_line(self, mock_leaf):
        fig, ax = WaveletScatteringTransform.plot(mock_leaf)
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_custom_figsize(self, mock_leaf):
        fig, _ = WaveletScatteringTransform.plot(mock_leaf, figsize=(10, 3))
        assert tuple(fig.get_size_inches()) == (10.0, 3.0)
        plt.close(fig)
