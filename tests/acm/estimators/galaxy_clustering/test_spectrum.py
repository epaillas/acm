from unittest.mock import MagicMock, patch

import lsstypes
import matplotlib.pyplot as plt
import numpy as np
import pytest

from acm.estimators.galaxy_clustering.backends.jaxpower import JaxpowerBackend
from acm.estimators.galaxy_clustering.spectrum import PowerSpectrumMultipoles

MODULE = "acm.estimators.galaxy_clustering.spectrum"

# ruff: noqa: ANN001, ANN201, ARG002, D101, D102, D103, INP001, S101

@pytest.fixture
def estimator(data_positions, randoms_positions):
    """Fixture for a PowerSpectrumMultipoles estimator with a dummy backend."""
    backend = JaxpowerBackend(data_positions, randoms_positions)
    return PowerSpectrumMultipoles(
        backend=backend,
        data_positions=data_positions,
        randoms_positions=randoms_positions,
    )

@pytest.fixture
def estimator_no_randoms(data_positions):
    """Fixture for a PowerSpectrumMultipoles estimator without randoms."""
    backend = JaxpowerBackend(data_positions, None)
    return PowerSpectrumMultipoles(
        backend=backend,
        data_positions=data_positions,
        randoms_positions=None,
    )

@pytest.fixture
def mock_spectrum():
    obj = MagicMock()
    coords_mock = MagicMock(return_value=np.linspace(0.01, 0.5, 50))
    obj.flatten.return_value = (MagicMock(coords=coords_mock),)
    obj.get.return_value.value.return_value = np.ones(50)
    return obj

class TestInit:

    def test_raise_when_backend_not_jaxpower(self, data_positions, randoms_positions, dummy_backend):
        """TypeError must be raised when backend is not a JaxpowerBackend."""
        with pytest.raises(TypeError, match="requires a JaxpowerBackend"):
            PowerSpectrumMultipoles(
                backend=dummy_backend,  # Not a JaxpowerBackend
                data_positions=data_positions,
                randoms_positions=randoms_positions,
            )

    def test_init_with_jaxpower_backend(self, data_positions, randoms_positions):
        """Initialization should succeed with a JaxpowerBackend."""
        backend = JaxpowerBackend(data_positions, randoms_positions)
        estimator = PowerSpectrumMultipoles(
            backend=backend,
            data_positions=data_positions,
            randoms_positions=randoms_positions,
        )
        assert isinstance(estimator.backend, JaxpowerBackend)

    def test_jit_cm2s_is_jitted_function(self, estimator):
        """Check that jit_cm2s is a jitted function."""
        assert callable(estimator.jit_cm2s) # FIXME: can we check if it's jitted?


class TestCompute:

    @patch(f"{MODULE}.BinMesh2SpectrumPoles")
    @patch(f"{MODULE}.cm2s")
    @patch(f"{MODULE}.compute_box2_normalization")
    @patch(f"{MODULE}.compute_fkp2_normalization")
    @patch(f"{MODULE}.compute_fkp2_shotnoise")
    @patch(f"{MODULE}.FKPField")
    def test_randoms_use_fkp(self, mock_fkpfield, mock_shotnoise, mock_fkp_normalization, mock_box_normalization, mock_cm2s, mock_binmesh, estimator):
        """Test that compute uses FKP weights path when randoms are provided."""
        estimator.jit_cm2s = mock_cm2s  # Mock the jitted function
        estimator.compute(edges={"step": 0.001}, ells=(0, 2, 4), los="z")
        mock_fkpfield.assert_called_once()
        mock_shotnoise.assert_called_once()
        mock_fkp_normalization.assert_called_once()
        mock_box_normalization.assert_not_called()
        mock_cm2s.assert_called_once()
        mock_binmesh.assert_called_once()

    @patch(f"{MODULE}.BinMesh2SpectrumPoles")
    @patch(f"{MODULE}.cm2s")
    @patch(f"{MODULE}.compute_box2_normalization")
    @patch(f"{MODULE}.compute_fkp2_normalization")
    @patch(f"{MODULE}.compute_fkp2_shotnoise")
    @patch(f"{MODULE}.FKPField")
    def test_no_randoms_use_box_normalization(self, mock_fkpfield, mock_shotnoise, mock_fkp_normalization, mock_box_normalization, mock_cm2s, mock_binmesh, estimator_no_randoms):
        """Test that compute uses box normalization path when randoms are not provided."""
        estimator_no_randoms.jit_cm2s = mock_cm2s  # Mock the jitted function
        estimator_no_randoms.compute(edges={"step": 0.001}, ells=(0, 2, 4), los="z")
        mock_fkpfield.assert_not_called()
        mock_shotnoise.assert_called_once()
        mock_box_normalization.assert_called_once()
        mock_fkp_normalization.assert_not_called()
        mock_cm2s.assert_called_once()
        mock_binmesh.assert_called_once()

    @patch(f"{MODULE}.BinMesh2SpectrumPoles")
    @patch(f"{MODULE}.cm2s")
    @patch(f"{MODULE}.compute_box2_normalization")
    @patch(f"{MODULE}.compute_fkp2_normalization")
    @patch(f"{MODULE}.compute_fkp2_shotnoise")
    @patch(f"{MODULE}.FKPField")
    def test_returns_cloned_spectrum(self, mock_fkpfield, mock_shotnoise, mock_fkp_normalization, mock_box_normalization, mock_cm2s, mock_binmesh, estimator):
        """Compute should return a cloned Mesh2SpectrumPoles object."""
        estimator.jit_cm2s = mock_cm2s  # Mock the jitted function
        result = estimator.compute(edges={"step": 0.001}, ells=(0, 2, 4), los="z")
        assert result is mock_cm2s.return_value.clone.return_value

class TestLoad:

    @patch(f"{MODULE}.lsstypes.read")
    def test_load_returns_lsstypes_object(self, mock_read, tmp_path):
        """The load method should return a Mesh2SpectrumPoles object."""
        mock_obj = MagicMock(spec=lsstypes.Mesh2SpectrumPoles)
        mock_read.return_value = mock_obj
        result = PowerSpectrumMultipoles.load(tmp_path / "result.h5")
        assert result is mock_obj
        assert isinstance(result, lsstypes.Mesh2SpectrumPoles)
        mock_read.assert_called_once_with(tmp_path / "result.h5")

class TestPlot:

    def test_create_fig_and_ax_when_none_provided(self, mock_spectrum):
        fig, ax = PowerSpectrumMultipoles.plot(mock_spectrum)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_use_provided_fig_and_ax(self, mock_spectrum):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = PowerSpectrumMultipoles.plot(mock_spectrum, fig=fig_in, ax=ax_in)
        assert fig_out is fig_in
        assert ax_out is ax_in
        plt.close(fig_in)

    def test_draws_one_line_per_multipole(self, mock_spectrum):
        fig, ax = PowerSpectrumMultipoles.plot(mock_spectrum, ells=(0, 2, 4))
        assert len(ax.lines) == 3
        plt.close(fig)

