import logging
from unittest.mock import MagicMock, patch

import lsstypes
import matplotlib  # noqa: ICN001
import matplotlib.pyplot as plt
import numpy as np
import pytest

from acm.estimators.galaxy_clustering.backends.jaxpower import JaxpowerBackend
from acm.estimators.galaxy_clustering.marked_spectrum import (
    MarkedPowerSpectrumMultipoles,
    MarkFKPField,
)

matplotlib.use("Agg")

BACKEND_KWARGS = {
    "boxsize": 100.0,
    "boxcenter": 50.0,
    "meshsize": 16,
}

MODULE = "acm.estimators.galaxy_clustering.marked_spectrum"

# ruff: noqa: ANN001, ANN201, D101, D102, D103, INP001, S101


class DummyMesh:
    """Minimal mesh implementing the arithmetic needed by set_mark."""

    def __init__(self, value, attrs=None):
        self.value = np.asarray(value)
        self.attrs = attrs or {}

    def clone(self, value=None, attrs=None):
        return DummyMesh(
            self.value.copy() if value is None else value,
            self.attrs if attrs is None else attrs,
        )

    def __pow__(self, power):
        return DummyMesh(self.value**power, self.attrs)

    def __mul__(self, other):
        if isinstance(other, DummyMesh):
            other = other.value
        return DummyMesh(self.value * other, self.attrs)

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, DummyMesh):
            other = other.value
        return DummyMesh(self.value + other, self.attrs)


@pytest.fixture
def estimator(data_positions, randoms_positions):
    """Marked-spectrum estimator with randoms and no density contrast."""
    backend = JaxpowerBackend(data_positions, randoms_positions, **BACKEND_KWARGS)
    return MarkedPowerSpectrumMultipoles(
        backend=backend,
        data_positions=data_positions,
        randoms_positions=randoms_positions,
    )


@pytest.fixture
def estimator_no_randoms(data_positions):
    """Marked-spectrum estimator without randoms."""
    backend = JaxpowerBackend(data_positions, None, **BACKEND_KWARGS)
    return MarkedPowerSpectrumMultipoles(
        backend=backend,
        data_positions=data_positions,
        randoms_positions=None,
    )


@pytest.fixture
def mock_spectrum():
    obj = MagicMock()
    obj.ells = (0, 2)
    obj.flatten.return_value = (
        MagicMock(
            coords=MagicMock(
                return_value=np.linspace(0.01, 0.5, 50)
            )
        ),
    )
    obj.get.return_value.value.return_value = np.ones(50)
    return obj


def _prepare_compute(estimator):
    """Replace mesh arithmetic by simple mocks for compute tests."""
    data_field = MagicMock()
    data_mesh = MagicMock()
    data_mesh.mean.return_value = 2.0
    data_field.paint.return_value = data_mesh
    estimator.backend.data_field = data_field

    mark = MagicMock()
    marked_data = MagicMock()
    marked_data.mean.return_value = 6.0
    mark.__mul__.return_value = marked_data

    normalised_mark = MagicMock()
    mark.__truediv__.return_value = normalised_mark

    marked_data_normalised = MagicMock()
    normalised_mark.__mul__.return_value = marked_data_normalised

    estimator.mark = mark
    estimator.jit_cm2s = MagicMock()

    return data_field, normalised_mark, marked_data_normalised


class TestInit:

    def test_no_density_contrast_logs_and_jit_is_callable(
        self,
        data_positions,
        randoms_positions,
        caplog,
    ):
        """No density contrast should leave the mark unset."""
        backend = JaxpowerBackend(data_positions, randoms_positions, **BACKEND_KWARGS)

        with caplog.at_level(logging.INFO):
            est = MarkedPowerSpectrumMultipoles(
                backend=backend,
                data_positions=data_positions,
                randoms_positions=randoms_positions,
            )

        assert "Density contrast not set" in caplog.text
        assert not hasattr(est, "mark")
        assert callable(est.jit_cm2s)

    def test_existing_density_contrast_sets_mark(
        self,
        data_positions,
        randoms_positions,
    ):
        """Initialization should set the mark when delta already exists."""
        backend = JaxpowerBackend(data_positions, randoms_positions, **BACKEND_KWARGS)
        backend._density_contrast = MagicMock()
        coefficients = (1.0, -0.5)

        with patch.object(
            MarkedPowerSpectrumMultipoles,
            "set_mark",
        ) as mock_set_mark:
            MarkedPowerSpectrumMultipoles(
                backend=backend,
                data_positions=data_positions,
                randoms_positions=randoms_positions,
                coefficients=coefficients,
            )

        mock_set_mark.assert_called_once_with(
            coefficients=coefficients
        )

    def test_rejects_non_jaxpower_backend(
        self,
        dummy_backend,
        data_positions,
        randoms_positions,
    ):
        """Only JaxpowerBackend is supported."""
        with pytest.raises(TypeError, match="JaxpowerBackend"):
            MarkedPowerSpectrumMultipoles(
                backend=dummy_backend,
                data_positions=data_positions,
                randoms_positions=randoms_positions,
            )


class TestSetMark:

    def test_raises_when_density_contrast_not_set(self, estimator):
        with pytest.raises(
            RuntimeError,
            match="Backend failed to set the density contrast",
        ):
            estimator.set_mark()

    def test_constructs_polynomial_mark(self, estimator):
        """set_mark should evaluate sum_n c_n delta**n."""
        delta = DummyMesh(
            np.array([-1.0, 0.0, 2.0]),
            attrs={"name": "delta"},
        )
        estimator.backend._density_contrast = delta

        coefficients = (1.0, 2.0, 3.0)
        mark = estimator.set_mark(
            coefficients=coefficients
        )

        expected = (
            1.0
            + 2.0 * delta.value
            + 3.0 * delta.value**2
        )

        np.testing.assert_allclose(
            mark.value,
            expected,
        )
        assert mark.attrs == delta.attrs
        assert estimator.mark is mark


class TestMarkFKPField:

    def test_particles_apply_mark_resampler_and_cache(self):
        """Marked particles should use the resampler and be cached."""
        mark = MagicMock()
        data = MagicMock()
        randoms = MagicMock()

        data.sum.return_value = 2.0
        randoms.sum.return_value = 4.0

        mark_weights = MagicMock()
        marked_positions = MagicMock()
        scaled_randoms = MagicMock()
        particles = MagicMock()
        cloned_particles = MagicMock()

        mark.read.return_value = mark_weights
        mark_weights.__mul__.return_value = marked_positions
        randoms.__rmul__.return_value = scaled_randoms
        marked_positions.__sub__.return_value = particles
        particles.clone.return_value = cloned_particles

        with patch(
            f"{MODULE}.FKPField.__init__",
            return_value=None,
        ):
            field = MarkFKPField(
                mark,
                data,
                randoms,
                resampler="tsc",
            )

        object.__setattr__(field, "data", data)
        object.__setattr__(field, "randoms", randoms)

        first = field.particles
        second = field.particles

        mark.read.assert_called_once_with(
            data,
            resampler="tsc",
        )
        particles.clone.assert_called_once_with(
            attrs=data.attrs
        )

        assert first is cloned_particles        
        assert second is cloned_particles


class TestCompute:

    def test_raises_when_mark_not_set(self, estimator):
        with pytest.raises(
            AttributeError,
            match="Mark has not been set",
        ):
            estimator.compute()

    @patch(f"{MODULE}.MarkFKPField")
    @patch(f"{MODULE}.compute_fkp2_shotnoise")
    @patch(f"{MODULE}.compute_fkp2_normalization")
    @patch(f"{MODULE}.compute_box2_normalization")
    @patch(f"{MODULE}.BinMesh2SpectrumPoles")
    def test_randoms_use_fkp_and_return_cloned_spectrum(
        self,
        mock_binmesh,
        mock_box_normalization,
        mock_fkp_normalization,
        mock_shotnoise,
        mock_mark_fkp,
        estimator,
    ):
        """Randoms should use FKP and return the cloned spectrum."""
        (
            data_field,
            normalised_mark,
            _,
        ) = _prepare_compute(estimator)

        randoms_field = MagicMock()
        estimator.backend.randoms_field = randoms_field

        mfkp = mock_mark_fkp.return_value

        norm = MagicMock()
        shotnoise = MagicMock()

        mock_fkp_normalization.return_value = norm
        mock_shotnoise.return_value = shotnoise

        result = estimator.compute(
            edges={"step": 0.001},
            ells=(0, 2, 4),
            los="z",
            resampler="tsc",
            interlacing=3,
            compensate=True,
        )

        bin_mesh = mock_binmesh.return_value

        data_field.paint.assert_called_once_with(
            out="real",
            resampler="tsc",
            interlacing=3,
            compensate=True,
        )

        mock_mark_fkp.assert_called_once_with(
            normalised_mark,
            data_field,
            randoms_field,
            resampler="tsc",
        )

        mfkp.paint.assert_called_once_with(
            out="real",
            resampler="tsc",
            interlacing=3,
            compensate=True,
        )

        mock_fkp_normalization.assert_called_once_with(
            mfkp,
            bin=bin_mesh,
        )

        mock_box_normalization.assert_not_called()

        mock_shotnoise.assert_called_once_with(
            mfkp,
            bin=bin_mesh,
        )

        estimator.jit_cm2s.assert_called_once_with(
            mfkp.paint.return_value,
            bin=bin_mesh,
            los="z",
        )

        estimator.jit_cm2s.return_value.clone.assert_called_once_with(
            norm=norm,
            num_shotnoise=shotnoise,
        )

        assert (
            result
            is estimator.jit_cm2s.return_value.clone.return_value
        )
        assert estimator.mark_bar == 3.0

    @patch(f"{MODULE}.MarkFKPField")
    @patch(f"{MODULE}.compute_fkp2_shotnoise")
    @patch(f"{MODULE}.compute_fkp2_normalization")
    @patch(f"{MODULE}.compute_box2_normalization")
    @patch(f"{MODULE}.BinMesh2SpectrumPoles")
    def test_no_randoms_use_box_normalization(
        self,
        mock_binmesh,
        mock_box_normalization,
        mock_fkp_normalization,
        mock_shotnoise,
        mock_mark_fkp,
        estimator_no_randoms,
    ):
        """No-randoms measurements should use box normalization."""
        (
            data_field,
            normalised_mark,
            marked_data_normalised,
        ) = _prepare_compute(estimator_no_randoms)

        mark_at_particles = MagicMock()
        normalised_data = MagicMock()
        marked_delta_mesh = MagicMock()

        normalised_mark.read.return_value = (
            mark_at_particles
        )
        mark_at_particles.__mul__.return_value = (
            normalised_data
        )
        marked_data_normalised.__sub__.return_value = (
            marked_delta_mesh
        )

        norm = MagicMock()
        shotnoise = MagicMock()

        mock_box_normalization.return_value = norm
        mock_shotnoise.return_value = shotnoise

        result = estimator_no_randoms.compute(
            edges={"step": 0.001},
            ells=(0, 2, 4),
            los="z",
            resampler="tsc",
        )

        bin_mesh = mock_binmesh.return_value

        mock_mark_fkp.assert_not_called()

        mock_box_normalization.assert_called_once_with(
            data_field,
            bin=bin_mesh,
        )

        mock_fkp_normalization.assert_not_called()

        normalised_mark.read.assert_called_once_with(
            data_field,
            resampler="tsc",
        )

        mock_shotnoise.assert_called_once_with(
            normalised_data,
            bin=bin_mesh,
        )

        estimator_no_randoms.jit_cm2s.assert_called_once_with(
            marked_delta_mesh,
            bin=bin_mesh,
            los="z",
        )

        assert (
            result
            is estimator_no_randoms.jit_cm2s.return_value.clone.return_value
        )


class TestLoad:

    @patch(f"{MODULE}.lsstypes.read")
    def test_load_returns_lsstypes_object(
        self,
        mock_read,
        tmp_path,
    ):
        mock_obj = MagicMock(
            spec=lsstypes.Mesh2SpectrumPoles
        )
        mock_read.return_value = mock_obj

        result = MarkedPowerSpectrumMultipoles.load(
            tmp_path / "result.h5"
        )

        assert result is mock_obj
        assert isinstance(
            result,
            lsstypes.Mesh2SpectrumPoles,
        )

        mock_read.assert_called_once_with(
            tmp_path / "result.h5"
        )


class TestPlot:

    def test_plot_behaviors(self, mock_spectrum):
        """Create axes, reuse supplied axes, and plot each multipole."""
        fig, ax = MarkedPowerSpectrumMultipoles.plot(
            mock_spectrum,
            ells=(0, 2),
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) == 2

        plt.close(fig)

        fig_in, ax_in = plt.subplots()

        fig_out, ax_out = (
            MarkedPowerSpectrumMultipoles.plot(
                mock_spectrum,
                fig=fig_in,
                ax=ax_in,
                ells=(0, 2),
            )
        )

        assert fig_out is fig_in
        assert ax_out is ax_in

        plt.close(fig_in)
