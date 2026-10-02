import logging
import time
from pathlib import Path

import jax
import lsstypes
import matplotlib.pyplot as plt
import numpy as np
from jaxpower import (
    BinMesh2SpectrumPoles,
    FKPField,
    compute_box2_normalization,
    compute_fkp2_normalization,
    compute_fkp2_shotnoise,
)
from jaxpower import compute_mesh2_spectrum as cm2s

from acm.typing import LsstypeObject

from .backends.jaxpower import JaxpowerBackend
from .base import BaseEstimator

logger = logging.getLogger(__name__)


class MarkFKPField(FKPField):
    """Extend jaxpower's FKPField to support marked galaxy fields.
    
    The marked FKP field fluctuation is defined as:
            F_M(x) = m'(x) * n_g(x) - alpha * n_r(x)
        where:
            m'(x) = m(x) / <m> is normalized mark,
            n_g(x) is the galaxy density field,
            n_r(x) is the random catalog density field,
            alpha = sum(w_g) / sum(w_r).
    """

    def __init__(self, mark, data, randoms, **kwargs) -> None:
        self.mark = mark
        super().__init__(data, randoms, **kwargs)

    @property
    def particles(self):
        particles = getattr(self, "_particles", None)

        if particles is None:
            # Interpolate the normalized grid mark at the data-particle positions.
            mark_as_positions_weight = self.mark.read(self.data, resampler="tsc")
            marked_positions = mark_as_positions_weight * self.data

            # Ratio of total galaxy weight to total random weight.
            alpha = self.data.sum() / self.randoms.sum()

            # weights are updated
            particles = (marked_positions - alpha * self.randoms).clone(
                attrs=self.data.attrs
            )
            self.__dict__["_particles"] = particles

        return particles


class MarkedPowerSpectrumMultipoles(BaseEstimator):
    """Calculate marked power-spectrum multipoles using jaxpower."""

    def __init__(
        self,
        backend: str | JaxpowerBackend,
        data_positions: np.ndarray,
        randoms_positions: np.ndarray | None = None,
        data_weights: np.ndarray | None = None,
        randoms_weights: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            backend,
            data_positions,
            randoms_positions,
            data_weights,
            randoms_weights,
            **kwargs,
        )

        self.jit_cm2s = jax.jit(cm2s, static_argnames=["los"], donate_argnums=[0])

        # This estimator relies on jaxpower-specific backend attributes.
        self.backend: JaxpowerBackend

    def compute_mark(
        self,
        smoothing_radius: float = 5.0,
        coefficients: tuple[float, ...] | list[float] = (0.0, 1.0),
        resampler: str = "tsc",
        **kwargs,
    ):
        """
        Compute and store the mark from the smoothed overdensity field.

        The mark is

            m(x) = c_0 + c_1 delta_R(x) + c_2 delta_R(x)^2 + ...

        Parameters
        ----------
        smoothing_radius: float, optional
            Gaussian smoothing radius for the overdensity field. Defaults to 5.0.
        coefficients: tuple[float, ...] | list[float], optional
            Polynomial coefficients ``(c_0, c_1, ...)``. Defaults to (0.0, 1.0).
        resampler: str, optional
            Resampling scheme used when painting the density field. Defaults to "tsc".
        **kwargs
            Additional keyword arguments passed to
            :meth:`JaxpowerBackend.set_density_contrast`, e.g. ``interlacing`` and
            ``compensate``.

        Returns
        -------
        mark
            Mesh field containing the mark.
        """
        self.backend.set_density_contrast(
            smoothing_radius=smoothing_radius,
            resampler=resampler,
            **kwargs,
        )

        delta_mesh = self.backend._density_contrast
        if delta_mesh is None:
            raise RuntimeError("Backend failed to set the density contrast.")

        mark = delta_mesh.clone(
            value=delta_mesh.value * 0,
            attrs=delta_mesh.attrs,
        )
        for n, coefficient in enumerate(coefficients):
            mark += coefficient * delta_mesh**n

        self.mark = mark

        # Preserve the old estimator's memory-saving behavior after the mark is built.
        self.backend._density_contrast = None

        return mark

    def compute(
        self,
        edges: np.ndarray | dict = {"step": 0.001},
        ells: tuple[int, ...] | list[int] = (0, 2, 4),
        los: str = "z",
        **kwargs,
    ) -> lsstypes.Mesh2SpectrumPoles:
        """
        Compute the marked power-spectrum multipoles.

        Parameters
        ----------
        edges: np.ndarray | dict, optional
            Marked-spectrum bin edges. Defaults to {"step": 0.001}.
        ells: tuple[int, ...] | list[int], optional
            Multipoles to compute. Defaults to (0, 2, 4).
        los: str, optional
            Line-of-sight convention passed to jaxpower. Defaults to "z".
        **kwargs
            Additional keyword arguments passed to the jaxpower ``paint`` methods,
            e.g. ``resampler``, ``interlacing`` and ``compensate``.

        Returns
        -------
        spectrum: lsstypes.Mesh2SpectrumPoles
            Computed marked power-spectrum multipoles.
        """
        if not hasattr(self, "mark"):
            raise AttributeError("Mark has not been set. Run compute_mark first.")

        t0 = time.time()
        mattrs = self.backend.mattrs
        bin_mesh = BinMesh2SpectrumPoles(mattrs, edges, ells)
        data_field = self.backend.data_field

        # Paint the unweighted galaxy-density field used to normalize the mark.
        data_mesh = data_field.paint(out="real", **kwargs)

        # Mean mark: <m> = <m n_g> / <n_g>.
        marked_data = self.mark * data_mesh
        mark_bar = marked_data.mean() / data_mesh.mean()
        self.mark_bar = mark_bar
        normalised_mark = self.mark / mark_bar
        marked_data_normalised = normalised_mark * data_mesh
        data_mean = data_mesh.mean()

        if self.backend.randoms_field is not None:
            logger.info(
                "Computing marked power spectrum using FKP estimator with randoms."
            )

            mfkp = MarkFKPField(
                normalised_mark,
                data_field,
                self.backend.randoms_field,
            )
            norm = compute_fkp2_normalization(mfkp, bin=bin_mesh)
            num_shotnoise = compute_fkp2_shotnoise(mfkp, bin=bin_mesh)
            marked_delta_mesh = mfkp.paint(out="real", **kwargs)
        else:
            logger.info(
                "Computing marked power spectrum using box normalization without randoms."
            )

            # Compute normalization with non marked positions
            norm = compute_box2_normalization(data_field, bin=bin_mesh)

            # Marked fluctuation: (m / <m>) n_g - <n_g>.
            marked_delta_mesh = marked_data_normalised - data_mean

            # Compute the shotnoise with the normalised marked positions (weights)
            normalised_mark_at_particle = normalised_mark.read(
                data_field,
                resampler="tsc",
            )
            normalised_data_at_particle = normalised_mark_at_particle * data_field
            num_shotnoise = compute_fkp2_shotnoise(
                normalised_data_at_particle,
                bin=bin_mesh,
            )

        # Preserve the old estimator's actual LOS behavior: do not override to
        # "firstpoint" automatically when randoms are present.
        spectrum = self.jit_cm2s(marked_delta_mesh, bin=bin_mesh, los=los)
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

        logger.info(
            f"Marked power spectrum computed in {time.time() - t0:.2f} s."
        )
        return spectrum

    @staticmethod
    def load(filename: str | Path) -> lsstypes.Mesh2SpectrumPoles:
        """Load a :class:`~lsstypes.Mesh2SpectrumPoles` object from file."""
        obj: lsstypes.Mesh2SpectrumPoles = lsstypes.read(filename)
        return obj

    @staticmethod
    def plot(
        obj: LsstypeObject,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        ells: tuple[int, ...] | list[int] | None = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot marked power-spectrum multipoles."""
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel(r"$k$ [h/Mpc]")
            ax.set_ylabel(r"$k M(k)$ [(Mpc/h)$^3$]")

        ells = ells or obj.ells
        k = obj.flatten(level=None)[0].coords("k")
        for ell in ells:
            pole = obj.get(ells=ell).value()
            ax.plot(k, pole * k, label=rf"$\ell={ell}$", **kwargs)
        return fig, ax
