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
    ParticleField,
    RealMeshField,
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

    def __init__(
        self,
        mark: RealMeshField,
        data: ParticleField,
        randoms: ParticleField,
        resampler: str = "cic",
        **kwargs,
    ) -> None:
        self.mark = mark
        # Same resampler as used to paint the field, so reading the mark at the
        # particle positions and painting the marked particles are consistent.
        self.resampler = resampler
        super().__init__(data, randoms, **kwargs)

    def clone(self, **kwargs) -> "MarkFKPField":
        """
        Create a new instance, updating some attributes.

        Overrides :meth:`jaxpower.FKPField.clone`, which only forwards ``data``
        and ``randoms``, so that ``mark`` and ``resampler`` are preserved, e.g.
        by :meth:`~jaxpower.FKPField.exchange` in distributed runs.
        """
        state_keys = ["mark", "data", "randoms", "resampler"]
        state = {k: getattr(self, k) for k in state_keys} | kwargs
        return self.__class__(**state)

    @property
    def particles(self) -> ParticleField:
        """Return the marked FKP field as a :class:`~jaxpower.ParticleField`."""
        particles = getattr(self, "_particles", None)

        if particles is None:
            # Interpolate the normalized grid mark at the data-particle positions.
            mark_as_positions_weight = self.mark.read(
                self.data,
                resampler=self.resampler,
            )
            marked_positions = mark_as_positions_weight * self.data

            # Ratio of total galaxy weight to total random weight.
            alpha = self.data.sum() / self.randoms.sum()

            # weights are updated
            particles = (marked_positions - alpha * self.randoms).clone(
                attrs=self.data.attrs
            )
            self.__dict__["_particles"] = particles

        return particles


def _flatten_mark_fkp(field: MarkFKPField) -> tuple[tuple, tuple]:
    return (field.data, field.randoms, field.mark), (field.resampler,)


def _unflatten_mark_fkp(aux: tuple, children: tuple) -> MarkFKPField:
    # Bypass __init__ (children may be jax tracers), as the parent dataclass does.
    field = object.__new__(MarkFKPField)
    data, randoms, mark = children
    field.__dict__.update(data=data, randoms=randoms, mark=mark, resampler=aux[0])
    return field


# Register as a pytree (like the parent FKPField) so the field can be passed
# through jax transformations (e.g. jit), with the mark carried along as data.
jax.tree_util.register_pytree_node(MarkFKPField, _flatten_mark_fkp, _unflatten_mark_fkp)


class MarkedPowerSpectrumMultipoles(BaseEstimator):
    """Calculate marked power-spectrum multipoles using jaxpower."""

    def __init__(
        self,
        backend: str | JaxpowerBackend,
        data_positions: np.ndarray,
        randoms_positions: np.ndarray | None = None,
        data_weights: np.ndarray | None = None,
        randoms_weights: np.ndarray | None = None,
        smoothing_radius: float | None = None,
        coefficients: tuple[float, ...] | list[float] = (0.0, 1.0),
        mark_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the marked power spectrum estimator.

        Sets the mark if ``smoothing_radius`` is given, so that only
        ``__init__`` and :meth:`compute` need to be called. Otherwise logs a
        message, and :meth:`set_mark` must be called before :meth:`compute`.

        Parameters
        ----------
        backend: str | JaxpowerBackend
            The backend to use for the estimator.
        data_positions: np.ndarray
            Positions of data galaxies, of shape (N, 3).
        randoms_positions: np.ndarray, optional
            Positions of random catalog, of shape (M, 3).
        data_weights: np.ndarray, optional
            Weights for data galaxies, of shape (N,).
        randoms_weights: np.ndarray, optional
            Weights for randoms, of shape (M,).
        smoothing_radius: float, optional
            Gaussian smoothing radius of the density contrast used in the mark,
            see :meth:`set_mark`. If None, the mark is not set at initialization.
        coefficients: tuple[float, ...] | list[float], optional
            Polynomial coefficients ``(c_0, c_1, ...)`` of the mark,
            see :meth:`set_mark`. Defaults to (0.0, 1.0).
        mark_kwargs: dict, optional
            Keyword arguments passed to :meth:`set_mark` for painting the density
            contrast, e.g. ``resampler``, ``interlacing`` and ``compensate``.
        **kwargs
            Additional keyword arguments for the backend.
        """
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

        if smoothing_radius is None:
            logger.info("No smoothing radius given, cannot set mark yet.")
        else:
            self.set_mark(
                smoothing_radius=smoothing_radius,
                coefficients=coefficients,
                **(mark_kwargs or {}),
            )

    def set_mark(
        self,
        smoothing_radius: float,
        coefficients: tuple[float, ...] | list[float] = (0.0, 1.0),
        **kwargs,
    ) -> RealMeshField:
        """
        Set the mark from the density contrast smoothed on ``smoothing_radius``.

        The mark is

            m(x) = c_0 + c_1 delta_R(x) + c_2 delta_R(x)^2 + ...

        The smoothed density contrast ``delta_R`` is computed with the backend's
        :meth:`~acm.estimators.galaxy_clustering.backends.jaxpower.JaxpowerBackend.set_density_contrast`,
        but the backend's own density contrast is restored afterwards, so a
        backend shared with other estimators is left unchanged.

        Parameters
        ----------
        smoothing_radius: float
            Gaussian smoothing radius of the density contrast, in Mpc/h.
        coefficients: tuple[float, ...] | list[float], optional
            Polynomial coefficients ``(c_0, c_1, ...)``. Defaults to (0.0, 1.0).
        **kwargs
            Additional keyword arguments for
            :meth:`~acm.estimators.galaxy_clustering.backends.jaxpower.JaxpowerBackend.set_density_contrast`,
            e.g. ``resampler``, ``interlacing`` and ``compensate``.

        Returns
        -------
        mark
            Mesh field containing the mark.
        """
        previous_density_contrast = self.backend._density_contrast
        try:
            self.backend.set_density_contrast(
                smoothing_radius=smoothing_radius, **kwargs
            )
            delta_mesh = self.backend._density_contrast
        finally:
            # Leave the (possibly shared) backend state untouched.
            self.backend._density_contrast = previous_density_contrast

        if delta_mesh is None:
            raise RuntimeError("Backend failed to compute the density contrast.")

        mark = delta_mesh.clone(
            value=delta_mesh.value * 0,
            attrs=delta_mesh.attrs,
        )
        for n, coefficient in enumerate(coefficients):
            mark += coefficient * delta_mesh**n

        self.mark = mark
        self.smoothing_radius = smoothing_radius
        self.coefficients = tuple(coefficients)
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
            raise AttributeError("Mark has not been set. Run set_mark first.")

        t0 = time.time()
        # Read the mark at particle positions with the same resampler used for
        # painting (jaxpower's paint default is "cic").
        resampler = kwargs.get("resampler", "cic")
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
                resampler=resampler,
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
                resampler=resampler,
            )
            normalised_data_at_particle = normalised_mark_at_particle * data_field
            num_shotnoise = compute_fkp2_shotnoise(
                normalised_data_at_particle,
                bin=bin_mesh,
            )

        # TODO: with randoms override los to "firstpoint" as in
        # PowerSpectrumMultipoles.compute. The user-given los is kept for now to
        # reproduce existing measurements; to be changed once validated on cutsky.
        spectrum = self.jit_cm2s(marked_delta_mesh, bin=bin_mesh, los=los)
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

        logger.info(f"Marked power spectrum computed in {time.time() - t0:.2f} s.")
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
