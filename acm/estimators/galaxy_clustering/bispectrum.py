import logging
import time
from pathlib import Path

import jax
import lsstypes
import matplotlib.pyplot as plt
import numpy as np
from jaxpower import (
    BinMesh3SpectrumPoles,
    FKPField,
    compute_box3_normalization,
    compute_fkp3_normalization,
    compute_fkp3_shotnoise,
)
from jaxpower import compute_mesh3_spectrum as cm3s

from acm.typing import LsstypeObject

from .backends.jaxpower import JaxpowerBackend
from .base import BaseEstimator

logger = logging.getLogger(__name__)


class BispectrumMultipoles(BaseEstimator):
    """Calculate the bispectrum multipoles using jaxpower. See https://github.com/adematti/jax-power/."""

    def __init__(
        self,
        backend: str | JaxpowerBackend,  # NOTE: restrained backend here
        data_positions: np.ndarray,
        randoms_positions: np.ndarray | None = None,
        data_weights: np.ndarray | None = None,
        randoms_weights: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        if not isinstance(backend, JaxpowerBackend):
            raise TypeError(
                f"BispectrumMultipoles requires a JaxpowerBackend, got {type(backend)}"
            )
        super().__init__(
            backend,
            data_positions,
            randoms_positions,
            data_weights,
            randoms_weights,
            **kwargs,
        )

        self.jit_cm3s = jax.jit(cm3s, static_argnames=["los"], donate_argnums=[0])

        # Type hint
        self.backend: JaxpowerBackend

    def compute(
        self,
        edges: np.ndarray | dict = {"step": 0.01},
        ells: tuple[int, ...] | list[int] | list[tuple[int, int, int]] | None = None,
        los: str = "z",
        basis: str = "scoccimarro",
        buffer_size: int = 30,
        mask_edges: str | list[str] | tuple[str, ...] | None = None,
        **kwargs,
    ) -> lsstypes.Mesh3SpectrumPoles:
        """
        Compute the bispectrum multipoles.

        Parameters
        ----------
        edges: np.ndarray | dict, optional
            The bin edges for the bispectrum. Can be specified as a numpy array
            of edges or as a dictionary with keys such as ``min``, ``max`` and
            ``step``. Default is {"step": 0.01}.
        ells: tuple[int, ...] | list[int] | list[tuple[int, int, int]], optional
            Multipoles to compute. Defaults to (0, 2) for the Scoccimarro basis
            and [(0, 0, 0), (0, 0, 2)] for Sugiyama bases.
        los: str, optional
            The line-of-sight convention to use. See
            :func:`jaxpower.compute_mesh3_spectrum` for details. Default is "z".
        basis: str, optional
            Basis for the bispectrum computation. Default is "scoccimarro".
        buffer_size: int, optional
            Number of meshes that can be kept in memory by the jaxpower binning
            operator. Default is 30.
        mask_edges: str | list[str] | tuple[str, ...], optional
            Edge mask expression(s) passed directly to
            :class:`jaxpower.BinMesh3SpectrumPoles`.
        **kwargs
            Additional keyword arguments passed to the backend's paint method and
            to :func:`jaxpower.compute_fkp3_shotnoise`. See
            :meth:`jaxpower.ParticleField.paint` for details.

        Returns
        -------
        spectrum: lsstypes.Mesh3SpectrumPoles
            The computed bispectrum multipoles as a
            :class:`~lsstypes.Mesh3SpectrumPoles` object.
        """
        t0 = time.time()
        if ells is None:
            ells = self._default_ells(basis)

        mattrs = self.backend.mattrs
        bin_mesh = BinMesh3SpectrumPoles(
            mattrs,
            edges=edges,
            basis=basis,
            ells=ells,
            buffer_size=buffer_size,
            mask_edges=mask_edges,
        )

        data_field = self.backend.data_field

        if self.backend.randoms_field is not None:
            logger.info("Computing bispectrum using FKP estimator with randoms.")
            fkp = FKPField(data_field, self.backend.randoms_field)
            norm = compute_fkp3_normalization(fkp, bin=bin_mesh)
            num_shotnoise = compute_fkp3_shotnoise(
                fkp,
                los=los,
                bin=bin_mesh,
                **kwargs,
            )
            delta_mesh = fkp.paint(out="real", **kwargs)
        else:
            logger.info("Computing bispectrum using box normalization (no randoms)")
            norm = compute_box3_normalization(data_field, bin=bin_mesh)
            num_shotnoise = compute_fkp3_shotnoise(
                data_field,
                los=los,
                bin=bin_mesh,
                **kwargs,
            )
            delta_mesh = data_field.paint(out="real", **kwargs)
            delta_mesh = delta_mesh - delta_mesh.mean()

        spectrum = self.jit_cm3s(delta_mesh, bin=bin_mesh, los=los)
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        logger.info(f"Bispectrum computed in {time.time() - t0:.2f} s.")
        return spectrum

    @staticmethod
    def _default_ells(
        basis: str,
    ) -> tuple[int, ...] | list[tuple[int, int, int]]:
        """Return ACM defaults for the requested bispectrum basis."""
        if "sugiyama" in basis:
            return [(0, 0, 0), (0, 0, 2)]
        return (0, 2)

    @staticmethod
    def load(filename: str | Path) -> lsstypes.Mesh3SpectrumPoles:
        """Load a :class:`~lsstypes.Mesh3SpectrumPoles` object from file."""
        obj: lsstypes.Mesh3SpectrumPoles = lsstypes.read(filename)
        return obj

    @staticmethod
    def plot(
        obj: LsstypeObject,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        ells: int | tuple[int, ...] | list[int] | list[tuple[int, int, int]] = (0, 2),
        weight_by_kprod: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot bispectrum multipoles from a :class:`~lsstypes.Mesh3SpectrumPoles` object.

        Parameters
        ----------
        obj: LsstypeObject
            The :class:`~lsstypes.Mesh3SpectrumPoles` object to plot.
        fig: plt.Figure, optional
            The matplotlib figure to plot on. If None, a new figure is created.
            Defaults to None.
        ax: plt.Axes, optional
            The matplotlib axes to plot on. If None, a new axes is created.
            Defaults to None.
        ells: int | tuple[int, ...] | list[int] | list[tuple[int, int, int]], optional
            List of multipoles to plot. Default is (0, 2).
        weight_by_kprod: bool, optional
            If True, plot the conventional coordinate-weighted bispectrum:
            ``k1 * k2 * k3 * B`` for Scoccimarro bases and ``k1 * k2 * B``
            for Sugiyama bases. Default is True.
        **kwargs
            Additional keyword arguments for the plot. See
            :meth:`matplotlib.pyplot.plot` for details.

        Returns
        -------
        fig, ax: tuple[plt.Figure, plt.Axes]
            The matplotlib figure and axes objects containing the plot.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        xlabel = ylabel = None
        for ell in BispectrumMultipoles._iter_ells_for_plot(obj, ells):
            pole = obj.get(ell)
            x, weights, xlabel, ylabel = BispectrumMultipoles._plot_coordinates(
                pole,
                weight_by_kprod=weight_by_kprod,
            )
            value = np.asarray(pole.value().real)
            ax.plot(x, weights * value, label=rf"$\ell={ell}$", **kwargs)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            if not weight_by_kprod:
                ylabel = r"$B_\ell(k)$"
            ax.set_ylabel(ylabel)
        return fig, ax

    @staticmethod
    def _plot_coordinates(
        pole: lsstypes.Mesh3SpectrumPole,
        weight_by_kprod: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | float, str, str]:
        """Return basis-aware plotting coordinates, weights, and axis labels."""
        k = np.asarray(pole.coords("k"))
        basis = str(getattr(pole, "basis", "")).lower()
        if "sugiyama" in basis:
            return BispectrumMultipoles._sugiyama_plot_coordinates(
                k,
                weight_by_kprod=weight_by_kprod,
            )
        return BispectrumMultipoles._scoccimarro_plot_coordinates(
            k,
            weight_by_kprod=weight_by_kprod,
        )

    @staticmethod
    def _scoccimarro_plot_coordinates(
        k: np.ndarray,
        weight_by_kprod: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | float, str, str]:
        """Return Scoccimarro-style triangle-index plotting arrays."""
        x = np.arange(len(k))
        weights = 1.0
        if weight_by_kprod:
            weights = np.prod(k, axis=-1) if k.ndim > 1 else k**3
        return (
            x,
            weights,
            "bin index",
            r"$k_1 k_2 k_3 B_\ell(k_1, k_2, k_3)$",
        )

    @staticmethod
    def _sugiyama_plot_coordinates(
        k: np.ndarray,
        weight_by_kprod: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | float, str, str]:
        """Return Sugiyama-style plotting arrays, using k on diagonal samples."""
        if k.ndim == 1:
            weights = k**2 if weight_by_kprod else 1.0
            return (
                k,
                weights,
                r"$k$ [$h/\mathrm{Mpc}$]",
                r"$k^2 B_\ell(k, k)$",
            )

        diagonal = k.shape[-1] >= 2 and np.allclose(k[..., 1], k[..., 0])
        weights = np.prod(k[..., :2], axis=-1) if weight_by_kprod else 1.0
        if diagonal:
            return (
                k[..., 0],
                weights,
                r"$k$ [$h/\mathrm{Mpc}$]",
                r"$k^2 B_\ell(k, k)$",
            )
        return (
            np.arange(len(k)),
            weights,
            "bin index",
            r"$k_1 k_2 B_\ell(k_1, k_2)$",
        )

    @staticmethod
    def _iter_ells_for_plot(
        obj: LsstypeObject,
        ells: int | tuple[int, ...] | list[int] | list[tuple[int, int, int]],
    ) -> list[int | tuple[int, int, int]]:
        """Return a list of multipoles, accepting a single Sugiyama tuple."""
        if isinstance(ells, int):
            return [ells]

        if isinstance(ells, tuple) and all(isinstance(ell, int) for ell in ells):
            obj_ells = getattr(obj, "ells", ())
            if ells in obj_ells:
                return [ells]

        return list(ells)
