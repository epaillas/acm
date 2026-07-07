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
            The line-of-sight convention to use.
            See :func:`jaxpower.compute_mesh3_spectrum` for details. Default is "z".
        basis: str, optional
            Basis for the bispectrum computation. Default is "scoccimarro".
        buffer_size: int, optional
            Number of meshes that can be kept in memory by the jaxpower binning operator.
            Default is 30.
        mask_edges: str | list[str] | tuple[str, ...], optional
            Edge mask expression(s) passed directly to :class:`jaxpower.BinMesh3SpectrumPoles`.
        **kwargs
            Additional keyword arguments passed to the backend's paint method and to :func:`jaxpower.compute_fkp3_shotnoise`.
            See :meth:`jaxpower.ParticleField.paint` for details.

        Returns
        -------
        spectrum: lsstypes.Mesh3SpectrumPoles
            The computed bispectrum multipoles as a :class:`~lsstypes.Mesh3SpectrumPoles` object.

        Note
        ----
        If no kwargs are passed, :meth:`jaxpower.ParticleField.paint` and :func:`jaxpower.compute_fkp3_shotnoise` kwargs
        will be set to their default values in jaxpower. The default compensation for the mass assignment scheme differs in these two methods.
        """
        t0 = time.time()
        if ells is None:
            ells = [(0, 0, 0), (0, 0, 2)] if "sugiyama" in basis else (0, 2)

        mattrs = self.backend.mattrs
        bin_mesh = BinMesh3SpectrumPoles(
            mattrs,
            edges=edges,
            ells=ells,
            basis=basis,
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
    def load(filename: str | Path) -> lsstypes.Mesh3SpectrumPoles:
        """Load a :class:`~lsstypes.Mesh3SpectrumPoles` object from file."""
        obj: lsstypes.Mesh3SpectrumPoles = lsstypes.read(filename)
        return obj

    @staticmethod
    def plot(
        obj: LsstypeObject,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        ells: tuple[int, ...] | list[int] | list[tuple[int, int, int]] = (0, 2),
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
        ells: tuple[int, ...] | list[int] | list[tuple[int, int, int]], optional
            List of multipoles to plot. Default is (0, 2).
        weight_by_kprod: bool, optional
            If True, plot the conventional coordinate-weighted bispectrum:
            ``k1 * k2 * k3 * B`` for Scoccimarro bases and ``k1 * k2 * B``
            for Sugiyama bases. Default is True.
        **kwargs
            Additional keyword arguments for the plot.
            See :meth:`matplotlib.pyplot.plot` for details.

        Returns
        -------
        fig, ax: tuple[plt.Figure, plt.Axes]
            The matplotlib figure and axes objects containing the plot.
        """
        # Handle object type here to not break LSP in typing
        if not isinstance(obj, lsstypes.Mesh3SpectrumPoles):
            raise TypeError(f"Expected a Mesh3SpectrumPoles object, got {type(obj)}")
        basis = str(obj.basis).lower()
        args = (obj, fig, ax, ells, weight_by_kprod)
        if "sugiyama" in basis:
            fig, ax = _plot_sugiyama(*args, **kwargs)  # ty:ignore[invalid-argument-type]
        elif "scoccimarro" in basis:
            fig, ax = _plot_scoccimarro(*args, **kwargs)  # ty:ignore[invalid-argument-type]
        else:
            raise ValueError(f"Plot method is not defined for basis {basis}.")
        return fig, ax


# %% Internal plot functions for bispectrum multipoles
def _plot_sugiyama(
    obj: lsstypes.Mesh3SpectrumPoles,
    fig: plt.Figure | None,
    ax: plt.Axes | None,
    ells: list[tuple[int, int, int]],
    weight_by_kprod: bool = True,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot function for the Sugiyama basis bispectrum multipoles.

    Parameters
    ----------
    obj: lsstypes.Mesh3SpectrumPoles
        The :class:`~lsstypes.Mesh3SpectrumPoles` object to plot.
    fig: plt.Figure | None
        The matplotlib figure to plot on. If None, a new figure will be created.
    ax: plt.Axes | None
        The matplotlib axes to plot on. If None, a new axes will be created.
    ells: list[tuple[int, int, int]]
        List of multipoles to plot. Expects a list of 3-tuples for Sugiyama basis multipoles.
    weight_by_kprod: bool, optional
        If True, plot the conventional coordinate-weighted bispectrum:
        ``k1 * k2 * B`` for Sugiyama bases. Default is True.
    **kwargs
        Additional keyword arguments for the plot. See :meth:`matplotlib.pyplot.plot` for details.

    Returns
    -------
    fig, ax: tuple[plt.Figure, plt.Axes]
        The matplotlib figure and axes objects containing the plot.

    Note
    ----
    This function infers 3 cases depending on the shape of the k-coordinates in the :class:`~lsstypes.Mesh3SpectrumPoles` object:
    - 1D-k case: k is a 1D array, and the x-axis will be k with optional k^2 weighting.
    - 2D-k case: k is a 2D array with shape (n, >2), and the x-axis will be k1 with optional k1*k2 weighting.
    - Diagonal case: k1 == k2 on the diagonal, and the x-axis will be k1 with optional k1*k2 weighting.
    """
    k = np.atleast_1d(obj.flatten(level=None)[0].coords("k"))

    # Default values: no weights
    x = np.arange(len(k))
    weights = 1.0
    xlabel = "bin index"
    ylabel = r"$B_\ell(k)$"
    if k.ndim == 1:  # 1D-k case
        x = k
        xlabel = r"$k$ [$h/\mathrm{Mpc}$]"
        if weight_by_kprod:  # Add 1D weights
            weights = k**2
            ylabel = r"$k^2 B_\ell(k, k)$"
    elif weight_by_kprod:  # Add 2D weights
        weights = np.prod(k[..., :2], axis=-1)
        ylabel = r"$k_1 k_2 B_\ell(k_1, k_2)$"

    # Diagonal case: k1 = k2
    if k.shape[-1] >= 2 and np.allclose(k[..., 1], k[..., 0]):
        x = k[..., 0]
        xlabel = r"$k$ [$h/\mathrm{Mpc}$]"

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    for ell in ells:
        pole = obj.get(ells=ell).value().real
        ax.plot(x, weights * pole, label=rf"$\ell={ell}$", **kwargs)
    return fig, ax


def _plot_scoccimarro(
    obj: lsstypes.Mesh3SpectrumPoles,
    fig: plt.Figure | None,
    ax: plt.Axes | None,
    ells: list[int] | tuple[int, ...],
    weight_by_kprod: bool = True,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot function for the Scoccimarro basis bispectrum multipoles.

    Parameters
    ----------
    obj: lsstypes.Mesh3SpectrumPoles
        The :class:`~lsstypes.Mesh3SpectrumPoles` object to plot.
    fig: plt.Figure | None
        The matplotlib figure to plot on. If None, a new figure will be created.
    ax: plt.Axes | None
        The matplotlib axes to plot on. If None, a new axes will be created.
    ells: list[int] | tuple[int, ...]
        List of multipoles to plot. Expects a list of integers for Scoccimarro basis multipoles.
    weight_by_kprod: bool, optional
        If True, plot the conventional coordinate-weighted bispectrum:
        ``k1 * k2 * k3 * B`` for Scoccimarro bases. Default is True.
    **kwargs
        Additional keyword arguments for the plot. See :meth:`matplotlib.pyplot.plot` for details.

    Returns
    -------
    fig, ax: tuple[plt.Figure, plt.Axes]
        The matplotlib figure and axes objects containing the plot.

    Note
    ----
    This function infers 2 cases depending on the shape of the k-coordinates in the :class:`~lsstypes.Mesh3SpectrumPoles` object:
    - 1D-k case: k is a 1D array, and the x-axis will be k with optional k^3 weighting.
    - 2D-k case: k is a 2D array with shape (n, 3), and the x-axis will be k1 with optional k1*k2*k3 weighting.
    """
    k = np.atleast_1d(obj.flatten(level=None)[0].coords("k"))

    # Default values: no weights
    x = np.arange(len(k))
    weights = 1.0
    xlabel = "bin index"
    ylabel = r"$B_\ell(k)$"
    if weight_by_kprod:  # Add 3D weights
        weights = np.prod(k, axis=-1) if k.ndim > 1 else k**3
        ylabel = r"$k_1 k_2 k_3 B_\ell(k_1, k_2, k_3)$"

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    for ell in ells:
        pole = obj.get(ells=ell).value().real
        ax.plot(x, weights * pole, label=rf"$\ell={ell}$", **kwargs)
    return fig, ax
