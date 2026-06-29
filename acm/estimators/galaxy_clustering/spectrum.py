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


class PowerSpectrumMultipoles(BaseEstimator):
    """Calculate the power spectrum multipoles using jaxpower. See https://github.com/adematti/jax-power/."""

    def __init__(
        self,
        backend: str | JaxpowerBackend, # NOTE: restrained backend here
        data_positions: np.ndarray,
        randoms_positions: np.ndarray | None = None,
        data_weights: np.ndarray | None = None,
        randoms_weights: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        if not isinstance(backend, JaxpowerBackend):
            raise TypeError(f"PowerSpectrumMultipoles requires a JaxpowerBackend, got {type(backend)}")
        super().__init__(
            backend,
            data_positions,
            randoms_positions,
            data_weights,
            randoms_weights,
            **kwargs,
        )

        self.jit_cm2s = jax.jit(cm2s, static_argnames=["los"], donate_argnums=[0])

        # Type hint
        self.backend: JaxpowerBackend

    def compute(
        self,
        edges: np.ndarray | dict = {"step": 0.001},
        ells: tuple[int, ...] | list[int] = (0, 2, 4),
        los: str = "z",
        **kwargs,
    ) -> lsstypes.Mesh2SpectrumPoles:
        """
        Compute the power spectrum multipoles.

        Parameters
        ----------
        edges: np.ndarray | dict, optional
            The bin edges for the power spectrum. Can be specified as a numpy array of edges or as a dictionary. Default is {"step": 0.001}.
        ells: tuple[int, ...] | list[int], optional
            The multipoles to compute. Default is (0, 2, 4).
        los: str, optional
            The line-of-sight convention to use. Overriden to "firstpoint" when using randoms.
            See :func:`jaxpower.compute_mesh2_spectrum` for details. Default is "z".
        **kwargs
            Additional keyword arguments for the computation. See :func:`jaxpower.compute_mesh2_spectrum` for details.

        Returns
        -------
        spectrum: lsstypes.Mesh2SpectrumPoles
            The computed power spectrum multipoles as a :class:`~lsstypes.Mesh2SpectrumPoles` object.
        """
        t0 = time.time()
        mattrs = self.backend.mattrs
        bin_mesh = BinMesh2SpectrumPoles(mattrs, edges, ells)

        data_mesh = self.backend.data_field.paint(out="real", **kwargs)

        if self.backend.randoms_field is not None: # <=> randoms_positions is not None but also checks if randoms_field is not None
            logger.info("Computing power spectrum using FKP estimator with randoms.")
            los = "firstpoint" # NOTE: override los to firstpoint when using randoms
            randoms_mesh = self.backend.randoms_field.paint(out="real", **kwargs)
            fkp = FKPField(data_mesh, randoms_mesh)
            norm = compute_fkp2_normalization(fkp, bin=bin_mesh)
            num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin_mesh)
            delta_mesh = fkp.paint(out="real", **kwargs)
        else:
            logger.info("Computing power spectrum using box normalization (no randoms)")
            norm = compute_box2_normalization(data_mesh, bin=bin_mesh)
            num_shotnoise = compute_fkp2_shotnoise(data_mesh, bin=bin_mesh)
            delta_mesh = data_mesh.paint(out="real", **kwargs)
            delta_mesh = delta_mesh - delta_mesh.mean()

        spectrum = self.jit_cm2s(delta_mesh, bin=bin_mesh, los=los)
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        logger.info(f"Power spectrum computed in {time.time() - t0:.2f} s.")
        return spectrum

    @staticmethod
    def load(filename: str | Path) -> lsstypes.Mesh2SpectrumPoles:
        """Load a :class:`~lsstypes.Mesh2SpectrumPoles` object from file."""
        obj: lsstypes.Mesh2SpectrumPoles = lsstypes.read(filename)
        return obj

    @staticmethod
    def plot(
        obj: LsstypeObject,
        ells: tuple[int, ...] | list[int] = (0, 2, 4),
        **kwargs,
    ) -> tuple:
        """
        Plot the Power Spectrum Multipoles from a :class:`~lsstypes.Mesh2SpectrumPoles` object.

        Parameters
        ----------
        obj: LsstypeObject
            The :class:`~lsstypes.Mesh2SpectrumPoles` object to plot.
        ells: tuple[int, ...] | list[int], optional
            List of multipoles to plot. Default is (0, 2, 4).
        **kwargs
            Additional keyword arguments for the plot. See :meth:`matplotlib.pyplot.subplots` for details.
            Can also include 'fig' and 'ax' to provide existing figure and axes for plotting,
            or 'figsize' to specify the size of the figure if new figure and axes are created.
            If 'fig' and 'ax' are provided, 'figsize' will be ignored.

        Returns
        -------
        fig, ax: tuple
            The matplotlib figure and axes objects containing the plot.
        """
        fig = kwargs.pop("fig", None)
        ax = kwargs.pop("ax", None)
        figsize = kwargs.pop("figsize", (8, 6))
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize, **kwargs)
            ax.set_xlabel(r"$k$ [h/Mpc]")
            ax.set_ylabel(r"$P(k)$ [(Mpc/h)$^3$]")

        k = obj.flatten(level=None)[0].coords("k")
        for ell in ells:
            pole = obj.get(ells=ell).value()
            ax.plot(k, pole*k**2, label=rf"\ell={ell}", **kwargs)
        return fig, ax
