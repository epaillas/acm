import logging
from pathlib import Path

import lsstypes
import matplotlib.pyplot as plt
from lsstypes.external import from_pycorr
from pycorr import TwoPointCorrelationFunction

from acm.utils.compression import LsstypeObject

from .base import BaseEstimator

logger = logging.getLogger(__name__)


class TwoPointCorrelationFunctionEstimator(BaseEstimator):
    """Estimator for the Two-Point Correlation Function, using :mod:`pycorr`."""

    def compute(self, **kwargs) -> LsstypeObject:
        """Compute the TPCF estimator."""
        correlation = TwoPointCorrelationFunction(
            data_positions1=self.data_positions,
            boxsize=self.backend.boxsize,
            position_type="pos",  # Positions are of shape (N, 3)
            **kwargs,
        )
        return from_pycorr(correlation)

    @staticmethod
    def load(filename: str | Path, ells: list[int] | None = None, **kwargs) -> LsstypeObject:
        """
        Load a Count2Correlation object from file.

        Parameters
        ----------
        filename: str | Path
            Path to the file containing the Count2Correlation object.
        ells: list[int] | None, optional
            List of multipoles to project the Count2Correlation object onto. If None, no projection is performed.
        **kwargs
            Additional keyword arguments for the projection. See :meth:`lsstypes.Count2Correlation.project` for details.

        Returns
        -------
        obj: lsstypes.Count2Correlation
            The loaded Count2Correlation object, optionally projected onto the specified multipoles.
        """
        obj: lsstypes.Count2Correlation = lsstypes.read(filename)
        if ells is not None:
            obj = obj.project(ells=ells, **kwargs)
        return obj

    @staticmethod
    def plot(obj: LsstypeObject, ells: list[int] = [0, 2, 4], **kwargs) -> tuple:
        """
        Plot the Two-Point Correlation Function (TPCF) from a Count2Correlation or Count2CorrelationPoles object.

        Parameters
        ----------
        obj: LsstypeObject
            The Count2Correlation or Count2CorrelationPoles object to plot.
        ells: list[int], optional
            List of multipoles to plot. Default is [0, 2, 4].
        **kwargs
            Additional keyword arguments for the plot. See :meth:`matplotlib.pyplot.subplots` for details.
            Can also include 'fig' and 'ax' to provide existing figure and axes for plotting,
            or 'figsize' to specify the size of the figure if new figure and axes are created.
            If 'fig' and 'ax' are provided, 'figsize' will be ignored.
        """
        if "fig" in kwargs and "ax" in kwargs:
            fig, ax = kwargs["fig"], kwargs["ax"]
        else:
            figsize = kwargs.pop("figsize", (8, 6))
            fig, ax = plt.subplots(figsize=figsize, **kwargs)
            ax.set_xlabel(r"$s$ [Mpc/h]")
            ax.set_ylabel(r"$s^2 \xi(s)$ [Mpc/h]$^2$")

        if isinstance(obj, lsstypes.Count2Correlation):
            logger.info(f"Got pair counts, projecting to multipoles: {ells}")
            obj = obj.project(ells=ells)

        s = obj.flatten(level=None)[0].coords("s")

        for ell in ells:
            pole = obj.get(ells=ell).value()
            ax.plot(s, pole*s**2, label=rf"\ell={ell}", **kwargs)

        return fig, ax
