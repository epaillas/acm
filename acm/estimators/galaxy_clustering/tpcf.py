import logging
from pathlib import Path

import lsstypes
import matplotlib.pyplot as plt
from lsstypes.external import from_pycorr
from pycorr import TwoPointCorrelationFunction

from acm.utils.plotting import set_plot_style

from .base import BaseEstimator

logger = logging.getLogger(__name__)


class TwoPointCorrelationFunctionEstimator(BaseEstimator):
    """Estimator for the Two-Point Correlation Function, using :mod:`pycorr`."""

    def compute(self, **kwargs) -> lsstypes.Count2Correlation:
        """Compute the TPCF estimator."""
        correlation = TwoPointCorrelationFunction(
            data_positions1=self.data_positions,
            randoms_positions1=self.randoms_positions,
            boxsize=self.backend.boxsize,
            position_type="pos",  # Positions are of shape (N, 3)
            **kwargs,
        )
        return from_pycorr(correlation)

    @staticmethod
    def load(
        filename: str | Path,
        project: bool = False,
        **kwargs,
    ) -> lsstypes.Count2Correlation:
        """
        Load a Count2Correlation object from file.

        Parameters
        ----------
        filename: str | Path
            Path to the file containing the Count2Correlation object.
        project: bool, optional
            Whether to project the loaded Count2Correlation object onto specified multipoles. Default is False.
        **kwargs
            Additional keyword arguments for the projection. See :meth:`~lsstypes.Count2Correlation.project` for details.

        Returns
        -------
        obj: lsstypes.Count2Correlation
            The loaded Count2Correlation object, optionally projected onto the specified multipoles.
        """
        obj: lsstypes.Count2Correlation = lsstypes.read(filename)
        if project:
            obj = obj.project(**kwargs)
        return obj

    @staticmethod
    @set_plot_style
    def plot(
        obj: lsstypes.Count2Correlation | lsstypes.Count2CorrelationPoles,
        ells: tuple[int, ...] | list[int] = (0, 2, 4),
        **kwargs,
    ) -> tuple:
        """
        Plot the Two-Point Correlation Function (TPCF) from a Count2Correlation or Count2CorrelationPoles object.

        Parameters
        ----------
        obj: lsstypes.Count2Correlation | lsstypes.Count2CorrelationPoles
            The Count2Correlation or Count2CorrelationPoles object to plot.
        ells: tuple[int, ...] | list[int], optional
            List of multipoles to plot. Default is (0, 2, 4).
        **kwargs
            Additional keyword arguments for the plot. See :meth:`matplotlib.pyplot.plot` for details.
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
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel(r"$s$ [Mpc/h]")
            ax.set_ylabel(r"$s^2 \xi(s)$ [Mpc/h]$^2$")

        if isinstance(obj, lsstypes.Count2Correlation):
            logger.debug(f"Got pair counts, projecting to multipoles: {ells}")
            obj = obj.project(ells=ells)

        s = obj.flatten(level=None)[0].coords("s")
        for ell in ells:
            pole = obj.get(ells=ell).value()
            ax.plot(s, pole * s**2, label=rf"\ell={ell}", **kwargs)
        return fig, ax
