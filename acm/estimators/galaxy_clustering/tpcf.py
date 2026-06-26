import logging
from pathlib import Path

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
    def load(fn: str | Path, **kwargs) -> LsstypeObject:
        """Load an estimator result from file."""
        raise NotImplementedError

    @staticmethod
    def plot(obj: LsstypeObject, **kwargs) -> tuple:
        """Plot the provided estimator result. Return figure and ax."""
        raise NotImplementedError
