"""Temporary estimator classes, that will eventually be replaced by ACM standardized classes.""" # noqa: INP001
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import jax
import numpy as np
from lsstypes.external import from_pycorr
from pycorr import TwoPointCorrelationFunction

from acm.utils.compression import LsstypeObject

logger = logging.getLogger('_estimators')

class Estimator(ABC):

    save_ext = ('h5', 'hdf5')

    def __init__(self) -> None:
        self.estimate: LsstypeObject | None = None

    @abstractmethod
    def compute(self, positions: np.ndarray, **kwargs) -> LsstypeObject:
        # TODO: log class name at computation start
        # TODO: handle allowed failure number here !
        ...

    def save(self, filename: str | Path, overwrite: bool = False) -> None:
        fn = Path(filename)
        if jax.process_index() != 0:  # Only process 0 saves to disk
            return  # Exit early for non-zero processes
        if self.estimate is None:
            raise ValueError("Can't save non-existing estimate. Run compute() beforehand.")
        if fn.suffix not in self.save_ext:
            raise ValueError(f"{fn} must have one of the following extensions: {self.save_ext}")
        if fn.exists() and overwrite is False:
            logger.info(f'File {fn} exists and {overwrite=}. Skipping...')
            return

        fn.parent.mkdir(exist_ok=True, parents=True)
        tmp_fn = fn.with_name(fn.stem + ".tmp" + fn.suffix)
        self.estimate.write(tmp_fn)
        logger.info(f"Writing {self.__class__.__name__} estimator to {fn}.")
        tmp_fn.replace(fn)  # Atomic move to avoid partial writes


def get_estimator(stat_name: str) -> type[Estimator]:
    """Return the relevant estimator method for a given stat_name value."""
    raise ValueError(f"{stat_name} is not a known estimator.")


class PycorrEstimator(Estimator):
    """Estimator for Pycorr TwoPointCorrelationFunction."""

    def compute(self, positions: np.ndarray, **kwargs) -> LsstypeObject:
        """
        Compute the TwoPointCorrelationFunction auto-correlation estimator.

        Positions are passed as `data_positions1`; other kwargs are passed as-is.
        See :func:`pycorr.TwoPointCorrelationFunction`
        """
        logger.debug("Computing TPCF.")
        correlation = TwoPointCorrelationFunction(
            data_positions1 = positions,
            **kwargs
        )
        self.estimate = from_pycorr(correlation)
        return self.estimate
