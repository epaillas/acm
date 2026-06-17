"""Temporary estimator classes, that will eventually be replaced by ACM standardized classes.""" # noqa: INP001
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from acm.utils.compression import LsstypeObject


class EstimatorClass(ABC):

    @abstractmethod
    def compute(self, positions: np.ndarray, **kwargs) -> LsstypeObject:
        # TODO: log class name at computation start
        # TODO: handle allowed failure number here !
        ...

    def save(self, filename: str | Path, overwrite: bool = False) -> None:
        filename = Path(filename)
        # TODO: ensure mkdir
        # TODO: enforce filename ?
        # TODO: log save


def get_estimator(stat_name: str) -> type[EstimatorClass]:
    """Return the relevant estimator method for a given stat_name value."""
    raise ValueError(f"{stat_name} is not a known estimator.")
