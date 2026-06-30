import logging
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acm.estimators.galaxy_clustering.backends import EstimatorBackend, load_backend
from acm.typing import LsstypeObject

logger = logging.getLogger(__name__)


class BaseEstimator(ABC):
    """
    Abstract class for galaxy clustering estimators.

    Used to compute the result from the backend, and provides methods to load a similar result from file.
    """

    save_ext = ("h5", "hdf5")

    def __init__(
        self,
        backend: str | EstimatorBackend,
        data_positions: np.ndarray,
        randoms_positions: np.ndarray | None = None,
        data_weights: np.ndarray | None = None,
        randoms_weights: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """Initialize the estimator with the specified backend."""
        # NOTE: Backend provides tests for the position and weights shapes
        self.backend = load_backend(
            backend,
            data_positions=data_positions,
            randoms_positions=randoms_positions,
            data_weights=data_weights,
            randoms_weights=randoms_weights,
            **kwargs,
        )
        # NOTE: no density contrast assignation because loaded bacend might already have it !
        logger.info(
            f"Initializing {self.__class__.__name__} with {self.backend.__class__.__name__}"
        )

        self.data_positions = data_positions
        self.randoms_positions = randoms_positions
        self.data_weights = data_weights
        self.randoms_weights = randoms_weights

    def __repr__(self) -> str:  # pragma: no cover
        """Provide a string representation of the estimator, including backend."""
        return f"{self.__class__.__name__}(backend={self.backend.__class__.__name__})"

    def save(
        self,
        obj: LsstypeObject,
        save_fn: str | Path,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """
        Save the provided estimator result to a .h5 file.

        Parameters
        ----------
        obj: LsstypeObject
            Estimator result to save.
        save_fn: str | Path
            Path to the file where the estimator result will be saved.
        overwrite: bool, optional
            Whether to overwrite the file if it already exists. Defaults to False.
        **kwargs
            Optional arguments for :class:`h5py.File`
        """
        save_fn = Path(save_fn)
        if save_fn.suffix not in self.save_ext:
            raise ValueError(
                f"{save_fn} must have one of the following extensions: {self.save_ext}"
            )
        if save_fn.exists() and overwrite is False:
            logger.info(f"File {save_fn} exists and {overwrite=}. Skipping...")
            # NOTE: Should this be at INFO or WARNING level ?
            return

        save_fn.parent.mkdir(exist_ok=True, parents=True)
        self._atomic_write(obj, save_fn, **kwargs)
        logger.info(f"Writing {self.__class__.__name__} estimator to {save_fn}.")

    @staticmethod
    def _atomic_write(obj: LsstypeObject, filename: Path, **kwargs) -> None:
        """
        Write data to a temporary file moved to the final file to avoid partial write issues.

        Parameters
        ----------
        obj: LsstypeObject
            Object to write to file.
        filename: Path
            Path used to create the temporary file and make the final move.
        **kwargs
            Optional arguments for :class:`h5py.File`
        """
        tmp_fn = filename.with_name(filename.stem + ".tmp" + filename.suffix)
        obj.write(tmp_fn, **kwargs)
        tmp_fn.replace(filename)  # Atomic move to avoid partial writes

    @abstractmethod
    def compute(self) -> LsstypeObject:
        """Compute the estimator."""
        ...

    @staticmethod
    @abstractmethod
    def load(filename: str | Path) -> LsstypeObject:
        """Load an estimator result from file."""
        ...

    @staticmethod
    @abstractmethod
    def plot(
        obj: LsstypeObject,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the provided estimator result. Return figure and ax."""
        ...
