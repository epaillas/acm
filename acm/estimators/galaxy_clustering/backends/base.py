from abc import ABC, abstractmethod

import numpy as np

from acm.utils.backends import BackendRegistry


class EstimatorBackend(ABC):
    """Root abstract class for all estimator backends."""

    def __init__(
        self,
        data_positions: np.ndarray,
        randoms_positions: np.ndarray | None = None,
        data_weights: np.ndarray | None = None,
        randoms_weights: np.ndarray | None = None,
    ) -> None:
        """
        Initialize the backend positional properties.

        Parameters
        ----------
        data_positions: np.ndarray
            Positions of data galaxies, of shape (N, 3).
        randoms_positions: np.ndarray, optional
            Positions of random catalog, of shape (M, 3).
        data_weights: np.ndarray, optional
            Weights for data galaxies, of shape (N,).
        randoms_weights: np.ndarray, optional
            Weights for randoms, of shape (M,).
        """
        # Shape checks
        if data_positions.ndim != 2 or data_positions.shape[1] != 3:
            raise ValueError("data_positions must be of shape (N, 3).")

        if (randoms_positions is not None) and (
            randoms_positions.ndim != 2 or randoms_positions.shape[1] != 3
        ):
            raise ValueError("randoms_positions must be of shape (M, 3).")

        if (data_weights is not None) and (
            data_weights.ndim != 1 or data_weights.shape[0] != data_positions.shape[0]
        ):
            raise ValueError(
                "data_weights must be 1D and have the same length as data_positions."
            )

        if randoms_weights is not None:
            if randoms_positions is None:
                raise ValueError("randoms_weights requires randoms_positions.")
            if (
                randoms_weights.ndim != 1
                or randoms_weights.shape[0] != randoms_positions.shape[0]
            ):
                raise ValueError(
                    "randoms_weights must be 1D and have the same length as randoms_positions."
                )

        # Assign internal attributes
        self._size_data = len(data_positions)
        self._size_randoms = (
            len(randoms_positions) if randoms_positions is not None else None
        )

    @property
    def size_data(self) -> int:
        """Number of data points."""
        return self._size_data

    @property
    def size_randoms(self) -> int:
        """Number of randoms points."""
        if self._size_randoms is None:
            raise ValueError("Randoms have not been set at initalization.")
        return self._size_randoms

    @property
    @abstractmethod
    def boxsize(self) -> float | list | np.ndarray:
        """Physical size of the box along each dimension."""
        ...

    @property
    @abstractmethod
    def boxcenter(self) -> float | list | np.ndarray:
        """Physical coordinates of the box center along each dimension."""
        ...

    @property
    @abstractmethod
    def meshsize(self) -> float | list | np.ndarray:
        """Number of mesh cells along each dimension."""
        ...

    @property
    @abstractmethod
    def cellsize(self) -> float | list | np.ndarray:
        """Physical size of each mesh cell."""
        ...

    @abstractmethod
    def set_density_contrast(self, **kwargs) -> None:
        """Compute the density contrast field."""
        ...

    @abstractmethod
    def read_density_contrast(
        self,
        positions: np.ndarray,
        resampler: str = "cic",
    ) -> np.ndarray:
        """
        Get the density contrast at the input positions.

        Parameters
        ----------
        positions : np.ndarray
            Input positions.
        resampler : str, optional
            Resampling scheme. Default is 'cic'.

        Returns
        -------
        np.ndarray
            Density contrast at the input positions.
        """
        ...

    @abstractmethod
    def get_query_positions(
        self,
        method: str = "randoms",
        nquery: int | None = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate query positions to sample the density PDF."""
        ...


# Create a registry for estimator backends
_registry = BackendRegistry(EstimatorBackend)
register_backend = _registry.register
load_backend = _registry.load
