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

        Expects positions to be passed in cartesian coordinates of shape (N, 3) and weights to be 1D arrays of shape (N,).

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
        size_data = self._get_size(data_positions, data_weights)
        if randoms_positions is not None:
            size_randoms = self._get_size(randoms_positions, randoms_weights)
        else:
            size_randoms = None
        
        if randoms_weights is not None and randoms_positions is None:
            raise ValueError("randoms_weights requires randoms_positions to be set.")

        # Assign internal attributes
        self._size_data = size_data
        self._size_randoms = size_randoms

        self._density_contrast = None  # Initialized here for access in estimators

    @staticmethod
    def _get_size(positions: np.ndarray, weights: np.ndarray | None) -> int:
        """Get the size of the positions and weights arrays, and perform shape checks."""
        size_pos = positions.shape[0]
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Positions must be of shape (N, 3).")
        if weights is not None:
            if weights.ndim != 1:
                raise ValueError("Weights must be 1D.")
            if weights.shape[0] != size_pos:
                raise ValueError("Weights must have the same length as positions.")
        return size_pos

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
    def boxsize(self) -> tuple | list | np.ndarray:
        """Physical size of the box along each dimension."""
        ...

    @property
    @abstractmethod
    def boxcenter(self) -> tuple | list | np.ndarray:
        """Physical coordinates of the box center along each dimension."""
        ...

    @property
    @abstractmethod
    def meshsize(self) -> tuple | list | np.ndarray:
        """Number of mesh cells along each dimension."""
        ...

    @property
    @abstractmethod
    def cellsize(self) -> tuple | list | np.ndarray:
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
