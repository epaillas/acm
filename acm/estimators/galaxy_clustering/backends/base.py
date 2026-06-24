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
                    "randoms_weights must be 1D and have the same length as randoms_weights."
                )

        # Assign internal attributes
        self._size_data = len(data_positions)
        self._density_contrast: np.ndarray | None = None

    @property
    def density_contrast(self) -> np.ndarray:
        """Density contrast field."""
        if self._density_contrast is None:
            raise AttributeError(
                "density_contrast has not been set, run set_density_contrast first."
            )
        return self._density_contrast

    @property
    def size_data(self) -> int:
        """Number of data points."""
        return self._size_data

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
    def cellsize(self) -> float:
        """Physical size of each mesh cell."""
        ...

    @abstractmethod
    def set_density_contrast(self, **kwargs) -> None:
        """Compute the density contrast field."""
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
