"""Legacy module for pypower filters."""

from abc import ABC, abstractmethod

import numpy as np


class BaseFilter(ABC):
    """Base class for filters in Fourier space."""

    def __init__(self, r: float) -> None:
        """
        Initialize the filter.

        Parameters
        ----------
        r : float
            The smoothing scale (radius) of the filter in Mpc/h.
        """
        self.r = r

    @abstractmethod
    def __call__(self, k: tuple, v: np.ndarray) -> np.ndarray:
        """
        Apply the filter.

        Parameters
        ----------
        k : tuple of arrays
            Wavenumber components.
        v : np.ndarray
            Field values in Fourier space.
        """
        ...


class TopHatFilter(BaseFilter):
    """
    Top-hat filter in Fourier space.

    Implements a top-hat filter that can be applied to mesh fields in
    Fourier space. Adapted from https://github.com/bccp/nbodykit/.
    """

    def __call__(self, k: tuple, v: np.ndarray) -> np.ndarray:
        """
        Apply the top-hat filter.

        Parameters
        ----------
        k : tuple of arrays
            Wavenumber components.
        v : np.ndarray
            Field values in Fourier space.

        Returns
        -------
        np.ndarray
            Filtered field values.
        """
        r = self.r
        k = sum(ki**2 for ki in k) ** 0.5
        kr = k * r
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 3 * (np.sin(kr) / kr**3 - np.cos(kr) / kr**2)
        w[k == 0] = 1.0
        return w * v


class GaussianFilter(BaseFilter):
    """
    Gaussian filter in Fourier space.

    Implements a Gaussian smoothing filter that can be applied to mesh
    fields in Fourier space.
    """

    def __call__(self, k: tuple, v: np.ndarray) -> np.ndarray:
        """
        Apply the Gaussian filter.

        Parameters
        ----------
        k : tuple of arrays
            Wavenumber components.
        v : np.ndarray
            Field values in Fourier space.

        Returns
        -------
        np.ndarray
            Filtered field values.
        """
        r = self.r
        k2 = sum(ki**2 for ki in k)
        return np.exp(-0.5 * k2 * r**2) * v


class NoFilter(BaseFilter):
    """
    Class that does not applies a filter.

    Implements a call function that returns the initial object.
    """

    def __call__(self, k: tuple, v: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Don't apply the filter."""
        return v
