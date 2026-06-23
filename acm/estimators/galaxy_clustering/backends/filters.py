"""Legacy module for pypower filters."""
import numpy as np


class TopHatFilter:
    """
    Top-hat filter in Fourier space.

    Implements a top-hat filter that can be applied to mesh fields in
    Fourier space. Adapted from https://github.com/bccp/nbodykit/.
    """

    def __init__(self, r: float) -> None:
        """
        Initialize the TopHat filter.

        Parameters
        ----------
        r : float
            The radius of the top-hat filter in Mpc/h.
        """
        self.r = r

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

class GaussianFilter:
    """
    Gaussian filter in Fourier space.

    Implements a Gaussian smoothing filter that can be applied to mesh
    fields in Fourier space.
    """

    def __init__(self, r: float) -> None:
        """
        Initialize the Gaussian filter.

        Parameters
        ----------
        r : float
            The smoothing scale (radius) of the Gaussian filter in Mpc/h.
        """
        self.r = r

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
