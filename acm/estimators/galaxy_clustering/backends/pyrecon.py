import logging
import time
from typing import Any

import numpy as np
from pyrecon import RealMesh

from acm.utils.decorators import kwargs_alias

from .base import EstimatorBackend, register_backend

logger = logging.getLogger(__name__)


def _make_array(  # TODO: move this to utils ?
    value: Any,  # noqa: ANN401
    shape: int | tuple[int],
    dtype: str | type = np.float64,
) -> np.ndarray:
    """Return a numpy array by broadcasting the value on the expected shape."""
    toret = np.full(shape, np.nan, dtype=dtype)
    toret[...] = value
    if np.any(np.isnan(toret)):
        raise ValueError(f"Broadcasted {value} to array but found NaN values inside.")
    return toret


@register_backend("pyrecon")
class PyreconBackend(EstimatorBackend):
    """Backend using pyrecon for galaxy clustering measurements.

    This backend uses the pyrecon package to create mesh fields from galaxy
    catalogs and compute density contrasts. It supports both data-only and
    data+randoms configurations for FKP-style estimators.

    Attributes
    ----------
    data_mesh: ParticleField
        pyrecon particle field for data.
    randoms_mesh: ParticleField | None
        pyrecon particle field for randoms, if provided.
    """

    @kwargs_alias(nmesh="meshsize")
    def __init__(
        self,
        data_positions: np.ndarray,
        boxsize: float | np.ndarray,
        meshsize: float | np.ndarray,
        randoms_positions: np.ndarray | None = None,
        data_weights: np.ndarray | None = None,
        randoms_weights: np.ndarray | None = None,
        boxcenter: float | np.ndarray = 0.0,
    ) -> None:
        super().__init__(
            data_positions,
            randoms_positions,
            data_weights,
            randoms_weights,
        )

        # Set private attributes
        self._boxsize = _make_array(boxsize, 3)
        self._boxcenter = _make_array(boxcenter, 3)
        self._meshsize = _make_array(meshsize, 3, dtype=int)

        data_mesh = RealMesh(boxsize=boxsize, boxcenter=boxcenter, nmesh=meshsize)
        data_mesh.assign_cic(data_positions, data_weights, wrap=True)

        randoms_mesh = None
        if randoms_positions is not None:
            randoms_mesh = RealMesh(
                boxsize=boxsize, boxcenter=boxcenter, nmesh=meshsize
            )
            randoms_mesh.assign_cic(randoms_positions, randoms_weights, wrap=True)

        # Store some extra attributes
        self.data_mesh = data_mesh
        self.randoms_mesh = randoms_mesh

        self._density_contrast = None

        logger.debug(
            f"Loaded {self.__class__.__name__} with boxsize {self.boxsize}, box center {self.boxcenter} and meshsize {self.meshsize}"
        )

    @property
    def boxsize(self) -> np.ndarray:
        """Physical size of the box along each dimension. If None, set to meshsize."""
        return self._boxsize

    @property
    def boxcenter(self) -> np.ndarray:
        """Physical coordinates of the box center along each dimension."""
        return self._boxcenter

    @property
    def meshsize(self) -> np.ndarray:
        """Number of mesh cells along each dimension."""
        return self._meshsize

    @property
    def cellsize(self) -> np.ndarray:
        """Physical size of each mesh cell."""
        return self._boxsize / self._meshsize

    def set_density_contrast(
        self,
        smoothing_radius: float | None = None,
        threshold: float = 0.01,
        **kwargs,
    ) -> None:
        """
        Compute the density contrast field.

        Computes the density contrast using
        either data only or data+randoms (FKP method). Optionally applies
        Gaussian smoothing using FFTW.

        Parameters
        ----------
        smoothing_radius: float, optional
            Gaussian smoothing scale in Mpc/h. If None, no smoothing is applied.
        threshold: float, optional
            Threshold value for randoms field to avoid division by zero.
            Defaults to 0.01.
        **kwargs
            Arguments passed when applying gaussian smoothing.
            See :meth:`_apply_smoothing`
        """
        t0 = time.time()

        data_mesh = self.data_mesh  # Already painted
        self._apply_smoothing(data_mesh, smoothing_radius, **kwargs)

        if self.randoms_mesh is not None:
            randoms_mesh = self.randoms_mesh
            self._apply_smoothing(randoms_mesh, smoothing_radius, **kwargs)

            logger.info("Using randoms to compute density contrast.")
            sum_data = np.sum(data_mesh.value)
            sum_randoms = np.sum(randoms_mesh.value)
            alpha = sum_data * 1.0 / sum_randoms
            delta_mesh = data_mesh - alpha * randoms_mesh

            ft = threshold * sum_randoms / self.size_randoms
            mask = randoms_mesh > ft
            delta_mesh[mask] /= alpha * randoms_mesh[mask]
            delta_mesh[~mask] = 0.0
        else:
            delta_mesh = data_mesh / np.mean(data_mesh) - 1

        self._density_contrast = delta_mesh
        logger.info(f"Set density contrast in {time.time() - t0:.2f} s.")

    def _apply_smoothing(
        self,
        mesh: RealMesh,
        radius: float | np.ndarray | None,
        method: str = "fftw",
        **kwargs,
    ) -> None:
        """Apply smoothing radius to a mesh, see :func:`pyrecon.RealMesh.smooth_gaussian`."""
        if radius is not None:
            mesh.smooth_gaussian(radius, method=method, **kwargs)

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
        if self._density_contrast is None:
            raise AttributeError(
                "Density contrast has not been set, run set_density_contrast first."
            )
        if resampler != "cic":
            raise NotImplementedError("Pyrecon backend only supports CIC resampling.")
        t0 = time.time()
        delta = self._density_contrast.read_cic(positions)
        logger.info(f"Read density contrast in {time.time() - t0:.2f} s.")
        return delta

    def get_query_positions(
        self,
        method: str = "randoms",
        nquery: int | None = None,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate query positions to sample the density PDF.

        Creates either a regular lattice of points at mesh cell centers or
        random points within the density mesh for sampling the density field.

        Parameters
        ----------
        method : str, default='randoms'
            Method to generate query points. Options:
            - 'lattice': Regular grid at mesh cell centers
            - 'randoms': Uniformly distributed random points
        nquery : int, optional
            Number of query points when method is 'randoms'.
            Default is 5 times the number of data points.
        seed : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        query_positions : ndarray, shape (nquery, 3)
            Query positions as float32 array.
        """
        t0 = time.time()
        boxcenter = self.boxcenter
        boxsize = self.boxsize
        cellsize = self.cellsize
        if method == "lattice":
            centres: list[np.ndarray] = []
            for ax in range(3):
                start = boxcenter[ax] - boxsize[ax] / 2 - cellsize[ax] / 2
                stop = boxcenter[ax] + boxsize[ax] / 2
                step = cellsize[ax]
                edges = np.arange(start, stop, step)
                centres.append(0.5 * (edges[:-1] + edges[1:]))
            lattice = [_l.flatten() for _l in np.meshgrid(*centres)]
            coords = np.vstack(lattice).T
            logger.info(f"Generated lattice query points in {time.time() - t0:.2f} s.")
        elif method == "randoms":
            rng = np.random.default_rng(seed)
            nquery = nquery or 5 * self.size_data
            coords = rng.random((nquery, 3)) * boxsize + (boxcenter - boxsize / 2)
            logger.info(f"Generated random query points in {time.time() - t0:.2f} s.")
        else:
            raise ValueError("method must be one of ['lattice', 'randoms']")
        return np.asarray(coords, dtype=np.float32)
