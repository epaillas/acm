import logging
import time

import numpy as np
from pypower import CatalogMesh

from acm.estimators.galaxy_clustering.backends import filters
from acm.utils.decorators import kwargs_alias

from .base import EstimatorBackend, register_backend

logger = logging.getLogger(__name__)


@register_backend("pypower")
class PypowerBackend(EstimatorBackend):
    """Backend using pypower for galaxy clustering measurements.

    This backend uses the pypower package to create mesh fields from galaxy
    catalogs and compute density contrasts. It supports both data-only and
    data+randoms configurations for FKP-style estimators.

    Attributes
    ----------
    mesh : CatalogMesh
        Pypower mesh object containing data and optional randoms.
    """

    @kwargs_alias(nmesh="meshsize")
    def __init__(
        self,
        data_positions: np.ndarray,
        randoms_positions: np.ndarray | None = None,
        data_weights: np.ndarray | None = None,
        randoms_weights: np.ndarray | None = None,
        interlacing: int = 0,
        resampler: str = "cic",
        **kwargs,
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
        **kwargs : dict
            Additional keyword arguments for mesh attributes.
            Resolves `meshsize` to `nmesh` for pypower compatibility.
            Forces `position_type`to "pos".
            See :func:`pypower.CatalogMesh`
        """
        super().__init__(
            data_positions,
            randoms_positions,
            data_weights,
            randoms_weights,
        )

        mesh = CatalogMesh(
            data_positions=data_positions,
            data_weights=data_weights,
            randoms_positions=randoms_positions,
            randoms_weights=randoms_weights,
            interlacing=interlacing,
            resampler=resampler,
            position_type="pos",  # NOTE: hardcoded w/ position arrays shapes
            **kwargs,
        )

        # Store some extra attributes
        self.mesh = mesh

        logger.debug(
            f"Loaded {self.__class__.__name__} with boxsize {self.boxsize}, box center {self.boxcenter} and meshsize {self.meshsize}"
        )

    @property
    def boxsize(self) -> np.ndarray:
        """Physical size of the box along each dimension."""
        return self.mesh.boxsize

    @property
    def boxcenter(self) -> np.ndarray:
        """Physical coordinates of the box center along each dimension."""
        return self.mesh.boxcenter

    @property
    def meshsize(self) -> np.ndarray:
        """Number of mesh cells along each dimension."""
        return self.mesh.nmesh

    @property
    def cellsize(self) -> np.ndarray:
        """Physical size of each mesh cell."""
        return self.mesh.boxsize / self.mesh.nmesh

    def set_density_contrast(
        self,
        smoothing_radius: float | None = None,
        filter_shape: str = "Gaussian",
        **kwargs,
    ) -> None:
        """
        Compute the density contrast field.

        Paints particles to a mesh and computes the density contrast using
        either data only or data+randoms (FKP method). Optionally applies
        smoothing with a specified filter.

        Parameters
        ----------
        smoothing_radius: float, optional
            Gaussian smoothing scale in Mpc/h. If None, no smoothing is applied.
        threshold: float, optional
            Threshold value for randoms field to avoid division by zero.
            Defaults to 0.01.
        method: str, optional
            Method to compute randoms threshold. Options: 'noise' or 'mean'.
            Defaults to "noise"
        filter_shape: str, optional
            Shape of the smoothing filter. Use one of the filters
            in :mod:`filters`.
            Defaults to "Gaussian"
        **kwargs
            Arguments passed when painting particles to mesh.
            See :meth:`pypower.CatalogMesh.to_mesh`
        """
        t0 = time.time()

        if smoothing_radius is not None:
            logger.info(
                f"Smoothing with {smoothing_radius} Mpc/h {filter_shape} kernel."
            )
            kernel = self._get_kernel(filter_shape, smoothing_radius)
        else:
            kernel = self._get_kernel("NoFilter", 0.0)

        data_mesh = self.mesh.to_mesh(field="data", **kwargs)
        _smoothed_mesh = data_mesh.r2c().apply(kernel)
        data_mesh = _smoothed_mesh.c2r()

        if self.mesh.with_randoms:
            randoms_mesh = self.mesh.to_mesh(field="data-normalized_randoms", **kwargs)
            randoms_mesh = randoms_mesh.r2c().apply(kernel)
            randoms_mesh = randoms_mesh.c2r()

            logger.info("Using randoms to compute density contrast.")
            sum_data = np.sum(data_mesh.value)
            sum_randoms = np.sum(randoms_mesh.value)
            alpha = sum_data / sum_randoms
            delta_mesh = data_mesh - alpha * randoms_mesh

            mask = randoms_mesh > 0
            delta_mesh[mask] /= alpha * randoms_mesh[mask]
            delta_mesh[~mask] = 0.0
        else:
            delta_mesh = data_mesh / np.mean(data_mesh) - 1

        self._density_contrast = np.asarray(delta_mesh)
        logger.info(f"Set density contrast in {time.time() - t0:.2f} s.")

    @staticmethod
    def _get_kernel(filter_shape: str, smoothing_radius: float) -> filters.BaseFilter:
        """Get the matching initialized filter from :mod:`filters`."""
        name = filter_shape.lower()
        if name.startswith("gaussian"):
            f = filters.GaussianFilter(smoothing_radius)
        elif name.startswith("tophat"):
            f = filters.TopHatFilter(smoothing_radius)
        elif name.startswith("nofilter"):
            f = filters.NoFilter(smoothing_radius)
        else:
            raise ValueError(f"{name} filter not found.")
        return f

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
        if method == "randoms":
            rng = np.random.default_rng(seed)
            nquery = nquery or 5 * self.size_data
            coords = rng.random((nquery, 3)) * boxsize + (boxcenter - boxsize / 2)
            logger.info(f"Generated random query points in {time.time() - t0:.2f} s.")
        else:
            raise ValueError("method must be one of ['lattice', 'randoms']")
        return np.asarray(coords, dtype=np.float32)
