import logging
import time

import numpy as np
import numpy.typing as npt
from pypower import CatalogMesh

logger = logging.getLogger(__name__)


class PypowerBackend:
    """Backend using pypower for galaxy clustering measurements.

    This backend uses the pypower package to create mesh fields from galaxy
    catalogs and compute density contrasts. It supports both data-only and
    data+randoms configurations for FKP-style estimators.

    Attributes
    ----------
    name : str
        Backend name identifier ('pypower').
    mesh : CatalogMesh
        Pypower mesh object containing data and optional randoms.
    size_data : int
        Number of data points.
    meshsize : array_like
        Number of mesh cells in each dimension.
    cellsize : array_like
        Size of each mesh cell.
    boxsize : array_like
        Size of the simulation box.
    boxcenter : array_like
        Center coordinates of the box.
    delta_mesh : array_like
        Density contrast field (set by set_density_contrast).
    data_mesh : array_like
        Data mesh field (set by set_density_contrast).
    """

    def __init__(
        self,
        data_positions: npt.NDArray,
        data_weights: npt.NDArray | None = None,
        randoms_positions: npt.NDArray | None = None,
        randoms_weights: npt.NDArray | None = None,
        **kwargs,
    ) -> None:
        """Initialize the pypower backend.

        Parameters
        ----------
        data_positions : array_like, shape (N, 3)
            Positions of data galaxies.
        data_weights : array_like, shape (N,), optional
            Weights for data galaxies.
        randoms_positions : array_like, shape (M, 3), optional
            Positions of random catalog.
        randoms_weights : array_like, shape (M,), optional
            Weights for randoms.
        **kwargs : dict
            Additional keyword arguments passed to CatalogMesh.
            Common options include:
            - boxsize : float or array_like
                Size of the box.
            - boxcenter : float or array_like
                Center of the box.
            - meshsize or nmesh : int or array_like
                Number of mesh cells per dimension.
        """
        self.name = "pypower"
        if "meshsize" in kwargs:
            kwargs["nmesh"] = kwargs.pop("meshsize")
        self.mesh = CatalogMesh(
            data_positions=data_positions,
            data_weights=data_weights,
            randoms_positions=randoms_positions,
            randoms_weights=randoms_weights,
            interlacing=0,
            resampler="cic",
            position_type="pos",
            **kwargs,
        )

        self.size_data = len(data_positions)
        self.meshsize = self.mesh.nmesh
        self.cellsize = self.mesh.boxsize / self.mesh.nmesh
        self.boxsize = self.mesh.boxsize
        self.boxcenter = self.mesh.boxcenter
        logger.info(f"Box size: {self.boxsize}")
        logger.info(f"Box center: {self.boxcenter}")
        logger.info(f"Box meshsize: {self.meshsize}")

    @property
    def has_randoms(self) -> bool:
        """Check if the backend has randoms.

        Returns
        -------
        bool
            True if random catalog is present, False otherwise.
        """
        return self.mesh.with_randoms

    def set_density_contrast(
        self,
        smoothing_radius: float | None = None,
        compensate: bool = False,
        filter_shape: str = "Gaussian",
    ) -> npt.NDArray:
        """Compute the density contrast field.

        Paints data (and optionally randoms) to a mesh and computes the density
        contrast. For data+randoms, uses the FKP method. Optionally applies
        smoothing with a specified filter.

        Parameters
        ----------
        smoothing_radius : float, optional
            Smoothing scale in Mpc/h. If provided, applies smoothing filter.
        compensate : bool, default=False
            Whether to compensate for the mass assignment window function.
        filter_shape : str, default='Gaussian'
            Shape of the smoothing filter. Options: 'Gaussian' or 'TopHat'.

        Returns
        -------
        delta_mesh : array_like
            Density contrast field.
        """
        t0 = time.time()
        data_mesh = self.mesh.to_mesh(field="data", compensate=compensate)
        if smoothing_radius:
            logger.info(f"Smoothing with {smoothing_radius} Mpc/h Gaussian kernel.")
            fs = getattr(self, filter_shape)
            data_mesh = data_mesh.r2c().apply(fs(r=smoothing_radius))
            data_mesh = data_mesh.c2r()
        if self.has_randoms:
            randoms_mesh = self.mesh.to_mesh(
                field="data-normalized_randoms",
                compensate=compensate,
            )
            if smoothing_radius:
                randoms_mesh = randoms_mesh.r2c().apply(fs(r=smoothing_radius))
                randoms_mesh = randoms_mesh.c2r()
            sum_data, sum_randoms = np.sum(data_mesh.value), np.sum(randoms_mesh.value)
            alpha = sum_data / sum_randoms
            delta_mesh = data_mesh - alpha * randoms_mesh
            mask = randoms_mesh > 0
            delta_mesh[mask] /= alpha * randoms_mesh[mask]
            delta_mesh[~mask] = 0.0
            # shift = self.mesh.boxsize / 2 - self.mesh.boxcenter
            self.randoms_mesh = randoms_mesh
        else:
            self.mean = np.mean(data_mesh)
            delta_mesh = data_mesh / self.mean - 1
        self.data_mesh = data_mesh
        self.delta_mesh = delta_mesh
        logger.info(f"Set density contrast in {time.time() - t0:.2f} s.")
        return self.delta_mesh

    def get_query_positions(
        self,
        method: str = "randoms",
        nquery: int | None = None,
        seed: int = 42,
    ) -> npt.NDArray:
        """Generate query positions to sample the density PDF.

        Creates either a regular lattice of points at mesh cell centers or
        random points within the mesh for sampling the density field.

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
            Query positions.
        """
        boxcenter = self.boxcenter
        boxsize = self.boxsize
        cellsize = self.cellsize
        if method == "lattice":
            logger.info("Generating lattice query points within the box.")
            xedges = np.arange(
                boxcenter[0] - boxsize[0] / 2 - cellsize[0] / 2,
                boxcenter[0] + boxsize[0] / 2,
                cellsize[0],
            )
            yedges = np.arange(
                boxcenter[1] - boxsize[1] / 2 - cellsize[1] / 2,
                boxcenter[1] + boxsize[1] / 2,
                cellsize[1],
            )
            zedges = np.arange(
                boxcenter[2] - boxsize[2] / 2 - cellsize[2] / 2,
                boxcenter[2] + boxsize[2] / 2,
                cellsize[2],
            )
            xcentres = 1 / 2 * (xedges[:-1] + xedges[1:])
            ycentres = 1 / 2 * (yedges[:-1] + yedges[1:])
            zcentres = 1 / 2 * (zedges[:-1] + zedges[1:])
            lattice_x, lattice_y, lattice_z = np.meshgrid(xcentres, ycentres, zcentres)
            lattice_x = lattice_x.flatten()
            lattice_y = lattice_y.flatten()
            lattice_z = lattice_z.flatten()
            return np.vstack((lattice_x, lattice_y, lattice_z)).T
        if method == "randoms":
            logger.info("Generating random query points within the box.")
            np.random.seed(seed)
            if nquery is None:
                nquery = 5 * self.size_data
            return np.random.rand(nquery, 3) * boxsize + (boxcenter - boxsize / 2)
        raise ValueError(f"Unknown method '{method}' for generating query points.")

    class TopHat:
        """Top-hat filter in Fourier space.

        Implements a top-hat filter that can be applied to mesh fields in
        Fourier space. Adapted from https://github.com/bccp/nbodykit/.

        Parameters
        ----------
        r : float
            The radius of the top-hat filter in Mpc/h.
        """

        def __init__(self, r: float) -> None:
            """Initialize the TopHat filter.

            Parameters
            ----------
            r : float
                The radius of the top-hat filter in Mpc/h.
            """
            self.r = r

        def __call__(self, k: tuple, v: npt.NDArray) -> npt.NDArray:
            """Apply the top-hat filter.

            Parameters
            ----------
            k : tuple of arrays
                Wavenumber components.
            v : array_like
                Field values in Fourier space.

            Returns
            -------
            array_like
                Filtered field values.
            """
            r = self.r
            k = sum(ki**2 for ki in k) ** 0.5
            kr = k * r
            with np.errstate(divide="ignore", invalid="ignore"):
                w = 3 * (np.sin(kr) / kr**3 - np.cos(kr) / kr**2)
            w[k == 0] = 1.0
            return w * v

    class Gaussian:
        """Gaussian filter in Fourier space.

        Implements a Gaussian smoothing filter that can be applied to mesh
        fields in Fourier space.

        Parameters
        ----------
        r : float
            The smoothing scale (radius) of the Gaussian filter in Mpc/h.
        """

        def __init__(self, r: float) -> None:
            """Initialize the Gaussian filter.

            Parameters
            ----------
            r : float
                The smoothing scale of the Gaussian filter in Mpc/h.
            """
            self.r = r

        def __call__(self, k: tuple, v: npt.NDArray) -> npt.NDArray:
            """Apply the Gaussian filter.

            Parameters
            ----------
            k : tuple of arrays
                Wavenumber components.
            v : array_like
                Field values in Fourier space.

            Returns
            -------
            array_like
                Filtered field values.
            """
            r = self.r
            k2 = sum(ki**2 for ki in k)
            return np.exp(-0.5 * k2 * r**2) * v
