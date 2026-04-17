import logging
import time
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jaxpower import (
    ComplexMeshField,
    MeshAttrs,
    ParticleField,
    RealMeshField,
    get_mesh_attrs,
)

logger = logging.getLogger(__name__)


class JaxpowerBackend:
    """Backend using jaxpower for galaxy clustering measurements.

    This backend uses the jaxpower package to create mesh fields from galaxy
    catalogs and compute density contrasts using JAX for GPU acceleration.
    Supports both data-only and data+randoms configurations for FKP-style
    estimators.

    Attributes
    ----------
    name : str
        Backend name identifier ('jaxpower').
    mattrs : MeshAttrs
        Mesh attributes object containing box properties.
    data_mesh : ParticleField
        JAX particle field for data.
    randoms_mesh : ParticleField or None
        JAX particle field for randoms, if provided.
    has_randoms : bool
        Whether random catalog is present.
    size_data : int
        Number of data points.
    boxsize : array_like
        Size of the simulation box.
    boxcenter : array_like
        Center coordinates of the box.
    meshsize : array_like
        Number of mesh cells in each dimension.
    cellsize : array_like
        Size of each mesh cell.
    delta_mesh : RealMeshField or ComplexMeshField
        Density contrast field (set by set_density_contrast).
    """

    def __init__(
        self,
        data_positions: npt.NDArray,
        data_weights: Optional[npt.NDArray] = None,
        randoms_positions: Optional[npt.NDArray] = None,
        randoms_weights: Optional[npt.NDArray] = None,
        **kwargs,
    ) -> None:
        """Initialize the jaxpower backend.

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
            Additional keyword arguments for mesh attributes.
            If all kwargs match MeshAttrs fields, they're used directly.
            Otherwise, mesh attributes are inferred from positions.
            Common options include:
            - boxsize : float or array_like
                Size of the box.
            - boxcenter : float or array_like
                Center of the box.
            - meshsize : int or array_like
                Number of mesh cells per dimension.
        """
        self.name = "jaxpower"
        pos = [p for p in [data_positions, randoms_positions] if p is not None]
        self.mattrs = get_mesh_attrs(*pos, **kwargs)

        self.data_mesh = ParticleField(
            data_positions,
            data_weights,
            attrs=self.mattrs,
            exchange=True,
            backend="jax",
        )
        self.has_randoms = False if randoms_positions is None else True
        self.size_data = len(data_positions)
        if self.has_randoms:
            self.randoms_mesh = ParticleField(
                randoms_positions,
                randoms_weights,
                attrs=self.mattrs,
                exchange=True,
                backend="jax",
            )
        self.boxsize = self.mattrs.boxsize
        self.boxcenter = self.mattrs.boxcenter
        self.meshsize = self.mattrs.meshsize
        self.cellsize = self.mattrs.cellsize
        if jax.process_index() == 0:
            logger.info(f"Box size: {self.boxsize}")
            logger.info(f"Box center: {self.boxcenter}")
            logger.info(f"Box meshsize: {self.meshsize}")

    def set_density_contrast(
        self,
        resampler: str = "cic",
        interlacing: bool = False,
        compensate: bool = False,
        halo_add: int = 0,
        smoothing_radius: Optional[float] = None,
        randoms_threshold_value: float = 0.01,
        randoms_threshold_method: str = "noise",
    ) -> Union[RealMeshField, ComplexMeshField]:
        """Compute the density contrast field.

        Paints particles to a mesh and computes the density contrast using
        either data only or data+randoms (FKP method). Optionally applies
        Gaussian smoothing.

        Parameters
        ----------
        resampler : str, default='cic'
            Resampling scheme for painting particles to mesh.
            Options: 'ngp', 'cic', 'tcs', 'pcs'.
        interlacing : bool, default=False
            Whether to use interlacing to reduce aliasing.
        compensate : bool, default=False
            Whether to apply compensation for the window function.
        halo_add : int, default=0
            Number of halo cells to add around the mesh.
        smoothing_radius : float, optional
            Gaussian smoothing scale in Mpc/h. If None, no smoothing is applied.
        randoms_threshold_value : float, default=0.01
            Threshold value for randoms field to avoid division by zero.
        randoms_threshold_method : str, default='noise'
            Method to compute randoms threshold. Options: 'noise' or 'mean'.

        Returns
        -------
        delta_mesh : RealMeshField or ComplexMeshField
            Density contrast field.
        """

        def _2r(mesh):
            if not isinstance(mesh, RealMeshField):
                mesh = mesh.c2r()
            return mesh

        def _2c(mesh):
            if not isinstance(mesh, ComplexMeshField):
                mesh = mesh.r2c()
            return mesh

        self.randoms_threshold_value = randoms_threshold_value
        self.randoms_threshold_method = randoms_threshold_method

        t0 = time.time()
        kw = dict(
            resampler=resampler,
            compensate=compensate,
            interlacing=interlacing,
            halo_add=halo_add,
        )
        data_mesh = self.data_mesh.paint(**kw, out="real")
        if self.has_randoms:
            randoms_mesh = self.randoms_mesh.paint(**kw, out="real")
            threshold_randoms = self._get_threshold_randoms(
                self.randoms_mesh,
                threshold_value=randoms_threshold_value,
                threshold_method=randoms_threshold_method,
            )
        else:
            threshold_randoms, randoms_mesh = None, None

        kernel = 1.0
        if smoothing_radius is not None:
            if jax.process_index() == 0:
                logger.info(f"Smoothing with {smoothing_radius} Mpc/h Gaussian kernel.")
            kernel = self.kernel_gaussian(
                self.mattrs, smoothing_radius=smoothing_radius
            )
            data_mesh = (_2c(data_mesh) * kernel).c2r()
            if randoms_mesh is not None:
                randoms_mesh = (_2c(randoms_mesh) * kernel).c2r()
        data_mesh = _2r(data_mesh)
        if randoms_mesh is not None:
            logger.info("Using randoms to compute density contrast.")
            randoms_mesh = _2r(randoms_mesh)
            sum_data, sum_randoms = data_mesh.sum(), randoms_mesh.sum()
            alpha = sum_data * 1.0 / sum_randoms
            self.delta_mesh = data_mesh - alpha * randoms_mesh
            if threshold_randoms is not None:
                self.delta_mesh = self.delta_mesh.clone(
                    value=jnp.where(
                        randoms_mesh.value > threshold_randoms,
                        self.delta_mesh.value / (alpha * randoms_mesh.value),
                        0.0,
                    )
                )
            else:
                self.delta_mesh = self.delta_mesh / (alpha * randoms_mesh)
        else:
            self.mean = data_mesh.mean()
            self.delta_mesh = data_mesh / self.mean - 1
        if jax.process_index() == 0:
            logger.info(f"Set density contrast in {time.time() - t0:.2f} s.")
        return self.delta_mesh

    def _get_threshold_randoms(
        self,
        randoms: ParticleField,
        threshold_value: float = 0.01,
        threshold_method: str = "noise",
    ) -> float:
        """Compute threshold for randoms field to avoid division by zero.

        Parameters
        ----------
        randoms : ParticleField
            Random particle field.
        threshold_value : float, default=0.01
            Threshold multiplier.
        threshold_method : str, default='noise'
            Method to compute threshold. Options:
            - 'noise': threshold based on shot noise
            - 'mean': threshold based on mean density

        Returns
        -------
        float
            Threshold value for randoms field.
        """
        assert threshold_method in ["noise", "mean"], (
            "threshold_method must be one of ['noise', 'mean']"
        )

        if threshold_method == "noise":
            threshold_randoms = (
                threshold_value * jnp.sum(randoms.weights**2) / randoms.sum()
            )
        else:
            threshold_randoms = threshold_value * randoms.sum() / randoms.size
        return threshold_randoms

    def get_query_positions(
        self, method: str = "randoms", nquery: Optional[int] = None, seed: int = 42
    ) -> npt.NDArray:
        """Generate query positions to sample the density PDF.

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
            x, y, z = self.mattrs.rcoords()
            xx, yy, zz = jnp.meshgrid(x, y, z)
            coords = jnp.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
            logger.info(f"Generated lattice query points in {time.time() - t0:.2f} s.")
        elif method == "randoms":
            np.random.seed(seed)
            if nquery is None:
                nquery = 5 * self.size_data
            coords = np.random.rand(nquery, 3) * boxsize + (boxcenter - boxsize / 2)
            logger.info(f"Generated random query points in {time.time() - t0:.2f} s.")
        return coords.astype(np.float32)

    def kernel_gaussian(
        self, mattrs: MeshAttrs, smoothing_radius: float = 10.0
    ) -> jnp.ndarray:
        """Generate Gaussian smoothing kernel in Fourier space.

        Parameters
        ----------
        mattrs : MeshAttrs
            Mesh attributes object.
        smoothing_radius : float, default=10.
            Smoothing scale in Mpc/h.

        Returns
        -------
        array_like
            Gaussian kernel in Fourier space.
        """
        return jnp.exp(
            -0.5
            * sum((kk * smoothing_radius) ** 2 for kk in mattrs.kcoords(sparse=True))
        )
