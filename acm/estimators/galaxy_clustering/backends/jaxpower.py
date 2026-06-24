import logging
import time

import jax
import numpy as np
from jaxpower import (
    ComplexMeshField,
    MeshAttrs,
    ParticleField,
    RealMeshField,
    get_mesh_attrs,
)

from .base import EstimatorBackend, register_backend

logger = logging.getLogger(__name__)


def _2r(mesh: RealMeshField | ComplexMeshField) -> RealMeshField:
    """FFT, from complex to real, if applicable."""
    if not isinstance(mesh, RealMeshField):
        mesh = mesh.c2r()
    return mesh


def _2c(mesh: RealMeshField | ComplexMeshField) -> ComplexMeshField:
    """FFT, from real to complex, if applicable."""
    if not isinstance(mesh, ComplexMeshField):
        mesh = mesh.r2c()
    return mesh


@register_backend("jaxpower")
class JaxpowerBackend(EstimatorBackend):
    """
    Backend using jaxpower for galaxy clustering measurements.

    This backend uses the jaxpower package to create mesh fields from galaxy
    catalogs and compute density contrasts using JAX for GPU acceleration.
    Supports both data-only and data+randoms configurations for FKP-style
    estimators.

    Attributes
    ----------
    mattrs: MeshAttrs
        Mesh attributes object containing box properties.
    data_mesh: ParticleField
        jaxpower particle field for data.
    randoms_mesh: ParticleField | None
        jaxpower particle field for randoms, if provided.
    """

    def __init__(
        self,
        data_positions: np.ndarray,
        randoms_positions: np.ndarray | None = None,
        data_weights: np.ndarray | None = None,
        randoms_weights: np.ndarray | None = None,
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
            If all kwargs match MeshAttrs fields, they're used directly.
            Otherwise, mesh attributes are inferred from positions.
            See :func:`jax.get_mesh_attrs`
        """
        super().__init__(
            data_positions,
            randoms_positions,
            data_weights,
            randoms_weights,
        )

        pos = [p for p in [data_positions, randoms_positions] if p is not None]
        mattrs: MeshAttrs = get_mesh_attrs(*pos, **kwargs)

        # Create meshes
        data_mesh = ParticleField(
            data_positions,  # ty:ignore[invalid-argument-type]
            data_weights,  # ty:ignore[invalid-argument-type]
            attrs=mattrs,
            exchange=True,
            backend="jax",
        )
        randoms_mesh = None
        if randoms_positions is not None:
            randoms_mesh = ParticleField(
                randoms_positions,  # ty:ignore[invalid-argument-type]
                randoms_weights,  # ty:ignore[invalid-argument-type]
                attrs=self.mattrs,
                exchange=True,
                backend="jax",
            )

        # Store some extra attributes
        self.mattrs = mattrs
        self.data_mesh = data_mesh
        self.randoms_mesh = randoms_mesh

        logger.debug(
            f"Loaded {self.__class__.__name__} with boxsize {self.boxsize}, box center {self.boxcenter} and meshsize {self.meshsize}"
        )

    @property
    def boxsize(self) -> jax.Array:
        """Physical size of the box along each dimension. If None, set to meshsize."""
        return self.mattrs.boxsize

    @property
    def boxcenter(self) -> jax.Array:
        """Physical coordinates of the box center along each dimension."""
        return self.mattrs.boxcenter

    @property
    def meshsize(self) -> np.ndarray:
        """Number of mesh cells along each dimension."""
        return self.mattrs.meshsize

    @property
    def cellsize(self) -> jax.Array:
        """Physical size of each mesh cell."""
        return self.mattrs.cellsize

    def set_density_contrast(
        self,
        smoothing_radius: float | None = None,
        threshold: float = 0.01,
        method: str = "noise",
        **kwargs,
    ) -> None:
        """
        Compute the density contrast field.

        Paints particles to a mesh and computes the density contrast using
        either data only or data+randoms (FKP method). Optionally applies
        Gaussian smoothing.

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
        **kwargs
            Arguments passed when painting particles to mesh.
            See :meth:`jaxpower.RealMeshField.paint`
        """
        t0 = time.time()

        if smoothing_radius is not None:
            logger.info(f"Smoothing with {smoothing_radius} Mpc/h Gaussian kernel.")
            kernel = self.gaussian_kernel(self.mattrs, smoothing_radius)
        else:
            kernel = 1.0  # NOTE: check if it's consistent with older implementation

        data_mesh: RealMeshField = self.data_mesh.paint(out="real", **kwargs)
        _smoothed_mesh: ComplexMeshField = _2c(data_mesh) * kernel  # ty:ignore[unsupported-operator]
        data_mesh = _2r(_smoothed_mesh)

        if self.randoms_mesh is not None:
            ft = self._get_field_threshold(self.randoms_mesh, threshold, method)

            randoms_mesh: RealMeshField = self.randoms_mesh.paint(out="real", **kwargs)
            _smoothed_mesh: ComplexMeshField = _2c(randoms_mesh) * kernel  # ty:ignore[unsupported-operator]
            randoms_mesh = _2r(_smoothed_mesh)

            logger.info("Using randoms to compute density contrast.")
            randoms_mesh = _2r(randoms_mesh)
            sum_data: RealMeshField = data_mesh.sum()  # ty:ignore[unresolved-attribute]
            sum_randoms: RealMeshField = randoms_mesh.sum()  # ty:ignore[unresolved-attribute]
            alpha: RealMeshField = sum_data * 1.0 / sum_randoms  # ty:ignore[unsupported-operator]
            delta_mesh: RealMeshField = data_mesh - alpha * randoms_mesh  # ty:ignore[unsupported-operator]

            _val = jax.numpy.where(  # keep values above threshold
                randoms_mesh.value > ft,
                delta_mesh.value / (alpha * randoms_mesh.value),  # ty:ignore[unsupported-operator]
                0.0,
            )
            delta_mesh = delta_mesh.clone(value=_val)
        else:
            delta_mesh: RealMeshField = data_mesh / data_mesh.mean() - 1  # ty:ignore[unresolved-attribute]

        self._density_contrast = np.asarray(delta_mesh)
        logger.info(f"Set density contrast in {time.time() - t0:.2f} s.")

    @staticmethod
    def gaussian_kernel(mattrs: MeshAttrs, smoothing_radius: float = 0.0) -> jax.Array:
        """
        Generate Gaussian smoothing kernel in Fourier space.

        Parameters
        ----------
        mattrs : MeshAttrs
            Mesh attributes object.
        smoothing_radius : float, default=10.0
            Smoothing scale in Mpc/h.

        Returns
        -------
        jax.Array
            Gaussian kernel in Fourier space.
        """
        coords = mattrs.kcoords(sparse=True)
        return jax.numpy.exp(-0.5 * sum(kc * smoothing_radius**2 for kc in coords))

    @staticmethod
    def _get_field_threshold(
        field: ParticleField,
        threshold: float = 0.01,
        method: str = "noise",
    ) -> float:
        """
        Compute threshold for a particle field to avoid division by zero.

        Parameters
        ----------
        field : ParticleField
            Random particle field.
        threshold : float, default=0.01
            Threshold multiplier.
        method : str, default='noise'
            Method to compute threshold. Options:
            - 'noise': threshold based on shot noise
            - 'mean': threshold based on mean density

        Returns
        -------
        float
            Threshold value for randoms field.
        """
        if method not in ["noise", "mean"]:
            raise ValueError("method must be one of ['noise', 'mean']")

        if method == "noise":
            val = threshold * jax.numpy.sum(field.weights**2) / field.sum()
        else:
            val = threshold * field.sum() / field.size
        return val

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
        if method == "lattice":
            x, y, z = self.mattrs.rcoords()
            xx, yy, zz = jax.numpy.meshgrid(x, y, z)
            coords = jax.numpy.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
            logger.info(f"Generated lattice query points in {time.time() - t0:.2f} s.")
        elif method == "randoms":
            rng = np.random.default_rng(seed)
            nquery = nquery or 5 * self.size_data
            coords = rng.random((nquery, 3)) * boxsize + (boxcenter - boxsize / 2)
            logger.info(f"Generated random query points in {time.time() - t0:.2f} s.")
        else:
            raise ValueError("method must be one of ['lattice', 'randoms']")
        return np.asarray(coords, dtype=np.float32)  # NOTE: float32 mandatory here ?
