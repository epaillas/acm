from jaxpower import MeshAttrs, ParticleField, FKPField, ComplexMeshField, RealMeshField, get_mesh_attrs
from jax import numpy as jnp
import jax
import numpy as np
import time
import logging


class JaxpowerBackend:
    def __init__(self, data_positions, data_weights=None, randoms_positions=None, randoms_weights=None, **kwargs):
        self.logger = logging.getLogger('JaxpowerBackend')
        self.name = 'jaxpower'
        # Set mesh attributes directly if passed, otherwise infer from positions
        if set(kwargs.keys()).issubset(set(MeshAttrs.__dataclass_fields__.keys())): 
            self.mattrs = MeshAttrs(**kwargs) 
        else:
            if jax.process_index() == 0:
                self.logger.info('Inferring mesh attributes from data and randoms positions.')  
            pos = [p for p in [data_positions, randoms_positions] if p is not None] # get_mesh_attrs raises an error if randoms_positions is None
            self.mattrs = get_mesh_attrs(*pos, **kwargs)
            
        self.data_mesh = ParticleField(data_positions, data_weights, attrs=self.mattrs, exchange=True, backend='jax')
        self.randoms_mesh = None
        self.has_randoms = False if randoms_positions is None else True
        self.size_data = len(data_positions)
        if self.has_randoms:
            self.randoms_mesh = ParticleField(randoms_positions, randoms_weights, attrs=self.mattrs, exchange=True, backend='jax')
        self.boxsize = self.mattrs.boxsize
        self.boxcenter = self.mattrs.boxcenter
        self.meshsize = self.mattrs.meshsize
        self.cellsize = self.mattrs.cellsize
        if jax.process_index() == 0:
            self.logger.info(f'Box size: {self.boxsize}')
            self.logger.info(f'Box center: {self.boxcenter}')
            self.logger.info(f'Box meshsize: {self.meshsize}')

    def set_density_contrast(self, resampler: str='cic', interlacing=False, compensate=False, halo_add: int=0, smoothing_radius: float = None, randoms_threshold_value: float = 0.01, randoms_threshold_method: str = 'noise'):
        def _2r(mesh):
            if not isinstance(mesh, RealMeshField):
                mesh = mesh.c2r()
            return mesh

        def _2c(mesh):
            if not isinstance(mesh, ComplexMeshField):
                mesh = mesh.r2c()
            return mesh

        t0 = time.time()
        kw = dict(resampler=resampler, compensate=compensate, interlacing=interlacing, halo_add=halo_add)
        data_mesh = self.data_mesh.paint(**kw, out='real')
        if self.has_randoms:
            randoms_mesh = self.randoms_mesh.paint(**kw, out='real')
            threshold_randoms = self._get_threshold_randoms(self.randoms_mesh, threshold_value=randoms_threshold_value, threshold_method=randoms_threshold_method)
        else:
            threshold_randoms, randoms_mesh = None, None

        kernel = 1.
        if smoothing_radius is not None:
            if jax.process_index() == 0:
                self.logger.info(f'Smoothing with {smoothing_radius} Mpc/h Gaussian kernel.')
            kernel = self.kernel_gaussian(self.mattrs, smoothing_radius=smoothing_radius)
            data_mesh = (_2c(data_mesh) * kernel).c2r()
            if randoms_mesh is not None:
                randoms_mesh = (_2c(randoms_mesh) * kernel).c2r()
        data_mesh = _2r(data_mesh)
        if randoms_mesh is not None:
            self.logger.info('Using randoms to compute density contrast.')
            randoms_mesh = _2r(randoms_mesh)
            sum_data, sum_randoms = data_mesh.sum(), randoms_mesh.sum()
            alpha = sum_data * 1. / sum_randoms
            self.delta_mesh = data_mesh - alpha * randoms_mesh
            if threshold_randoms is not None:
                self.delta_mesh = self.delta_mesh.clone(value=jnp.where(randoms_mesh.value > threshold_randoms, self.delta_mesh.value / (alpha * randoms_mesh.value), 0.))
        else:
            self.mean = data_mesh.mean()
            print(data_mesh.min(), data_mesh.max(), self.mean)
            self.delta_mesh = data_mesh / self.mean - 1.
        if jax.process_index() == 0:
            self.logger.info(f'Set density contrast in {time.time() - t0:.2f} s.')
        return self.delta_mesh

    def _get_threshold_randoms(self, randoms, threshold_value: float = 0.01, threshold_method: str = 'noise'):
        assert threshold_method in ['noise', 'mean'], "threshold_method must be one of ['noise', 'mean']"

        if threshold_method == 'noise':
            threshold_randoms = threshold_value * jnp.sum(randoms.weights**2) / randoms.sum()
        else:
            threshold_randoms = threshold_value * randoms.sum() / randoms.size
        return threshold_randoms

    def get_query_positions(self, method='randoms', nquery=None, seed=42):
        """
        Get query positions to sample the density PDF, either using random points within the
        density mesh, or the positions at the center of each mesh cell.

        Parameters        
        ----------
        method : str, optional
            Method to generate query points. Options are 'lattice' or 'randoms'.
        nquery : int, optional
            Number of query points used when method is 'randoms'.
        seed : int, optional
            Seed for random number generator.

        Returns
        -------
        query_positions : array_like
            Query positions.
        """
        t0 = time.time()
        boxcenter = self.boxcenter
        boxsize = self.boxsize
        cellsize = self.cellsize
        if method == 'lattice':
            x, y, z = self.mattrs.rcoords()
            xx, yy, zz = jnp.meshgrid(x, y, z)
            coords = jnp.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
            self.logger.info(f'Generated lattice query points in {time.time() - t0:.2f} s.')
        elif method == 'randoms':
            np.random.seed(seed)
            if nquery is None:
                nquery = 5 * self.size_data
            coords = np.random.rand(nquery, 3) * boxsize + (boxcenter - boxsize / 2)
            self.logger.info(f'Generated random query points in {time.time() - t0:.2f} s.')
        return coords.astype(np.float32)

    def kernel_gaussian(self, mattrs: MeshAttrs, smoothing_radius=10.):
        return jnp.exp(- 0.5 * sum((kk * smoothing_radius)**2 for kk in mattrs.kcoords(sparse=True)))