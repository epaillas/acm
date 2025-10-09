from jaxpower import MeshAttrs, ParticleField, FKPField, ComplexMeshField, RealMeshField
from jax import numpy as jnp
import numpy as np
import time
import jax


class BaseEstimator:
    """
    Base estimator class.
    """
    def __init__(self):
        pass


class BaseEnvironmentEstimator(BaseEstimator):
    """
    Base estimator class for environment-based estimators.
    """
    def __init__(self, data, **kwargs):
        super().__init__()
        self.mattrs = MeshAttrs(**kwargs)
        self.data_mesh = ParticleField(data, attrs=self.mattrs, exchange=True, backend='jax', out='complex')
        self.randoms_mesh = None
        self.size_data = len(data)
        # self.data_mesh = RealMesh(**kwargs)
        if 'randoms' in kwargs:
            self.randoms_mesh = ParticleField(randoms, attrs=self.mattrs, exchange=True, backend='jax', out='complex')
        self.logger.info(f'Box size: {self.data_mesh.boxsize}')
        self.logger.info(f'Box center: {self.data_mesh.boxcenter}')
        self.logger.info(f'Box meshsize: {self.data_mesh.meshsize}')

    @property
    def has_randoms(self):
        return self.randoms_mesh is not None

    def set_density_contrast(self, resampler: str='cic', halo_add: int=0, smoothing_radius: float=15., threshold_randoms: float=0.01):
        data, randoms = self.data_mesh, None
        kw = dict(resampler=resampler, compensate=False, interlacing=0, halo_add=halo_add)
        mesh_data = data.paint(**kw, out='complex')
        del data
        # if self.has_randoms:
        #     mesh_randoms = randoms.paint(**kw, out='complex')
        #     threshold_randoms = _get_threshold_randoms(randoms, threshold_randoms=threshold_randoms)
        # else:
        threshold_randoms, mesh_randoms = None, None
        return self.estimate_mesh_delta(mesh_data, mesh_randoms=mesh_randoms, threshold_randoms=threshold_randoms, smoothing_radius=smoothing_radius)

    def estimate_mesh_delta(self, mesh_data: RealMeshField | ComplexMeshField, mesh_randoms: RealMeshField | ComplexMeshField=None, threshold_randoms: float | jax.Array=None, smoothing_radius: float=15.):
        mattrs = mesh_data.attrs

        def _2r(mesh):
            if not isinstance(mesh, RealMeshField):
                mesh = mesh.c2r()
            return mesh

        def _2c(mesh):
            if not isinstance(mesh, ComplexMeshField):
                mesh = mesh.r2c()
            return mesh

        kernel = 1.
        if smoothing_radius is not None:
            kernel = self.kernel_gaussian(mattrs, smoothing_radius=smoothing_radius)
            mesh_data = (_2c(mesh_data) * kernel).c2r()
            if mesh_randoms is not None:
                mesh_randoms = (_2c(mesh_randoms) * kernel).c2r()

        mesh_data = _2r(mesh_data)
        if mesh_randoms is not None:
            mesh_randoms = _2r(mesh_randoms)
            sum_data, sum_randoms = mesh_data.sum(), mesh_randoms.sum()
            alpha = sum_data * 1. / sum_randoms
            self.delta_mesh = mesh_data - alpha * mesh_randoms
            if threshold_randoms is not None:
                self.delta_mesh = self.delta_mesh.clone(value=jnp.where(mesh_randoms.value > threshold_randoms, self.delta_mesh.value / (alpha * mesh_randoms.value), 0.))
        else:
            self.delta_mesh = mesh_data / mesh_data.mean() - 1.
        return self.delta_mesh

    def _get_threshold_randoms(randoms, threshold_randoms: float=0.01):

        if isinstance(threshold_randoms, tuple):
            threshold_method, threshold_value = threshold_randoms
        else:
            threshold_method, threshold_value = 'noise', threshold_randoms
        assert threshold_method in ['noise', 'mean']

        if threshold_method == 'noise':
            threshold_randoms = threshold_value * jnp.sum(randoms.weights**2) / randoms.sum()
        else:
            threshold_randoms = threshold_value * randoms.sum() / randoms.size
        return threshold_randoms

    # def set_density_contrast(self, smoothing_radius=None, check=False, ran_min=0.01, save_wisdom=False):
    #     """
    #     Set the density contrast.

    #     Parameters
    #     ----------
    #     smoothing_radius : float, optional
    #         Smoothing radius.
    #     check : bool, optional
    #         Check if there are enough randoms.
    #     ran_min : float, optional
    #         Minimum randoms.
    #     nquery_factor : int, optional
    #         Factor to multiply the number of data points to get the number of query points.
            
    #     Returns
    #     -------
    #     delta_mesh : array_like
    #         Density contrast.
    #     """
    #     def _2r(mesh):
    #         if not isinstance(mesh, RealMeshField):
    #             mesh = mesh.c2r()
    #         return mesh

    #     def _2c(mesh):
    #         if not isinstance(mesh, ComplexMeshField):
    #             mesh = mesh.r2c()
    #         return mesh

    #     t0 = time.time()
    #     if smoothing_radius:
    #         self.logger.info(f'Smoothing with radius {smoothing_radius} Mpc/h.')
    #         kernel = self.kernel_gaussian(self.mattrs, smoothing_radius=smoothing_radius)
    #         self.data_mesh = (_2c(self.data_mesh) * kernel).c2r()
    #     self.data_mesh = _2r(self.data_mesh)
    #         # self.data_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=save_wisdom,)
    #     if self.has_randoms:
    #         if smoothing_radius:
    #             self.randoms_mesh *= self.kernel_gaussian(self.randoms_mesh.attrs, smoothing_radius)
    #         fkp = FKPField(self.data_mesh, self.randoms_mesh)
    #         self.delta_mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    #         # if check:
    #         #     mask_nonzero = self.randoms_mesh.value > 0.
    #         #     nnonzero = mask_nonzero.sum()
    #         #     if nnonzero < 2: raise ValueError('Very few randoms.')
    #         # if smoothing_radius:
    #         #     self.randoms_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=save_wisdom)
    #         # sum_data, sum_randoms = np.sum(self.data_mesh.value), np.sum(self.randoms_mesh.value)
    #         # alpha = sum_data * 1. / sum_randoms
    #         # self.delta_mesh = self.data_mesh - alpha * self.randoms_mesh
    #         # self.ran_min = ran_min * sum_randoms / self._size_randoms
    #         # mask = self.randoms_mesh > self.ran_min
    #         # self.delta_mesh[mask] /= alpha * self.randoms_mesh[mask]
    #         # self.delta_mesh[~mask] = 0.0
    #     else:
    #         # self.delta_mesh = self.data_mesh / np.mean(self.data_mesh) - 1.
    #         self.delta_mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    #         self.delta_mesh = self.delta_mesh / self.delta_mesh.mean()
    #     self.logger.info(f'Set density contrast in {time.time() - t0:.2f} seconds.')
    #     return self.delta_mesh

    def get_query_positions(self, mesh, method='randoms', nquery=None, seed=42):
        """
        Get query positions to sample the density PDF, either using random points within a mesh,
        or the positions at the center of each mesh cell.

        Parameters        
        ----------
        mesh : RealMesh
            Mesh.
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
        boxcenter = mesh.boxcenter
        boxsize = mesh.boxsize
        cellsize = mesh.cellsize
        if method == 'lattice':
            self.logger.info('Generating lattice query points within the box.')
            xedges = np.arange(boxcenter[0] - boxsize[0]/2, boxcenter[0] + boxsize[0]/2 + cellsize[0], cellsize[0])
            yedges = np.arange(boxcenter[1] - boxsize[1]/2, boxcenter[1] + boxsize[1]/2 + cellsize[1], cellsize[1])
            zedges = np.arange(boxcenter[2] - boxsize[2]/2, boxcenter[2] + boxsize[2]/2 + cellsize[2], cellsize[2])
            xcentres = 1/2 * (xedges[:-1] + xedges[1:])
            ycentres = 1/2 * (yedges[:-1] + yedges[1:])
            zcentres = 1/2 * (zedges[:-1] + zedges[1:])
            lattice_x, lattice_y, lattice_z = np.meshgrid(xcentres, ycentres, zcentres)
            lattice_x = lattice_x.flatten()
            lattice_y = lattice_y.flatten()
            lattice_z = lattice_z.flatten()
            return np.vstack((lattice_x, lattice_y, lattice_z)).T
        elif method == 'randoms':
            self.logger.info('Generating random query points within the box.')
            np.random.seed(seed)
            if nquery is None:
                nquery = 5 * self._size_data
            return np.random.rand(nquery, 3) * boxsize

    def kernel_gaussian(self, mattrs: MeshAttrs, smoothing_radius=10.):
        return jnp.exp(- 0.5 * sum((kk * smoothing_radius)**2 for kk in mattrs.kcoords(sparse=True)))