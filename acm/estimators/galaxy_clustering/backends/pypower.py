from pypower import CatalogMesh
import numpy as np
import jax.numpy as jnp
import time
import logging


class PypowerBackend:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger('PypowerBackend')
        self.name = 'pypower'
        if 'meshsize' in kwargs:
            kwargs['nmesh'] = kwargs.pop('meshsize')
        self.mesh = CatalogMesh(**kwargs, interlacing=0, resampler='cic', position_type='pos')
        
        self.meshsize = self.mesh.nmesh
        self.cellsize = self.mesh.boxsize / self.mesh.nmesh
        self.boxsize = self.mesh.boxsize
        self.boxcenter = self.mesh.boxcenter
        self.logger.info(f'Box size: {self.boxsize}')
        self.logger.info(f'Box center: {self.boxcenter}')
        self.logger.info(f'Box meshsize: {self.meshsize}')

    @property
    def has_randoms(self):
        return self.mesh.with_randoms

    def set_density_contrast(self, smoothing_radius=None, compensate=False, filter_shape='Gaussian'):
        """
        Set the density contrast.

        Parameters
        ----------
        smoothing_radius : float, optional
            Smoothing radius.
        check : bool, optional
            Check if there are enough randoms.
        ran_min : float, optional
            Minimum randoms.
        nquery_factor : int, optional
            Factor to multiply the number of data points to get the number of query points.
            
        Returns
        -------
        delta_mesh : array_like
            Density contrast.
        """
        t0 = time.time()
        data_mesh = self.mesh.to_mesh(field='data', compensate=compensate)
        if smoothing_radius:
            data_mesh = data_mesh.r2c().apply(
            getattr(self, filter_shape)(r=smoothing_radius))
            data_mesh = data_mesh.c2r()
        if self.has_randoms:
            randoms_mesh = self.mesh.to_mesh(field='data-normalized_randoms',
                compensate=compensate)
            randoms_mesh = randoms_mesh.r2c().apply(
                getattr(self, filter_shape)(r=smoothing_radius))
            randoms_mesh = randoms_mesh.c2r()
            sum_data, sum_randoms = np.sum(data_mesh.value), np.sum(randoms_mesh.value)
            alpha = sum_data / sum_randoms
            delta_mesh = data_mesh - alpha * randoms_mesh
            mask = randoms_mesh > 0
            delta_mesh[mask] /= alpha * randoms_mesh[mask]
            delta_mesh[~mask] = 0.0
            shift = self.mesh.boxsize / 2 - self.mesh.boxcenter
        else:
            sum_data = np.sum(data_mesh)
            print(np.min(data_mesh), np.max(data_mesh), np.mean(data_mesh))
            delta_mesh = data_mesh/np.mean(data_mesh) - 1
        self.data_mesh = data_mesh
        self.size_data = int(sum_data)
        self.delta_mesh = delta_mesh
        self.logger.info(f'Set density contrast in {time.time() - t0:.2f} seconds.')
        return self.delta_mesh

    def get_query_positions(self, method='randoms', nquery=None, seed=42):
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
        boxcenter = self.boxcenter
        boxsize = self.boxsize
        cellsize = self.cellsize
        if method == 'lattice':
            self.logger.info('Generating lattice query points within the box.')
            xedges = np.arange(boxcenter[0] - boxsize[0]/2 - cellsize[0]/2, boxcenter[0] + boxsize[0]/2, cellsize[0])
            yedges = np.arange(boxcenter[1] - boxsize[1]/2 - cellsize[1]/2, boxcenter[1] + boxsize[1]/2, cellsize[1])
            zedges = np.arange(boxcenter[2] - boxsize[2]/2 - cellsize[2]/2, boxcenter[2] + boxsize[2]/2, cellsize[2])
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
                nquery = 5 * self.size_data
            return np.random.rand(nquery, 3) * boxsize + (boxcenter - boxsize / 2)

    class TopHat(object):
        '''Top-hat filter in Fourier space
        adapted from https://github.com/bccp/nbodykit/

        Parameters
        ----------
        r : float
            the radius of the top-hat filter
        '''
        def __init__(self, r):
            self.r = r

        def __call__(self, k, v):
            r = self.r
            k = sum(ki ** 2 for ki in k) ** 0.5
            kr = k * r
            with np.errstate(divide='ignore', invalid='ignore'):
                w = 3 * (np.sin(kr) / kr ** 3 - np.cos(kr) / kr ** 2)
            w[k == 0] = 1.0
            return w * v


    class Gaussian(object):
        '''Gaussian filter in Fourier space

        Parameters
        ----------
        r : float
            the radius of the Gaussian filter
        '''
        def __init__(self, r):
            self.r = r

        def __call__(self, k, v):
            r = self.r
            k2 = sum(ki ** 2 for ki in k)
            return np.exp(- 0.5 * k2 * r**2) * v