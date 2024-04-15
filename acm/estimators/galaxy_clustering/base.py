from pyrecon import RealMesh
from pypower import CatalogMesh
from PolyBin3D import PolyBin3D
import numpy as np
import time


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
    def __init__(self, **kwargs):
        super().__init__()
        self.data_mesh = RealMesh(**kwargs)
        self.randoms_mesh = RealMesh(**kwargs)
        self.logger.info(f'Box size: {self.data_mesh.boxsize}')
        self.logger.info(f'Box center: {self.data_mesh.boxcenter}')
        self.logger.info(f'Box nmesh: {self.data_mesh.nmesh}')

    def assign_data(self, positions, weights=None, wrap=False, clear_previous=True):
        """
        Assign data to the mesh.

        Parameters
        ----------
        positions : array_like
            Positions of the data points.
        weights : array_like, optional
            Weights of the data points. If not provided, all points are 
            assumed to have the same weight.
        wrap : bool, optional
            Wrap the data points around the box, assuming periodic boundaries.
        clear_previous : bool, optional
            Clear previous data.
        """
        if clear_previous:
            self.data_mesh.value = None
        if self.data_mesh.value is None:
            self._size_data = 0
        self.data_mesh.assign_cic(positions=positions, weights=weights, wrap=wrap)
        self._size_data += len(positions)

    def assign_randoms(self, positions, weights=None):
        """
        Assign randoms to the mesh.

        Parameters
        ----------
        positions : array_like
            Positions of the random points.
        weights : array_like, optional
            Weights of the random points. If not provided, all points are 
            assumed to have the same weight.
        """
        if self.randoms_mesh.value is None:
            self._size_randoms = 0
        self.randoms_mesh.assign_cic(positions=positions, weights=weights)
        self._size_randoms += len(positions)

    @property
    def has_randoms(self):
        return self.randoms_mesh.value is not None

    def set_density_contrast(self, smoothing_radius=None, check=False, ran_min=0.01, save_wisdom=False):
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
        if smoothing_radius:
            self.data_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=save_wisdom,)
        if self.has_randoms:
            if check:
                mask_nonzero = self.randoms_mesh.value > 0.
                nnonzero = mask_nonzero.sum()
                if nnonzero < 2: raise ValueError('Very few randoms.')
            if smoothing_radius:
                self.randoms_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=save_wisdom)
            sum_data, sum_randoms = np.sum(self.data_mesh.value), np.sum(self.randoms_mesh.value)
            alpha = sum_data * 1. / sum_randoms
            self.delta_mesh = self.data_mesh - alpha * self.randoms_mesh
            self.ran_min = ran_min * sum_randoms / self._size_randoms
            mask = self.randoms_mesh > self.ran_min
            self.delta_mesh[mask] /= alpha * self.randoms_mesh[mask]
            self.delta_mesh[~mask] = 0.0
        else:
            self.delta_mesh = self.data_mesh / np.mean(self.data_mesh) - 1.
        self.logger.info(f'Set density contrast in {time.time() - t0:.2f} seconds.')
        return self.delta_mesh

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


class BaseCatalogMeshEstimator(BaseEstimator):
    def __init__(self, **kwargs):
        self.mesh = CatalogMesh(**kwargs)
        self.logger.info(f'Box size: {self.mesh.boxsize}')
        self.logger.info(f'Box center: {self.mesh.boxcenter}')
        self.logger.info(f'Box nmesh: {self.mesh.nmesh}')

    @property
    def has_randoms(self):
        return self.mesh.with_randoms

    def set_density_contrast(self, smoothing_radius=None, compensate=True, filter_shape='Gaussian'):
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
        self.logger.info('Setting density contrast.')
        data_mesh = self.mesh.to_mesh(field='data', compensate=compensate)
        if smoothing_radius:
            print('smoothing')
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
            nmesh = self.mesh.nmesh
            sum_data = np.sum(data_mesh)
            delta_mesh = data_mesh/np.mean(data_mesh) - 1
            shift = 0
        self.data_mesh = data_mesh
        self._size_data = int(sum_data)
        self.delta_mesh = delta_mesh
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
        boxcenter = self.mesh.boxcenter
        boxsize = self.mesh.boxsize
        cellsize = self.mesh.boxsize / self.mesh.nmesh
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


class BasePolyBinEstimator(PolyBin3D):
    """
    Base class for PolyBin. Inherits from the PolyBin3D code developed by Oliver Philcox & Thomas Flöss
    (https://github.com/oliverphilcox/PolyBin3D).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)