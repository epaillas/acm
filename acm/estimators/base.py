import logging
from pyrecon import RealMesh
import numpy as np


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

    def set_density_contrast(self, smoothing_radius=None, check=False, ran_min=0.01):
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
        if smoothing_radius:
            self.data_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True,)
        if self.has_randoms:
            if check:
                mask_nonzero = self.randoms_mesh.value > 0.
                nnonzero = mask_nonzero.sum()
                if nnonzero < 2: raise ValueError('Very few randoms.')
            if smoothing_radius:
                self.randoms_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True)
            sum_data, sum_randoms = np.sum(self.data_mesh.value), np.sum(self.randoms_mesh.value)
            alpha = sum_data * 1. / sum_randoms
            self.delta_mesh = self.data_mesh - alpha * self.randoms_mesh
            self.ran_min = ran_min * sum_randoms / self._size_randoms
            mask = self.randoms_mesh > self.ran_min
            self.delta_mesh[mask] /= alpha * self.randoms_mesh[mask]
            self.delta_mesh[~mask] = 0.0
        else:
            self.delta_mesh = self.data_mesh / np.mean(self.data_mesh) - 1.
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