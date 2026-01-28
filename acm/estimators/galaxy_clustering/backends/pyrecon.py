from pyrecon import RealMesh
import numpy as np
import time
import logging


class PyreconBackend:
    def __init__(self, data_positions, data_weights=None, randoms_positions=None, randoms_weights=None, **kwargs):
        self.logger = logging.getLogger('PyreconBackend')
        self.name = 'pyrecon'
        
        # Extract mesh parameters
        boxsize = kwargs.get('boxsize', None)
        boxcenter = kwargs.get('boxcenter', 0.0)
        meshsize = kwargs.get('meshsize', None)
        
        if boxsize is None:
            raise ValueError('boxsize must be provided for pyrecon backend')
        if meshsize is None:
            raise ValueError('meshsize must be provided for pyrecon backend')
        
        # Convert to array format
        if np.isscalar(boxsize):
            boxsize = np.array([boxsize, boxsize, boxsize])
        else:
            boxsize = np.asarray(boxsize)
            
        if np.isscalar(boxcenter):
            boxcenter = np.array([boxcenter, boxcenter, boxcenter])
        else:
            boxcenter = np.asarray(boxcenter)
            
        if np.isscalar(meshsize):
            meshsize = np.array([meshsize, meshsize, meshsize], dtype=int)
        else:
            meshsize = np.asarray(meshsize, dtype=int)
        
        # Store mesh attributes
        self.boxsize = boxsize
        self.boxcenter = boxcenter
        self.meshsize = meshsize
        self.cellsize = boxsize / meshsize
        
        # Initialize meshes
        self.data_mesh = RealMesh(boxsize=boxsize, boxcenter=boxcenter, nmesh=meshsize)
        self.randoms_mesh = RealMesh(boxsize=boxsize, boxcenter=boxcenter, nmesh=meshsize) if randoms_positions is not None else None
        
        # Assign data and randoms
        self.size_data = 0
        self._size_randoms = 0
        
        if data_positions is not None:
            self.assign_data(data_positions, weights=data_weights)
        
        if randoms_positions is not None:
            self.assign_randoms(randoms_positions, weights=randoms_weights)
        
        self.logger.info(f'Box size: {self.boxsize}')
        self.logger.info(f'Box center: {self.boxcenter}')
        self.logger.info(f'Box meshsize: {self.meshsize}')

    def assign_data(self, positions, weights=None, wrap=True, clear_previous=True):
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
            self.size_data = 0
        self.data_mesh.assign_cic(positions=positions, weights=weights, wrap=wrap)
        self.size_data += len(positions)

    def assign_randoms(self, positions, weights=None, wrap=True):
        """
        Assign randoms to the mesh.

        Parameters
        ----------
        positions : array_like
            Positions of the random points.
        weights : array_like, optional
            Weights of the random points. If not provided, all points are 
            assumed to have the same weight.
        wrap : bool, optional
            Wrap the random points around the box, assuming periodic boundaries.
        """
        if self.randoms_mesh is None:
            self.randoms_mesh = RealMesh(boxsize=self.boxsize, boxcenter=self.boxcenter, nmesh=self.meshsize)
        if self.randoms_mesh.value is None:
            self._size_randoms = 0
        self.randoms_mesh.assign_cic(positions=positions, weights=weights, wrap=wrap)
        self._size_randoms += len(positions)

    @property
    def has_randoms(self):
        return self.randoms_mesh is not None and self.randoms_mesh.value is not None

    def set_density_contrast(self, smoothing_radius=None, check=False, ran_min=0.01, save_wisdom=False):
        """
        Set the density contrast.

        Parameters
        ----------
        smoothing_radius : float, optional
            Smoothing radius in Mpc/h.
        check : bool, optional
            Check if there are enough randoms.
        ran_min : float, optional
            Minimum randoms threshold (as fraction of mean).
        save_wisdom : bool, optional
            Save FFTW wisdom for future use.
            
        Returns
        -------
        delta_mesh : RealMesh
            Density contrast.
        """
        t0 = time.time()
        
        if smoothing_radius:
            self.logger.info(f'Smoothing with {smoothing_radius} Mpc/h Gaussian kernel.')
            self.data_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=save_wisdom)
        
        if self.has_randoms:
            if check:
                mask_nonzero = self.randoms_mesh.value > 0.
                nnonzero = mask_nonzero.sum()
                if nnonzero < 2: 
                    raise ValueError('Very few randoms.')
            
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
            self.mean = np.mean(self.data_mesh)
            self.delta_mesh = self.data_mesh / self.mean - 1.
        
        self.logger.info(f'Set density contrast in {time.time() - t0:.2f} seconds.')
        return self.delta_mesh

    def get_query_positions(self, method='randoms', nquery=None, seed=42):
        """
        Get query positions to sample the density PDF, either using random points within a mesh,
        or the positions at the center of each mesh cell.

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
