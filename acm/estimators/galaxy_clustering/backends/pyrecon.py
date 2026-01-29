from pyrecon import RealMesh
import numpy as np
import time
import logging


class PyreconBackend:
    """Backend using pyrecon for galaxy clustering measurements.
    
    This backend uses the pyrecon package to create mesh fields from galaxy
    catalogs and compute density contrasts. It supports both data-only and
    data+randoms configurations and provides methods for assigning particles
    to meshes incrementally.
    
    Attributes
    ----------
    name : str
        Backend name identifier ('pyrecon').
    boxsize : ndarray
        Size of the simulation box in each dimension.
    boxcenter : ndarray
        Center coordinates of the box.
    meshsize : ndarray
        Number of mesh cells in each dimension.
    cellsize : ndarray
        Size of each mesh cell.
    data_mesh : RealMesh
        Pyrecon mesh object for data.
    randoms_mesh : RealMesh or None
        Pyrecon mesh object for randoms, if provided.
    size_data : int
        Number of data points assigned to the mesh.
    delta_mesh : RealMesh
        Density contrast field (set by set_density_contrast).
    ran_min : float
        Minimum randoms threshold value (set by set_density_contrast).
    """
    def __init__(self, data_positions, data_weights=None, randoms_positions=None, randoms_weights=None, **kwargs):
        """Initialize the pyrecon backend.
        
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
            Additional keyword arguments for mesh configuration.
            Required:
            - boxsize : float or array_like
                Size of the box.
            - meshsize : int or array_like
                Number of mesh cells per dimension.
            Optional:
            - boxcenter : float or array_like, default=0.0
                Center of the box.
                
        Raises
        ------
        ValueError
            If boxsize or meshsize are not provided.
        """
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
        """Assign data particles to the mesh.
        
        Uses Cloud-in-Cell (CIC) interpolation to paint particles onto the mesh.

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            Positions of the data points.
        weights : array_like, shape (N,), optional
            Weights of the data points. If not provided, all points are 
            assumed to have unit weight.
        wrap : bool, default=True
            Wrap the data points around the box, assuming periodic boundaries.
        clear_previous : bool, default=True
            Clear previous data before assignment. If False, particles are
            added to existing mesh values.
        """
        if clear_previous:
            self.data_mesh.value = None
        if self.data_mesh.value is None:
            self.size_data = 0
        self.data_mesh.assign_cic(positions=positions, weights=weights, wrap=wrap)
        self.size_data += len(positions)

    def assign_randoms(self, positions, weights=None, wrap=True):
        """Assign random particles to the mesh.
        
        Uses Cloud-in-Cell (CIC) interpolation to paint random particles onto
        the randoms mesh.

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            Positions of the random points.
        weights : array_like, shape (N,), optional
            Weights of the random points. If not provided, all points are 
            assumed to have unit weight.
        wrap : bool, default=True
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
        """Check if the backend has randoms.
        
        Returns
        -------
        bool
            True if random catalog is present and has been assigned to mesh,
            False otherwise.
        """
        return self.randoms_mesh is not None and self.randoms_mesh.value is not None

    def set_density_contrast(self, smoothing_radius=None, check=False, ran_min=0.01, save_wisdom=False):
        """Compute the density contrast field.
        
        Computes the density contrast using data mesh and optionally randoms
        mesh (FKP method). Optionally applies Gaussian smoothing using FFTW.

        Parameters
        ----------
        smoothing_radius : float, optional
            Gaussian smoothing scale in Mpc/h. If None, no smoothing is applied.
        check : bool, default=False
            Check if there are enough randoms in the mesh to avoid
            numerical issues.
        ran_min : float, default=0.01
            Minimum randoms threshold as fraction of mean randoms density.
            Cells with randoms below this threshold are set to zero.
        save_wisdom : bool, default=False
            Save FFTW wisdom to disk for faster future FFTs.
            
        Returns
        -------
        delta_mesh : RealMesh
            Density contrast field.
            
        Raises
        ------
        ValueError
            If check=True and very few randoms are found.
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
