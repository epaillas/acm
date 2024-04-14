from pyrecon import RealMesh
from pypower import CatalogMesh
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import legendre
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



class BasePolyBinEstimator(BaseEstimator):
    """Base class for PolyBin3D.
    
    Inputs:
    - boxsize: 3D box-size (with 1 or 3 components)
    - gridsize: Number of Fourier modes per dimension (with 1 or 3 components)
    - Pk: Fiducial power spectrum monopole (including noise), in ordering {k, P_0(k)}, optional. This is used for the ideal estimators and for creating synthetic realizations for the optimal estimators. If unset, unit power will be assumed.
    - boxcenter: Center of the 3D box (with 1 or 3 components)
    - pixel_window: Which voxel window function to use. See self.pixel_windows for a list of options (including "none").
    - backend: Which backend to use to compute FFTs. Options: "fftw" [requires pyfftw].   
    - nthreads: How many CPUs to use for the FFT calculations. Default: 1.   
    - sightline: Whether to assume local or global line-of-sight. Options: "local" [relative to each pair], "global" [relative to z-axis].   
    """
    def __init__(self, boxsize, gridsize, Pk=None, boxcenter=[0,0,0], pixel_window='none', backend='fftw', nthreads=1, sightline='global'):
        
        # Load attributes
        self.backend = backend
        self.nthreads = nthreads
        self.sightline = sightline
        self.pixel_window = pixel_window
        
        # Check and load boxsize
        if type(boxsize)==float:
            self.boxsize = np.asarray([boxsize,boxsize,boxsize])
        else:
            assert len(boxsize)==3, "Need to specify x, y, z boxsize"
            self.boxsize = np.asarray(boxsize)
        
        # Check and load box center
        if type(boxcenter)==float:
            self.boxcenter = np.asarray([boxcenter,boxcenter,boxcenter])
        else:
            assert len(boxcenter)==3, "Need to specify x, y, z boxcenter"
            self.boxcenter = np.asarray(boxcenter)
        
        # Check and load gridsize
        if type(gridsize)==int:
            self.gridsize = np.asarray([gridsize,gridsize,gridsize],dtype=int)
        else:
            assert len(gridsize)==3, "Need to specify n_x, n_y, n_z gridsize"
            self.gridsize = np.asarray(gridsize,dtype=int)

        assert self.sightline in ['local','global'], "Sight-line must be 'local' or 'global'"
        if self.sightline=='global':
            assert np.abs(self.boxcenter).sum()==0., "Global sightlines should only be used for simulation data!"
        if self.sightline=='local':
            assert np.abs(self.boxcenter).sum()!=0., "Local sightlines should only be used for lightcone data!"
        
        # Derived parameters
        self.volume = self.boxsize.prod()
        self.kF = 2.*np.pi/self.boxsize
        self.kNy = self.gridsize*self.kF*0.5 # Nyquist
        
        # Load fiducial spectra
        if (np.asarray(Pk)==None).all():
            k_tmp = np.arange(np.min(self.kF)/10.,np.max(self.kNy)*2,np.min(self.kF)/10.)
            self.Pfid = np.asarray([k_tmp, 1.+0.*k_tmp])
        else:
            assert len(Pk)==2, "Pk should contain k and P_0(k) columns"
            self.Pfid = np.asarray(Pk)
        self.kmin, self.kmax = np.min(self.Pfid[0,0]), np.max(self.Pfid[0,-1])

        # Compute the real-space coordinate grid
        if self.sightline=='local': # not used else!
            offset = self.boxcenter+0.5*self.boxsize/self.gridsize
            r_arrs = [np.fft.fftshift(np.arange(-self.gridsize[i]//2,self.gridsize[i]//2))*self.boxsize[i]/self.gridsize[i]+offset[i] for i in range(3)]
            self.r_grids = np.meshgrid(*r_arrs,indexing='ij')
        print("\n# Dimensions: [%.2e, %.2e, %.2e] Mpc/h"%(self.boxsize[0],self.boxsize[1],self.boxsize[2]))
        print("# Center: [%.2e, %.2e, %.2e] Mpc/h"%(self.boxcenter[0],self.boxcenter[1],self.boxcenter[2]))
        print("# Line-of-sight: %s"%self.sightline)
        
        # Compute the Fourier-space coordinate grid
        k_arrs = [np.fft.fftshift(np.arange(-self.gridsize[i]//2,self.gridsize[i]//2))*self.kF[i] for i in range(3)]
        self.k_grids = np.meshgrid(*k_arrs,indexing='ij')
        self.modk_grid = np.sqrt(self.k_grids[0]**2+self.k_grids[1]**2+self.k_grids[2]**2)
        
        print("# Fourier-space grid: [%d, %d, %d]"%(self.gridsize[0],self.gridsize[1],self.gridsize[2])) 
        print("# Fundamental frequency: [%.3f, %.3f, %.3f] h/Mpc"%(self.kF[0],self.kF[1],self.kF[2]))
        print("# Nyquist frequency: [%.3f, %.3f, %.3f] h/Mpc"%(self.kNy[0],self.kNy[1],self.kNy[2]))

        # Account for pixel window if necessary
        self.pixel_windows = ['none','cic','tsc','pcs','interlaced-cic','interlaced-tsc','interlaced-pcs']
        assert self.pixel_window in self.pixel_windows, "Unknown pixel window '%s' supplied!"%pixel_window
        print("# Pixel window: %s"%self.pixel_window)
        if self.pixel_window!='none':
            def _pixel_window_1d(k):
                """Pixel window functions including first-order alias corrections (copied from nbodykit)"""
                if self.pixel_window == 'cic':
                    s = np.sin(k)**2.
                    v = (1.-2./3.*s)**0.5
                elif self.pixel_window == 'tsc':
                    s = np.sin(k)**2.
                    v = (1.-s+2./15*s**2.)**0.5
                elif self.pixel_window=='pcs':
                    s = np.sin(k)**2.
                    v = (1.-4./3.*s+2./5.*s**2-4./315.*s**3)**0.5
                elif self.pixel_window == 'interlaced-cic':
                    s = np.sinc(k/np.pi)
                    v = np.ones_like(s)
                    v[s!=0.] = s[s!=0.]**2.
                elif self.pixel_window == 'interlaced-tsc':
                    s = np.sinc(k/np.pi)
                    v = s**3.
                elif self.pixel_window == 'interlaced-pcs':
                    s = np.sinc(k/np.pi)
                    v = s**4.
                else:
                    raise Exception("Unkown window function '%s' supplied!"%self.pixel_window)
                return v

            windows_1d = [_pixel_window_1d(np.pi/self.gridsize[i]*k_arrs[i]/self.kF[i]) for i in range(3)]
            self.pixel_window_grid = np.asarray(np.meshgrid(*windows_1d,indexing='ij')).prod(axis=0)
        
        # Define angles polynomials [for generating grids and weighting]
        if self.sightline=='global':
            sight_vector = np.asarray([0.,0.,1.])
        else:
            sight_vector = self.boxcenter/np.sqrt(np.sum(self.boxcenter**2))
        self.muk_grid = np.zeros(self.gridsize)
        self.muk_grid[self.modk_grid!=0] = np.sum(sight_vector[:,None]*np.asarray(self.k_grids)[:,self.modk_grid!=0],axis=0)/self.modk_grid[self.modk_grid!=0]
        
        # Set up relevant FFT modules
        if self.backend=='fftw':
            import pyfftw

            # Set-up FFT arrays
            self.fftw_in  = pyfftw.empty_aligned(self.gridsize,dtype='complex128')
            self.fftw_out = pyfftw.empty_aligned(self.gridsize,dtype='complex128')

            # plan FFTW
            self.fftw_plan = pyfftw.FFTW(self.fftw_in, self.fftw_out, axes=(0,1,2),flags=('FFTW_ESTIMATE',),direction='FFTW_FORWARD', threads=self.nthreads)
            self.ifftw_plan = pyfftw.FFTW(self.fftw_in, self.fftw_out, axes=(0,1,2),flags=('FFTW_ESTIMATE',),direction='FFTW_BACKWARD', threads=self.nthreads)
            self.np = np
            print('# Using fftw backend')
        elif self.backend=='jax':
            import jax
            import jax.numpy as jnp
            jax.config.update("jax_enable_x64", False)
            self.np = jnp
            self.jax = jax
            print('# Using jax backend')
            
        else:
            raise Exception("Only 'fftw' and 'jax' backends are currently implemented!")
        
        # Apply Pk to grid
        self.Pk0_grid = interp1d(self.Pfid[0], self.Pfid[1], bounds_error=False)(self.modk_grid)
        
        # Invert Pk
        self.invPk0_grid = np.zeros(self.gridsize)
        good_filter = (self.Pk0_grid!=0)&np.isfinite(self.Pk0_grid)
        self.invPk0_grid[good_filter] = 1./self.Pk0_grid[good_filter] 
        self.Pk0_grid[~np.isfinite(self.Pk0_grid)] = 0.
        
        # Counter for FFTs
        self.n_FFTs_forward = 0
        self.n_FFTs_reverse = 0
        
    # Basic FFTs
    def to_fourier(self, _input_rmap):
        """Transform from real- to Fourier-space."""
        self.n_FFTs_forward += 1

        if self.backend=='fftw':
            # Load-in grid
            self.fftw_in[:] = _input_rmap
            # Perform FFT
            self.fftw_plan(self.fftw_in,self.fftw_out)
            # Return output
            return self.fftw_out.copy()

        elif self.backend=='jax':
            arr = self.np.fft.fftn(_input_rmap,axes=(-3,-2,-1))
            return arr
        
    def to_real(self, _input_kmap):
        """Transform from Fourier- to real-space."""
        self.n_FFTs_reverse += 1

        if self.backend=='fftw':
            # Load-in grid
            self.fftw_in[:] = _input_kmap
            # Perform FFT
            self.ifftw_plan(self.fftw_in,self.fftw_out)
            # Return output
            return self.fftw_out.copy()

        elif self.backend=='jax':
            arr = self.np.fft.ifftn(_input_kmap,axes=(-3,-2,-1))
            return arr

    # Spherical harmonic functions
    def _safe_divide(self, x, y):
        """Divide two arrays, replacing any NaN values with zero."""
        out = np.zeros_like(y)
        out[y!=0] = x[y!=0]/y[y!=0]
        return out

    def _compute_real_harmonics(self, coordinates, lmax, odd_l=False):
        """Compute the real spherical harmonics on the coordinate grid. These are hard-coded for speed up to lmax = 4.

        Note that we drop a factor of Sqrt[(2l+1)/4pi] for convenience, and compute odd harmonics only if odd_l=True"""
        assert lmax>=1

        # Define coordinates (with unit norm)
        norm = np.sqrt(coordinates[0]**2+coordinates[1]**2+coordinates[2]**2)
        xh = self._safe_divide(coordinates[0], norm)
        yh = self._safe_divide(coordinates[1], norm)
        zh = self._safe_divide(coordinates[2], norm)

        Ylms = {}
        if (odd_l and lmax>=1):
            Ylms[1] = np.asarray([yh,
                                    zh,
                                    xh],dtype=np.float64)
        if lmax>=2:
            Ylms[2] = np.asarray([6.*xh*yh*np.sqrt(1./12.),
                                    3.*yh*zh*np.sqrt(1./3.),
                                    (zh**2-xh**2/2.-yh**2/2.),
                                    3.*xh*zh*np.sqrt(1./3.),
                                    (3*xh**2-3*yh**2)*np.sqrt(1./12.)],dtype=np.float64)
        if (odd_l and lmax>=3):
            Ylms[3] = np.asarray([(45.*xh**2*yh-15.*yh**3.)*np.sqrt(1./360.),
                                    (30.*xh*yh*zh)*np.sqrt(1./60.),
                                    (-1.5*xh**2*yh-1.5*yh**3+6.*yh*zh**2)*np.sqrt(1./6.),
                                    (-1.5*xh**2*zh-1.5*yh**2*zh+zh**3),
                                    (-1.5*xh**3-1.5*xh*yh**2+6.*xh*zh**2)*np.sqrt(1./6.),
                                    (15.*xh**2*zh-15.*yh**2*zh)*np.sqrt(1./60.),
                                    (15.*xh**3-45.*xh*yh**2)*np.sqrt(1./360.)],dtype=np.float64)
        if lmax>=4:
            Ylms[4] = np.asarray([(420.*xh**3*yh-420.*xh*yh**3)*np.sqrt(1./20160.),
                                    (315.*xh**2*yh*zh-105.*yh**3*zh)*np.sqrt(1./2520.),
                                    (-15.*xh**3*yh-15.*xh*yh**3+90.*xh*yh*zh**2)*np.sqrt(1./180.),
                                    (-7.5*xh**2*yh*zh-7.5*yh**3*zh+10.*yh*zh**3)*np.sqrt(1./10.),
                                    35./8.*zh**4.-15./4.*zh**2.+3./8.,
                                    (-15./2.*xh**3*zh-15.*xh*yh**2*zh/2.+10.*xh*zh**3)*np.sqrt(1./10.),
                                    (-15./2.*xh**4+45.*xh**2*zh**2+15./2.*yh**4.-45.*yh**2.*zh**2)*np.sqrt(1./180.),
                                    (105.*xh**3*zh-315.*xh*yh**2*zh)*np.sqrt(1./2520.),
                                    (105.*xh**4-630.*xh**2*yh**2+105.*yh**4)*np.sqrt(1./20160.)],dtype=np.float64)
        return Ylms

    # Gaussian random field routines
    def generate_data(self, seed=None, Pk_input=[], output_type='real', include_pixel_window=True):
        """
        Generate a Gaussian periodic map with a given set of P(k) multipoles (or the fiducial power spectrum monopole, if unspecified). 
        The input Pk are expected to by in the form {k, P_0, [P_2], [P_4]}, where P_2, P_4 are optional.

        No mask is added at this stage (except for a pixelation window, unless include_pixel_window=False), and the output can be in real- or Fourier-space.
        """
        assert output_type in ['fourier','real'], "Valid output types are 'fourier' and 'real' only!"
        
        # Define seed
        if seed!=None:
            np.random.seed(seed)

        # Define input power spectrum on the k-space grid
        if len(Pk_input)==0:
            Pk_grid = self.Pk0_grid
        else:
            assert len(np.asarray(Pk_input).shape)==2, "Pk should contain k and P_0, (and optionally P_2, P_4) columns"
            assert len(Pk_input) in [2,3,4], "Pk should contain k and P_0, (and optionally P_2, P_4) columns"
            
            Pk_grid = interp1d(Pk_input[0], Pk_input[1], bounds_error=False, fill_value=0.)(self.modk_grid)
            if len(Pk_input)>2:
                Pk_grid += legendre(2)(self.muk_grid)*interp1d(Pk_input[0], Pk_input[2], bounds_error=False, fill_value=0.)(self.modk_grid)
            if len(Pk_input)>3:
                Pk_grid += legendre(4)(self.muk_grid)*interp1d(Pk_input[0], Pk_input[3], bounds_error=False, fill_value=0.)(self.modk_grid)
        
        # Generate random Gaussian maps with input P(k)
        rand_fourier = (np.random.randn(*self.gridsize)+1.0j*np.random.randn(*self.gridsize))*np.sqrt(Pk_grid)
        rand_fourier[self.modk_grid==0] = 0.

        # Add pixel window function to delta(k)
        if self.pixel_window!='none' and include_pixel_window:
            rand_fourier *= self.pixel_window_grid
        
        # Force map to be real and normalize
        rand_real = self.to_real(rand_fourier).real
        rand_real *= self.gridsize.prod()/np.sqrt(self.volume)

        if output_type=='real': return rand_real
        else: return self.to_fourier(rand_real)
        
    def applyAinv(self, input_data, input_type='real', output_type='real'):
        """Apply the exact inverse weighting A^{-1} to a map. This assumes a diagonal-in-ell C_l^{XY} weighting, as produced by generate_data.
        
        Note that the code has two input and output options: "fourier" or "real", to avoid unnecessary transforms.
        """
    
        assert input_type in ['fourier','real'], "Valid input types are 'fourier' and 'real' only!"
        assert output_type in ['fourier','real'], "Valid output types are 'fourier' and 'real' only!"

        # Transform to harmonic space, if necessary
        if input_type=='real': input_fourier = self.to_fourier(input_data)
        else: input_fourier = input_data.copy()
            
        # Divide by covariance
        Sinv_fourier = input_fourier*self.invPk0_grid
        
        # Optionally divide by pixel window
        if self.pixel_window!='none':
            Sinv_fourier /= self.pixel_window_grid**2

        # Optionally return to map-space
        if output_type=='real': return self.to_real(Sinv_fourier)
        else: return Sinv_fourier