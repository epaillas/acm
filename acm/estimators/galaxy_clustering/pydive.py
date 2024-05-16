from .base import BaseEstimator
from .src.pydive import get_void_catalog_full, get_void_catalog_cgal
import logging
import numpy as np
import time
import pandas as pd

def _default_sample_function(void_cat, column = 'R'):
    limits = np.percentile(void_cat[column], np.linspace(0, 100, 7))
    toret = []
    for i in range(len(limits)-1):
        mask = (void_cat[column] > limits[i]) & (void_cat[column] < limits[i+1])
        try:
            toret.append(void_cat.loc[mask,['x', 'y', 'z', 'R', 'dtfe', 'sphericity']].values)
        except KeyError:
            toret.append(void_cat.loc[mask,['x', 'y', 'z', 'R']].values)
    return toret

class DTVoid(BaseEstimator):
    """
    Class to compute Delaunay Triangulation (DT) Sphere clustering, as in https://arxiv.org/abs/1511.04299.
    """
    def __init__(self, **kwargs):
        self.logger = logging.getLogger('DTVoid')
        self.logger.info('Initializing DTVoid.')
        self.cosmo = kwargs.pop('cosmo', None)
        self.zrange = kwargs.pop('zrange', None)
        self.boxmin = kwargs.pop('boxmin', [0,0,0])
        self.boxsize = kwargs.pop('boxsize', None)
        self.data_type = kwargs.pop('data_type', 'xyz')
        self.is_box = self.boxsize is not None
        self.void_randoms = kwargs.pop('void_randoms', None)
        self.has_randoms = self.void_randoms is not None
        #if not self.is_box:
        #    assert self.cosmo is not None, "Cosmology (cosmo) required for lightcone computation"
        #    assert self.zrange is not None, "Redshift range (zrange) required for lightcone computation"
        #    from scipy.interpolate import interp1d
        #    zmin, zmax = self.zrange
        #    self.cosmo_cache = dict(r = self.cosmo.comoving_radial_distance(np.linspace(zmin, zmax, 10000)), z = np.linspace(zmin, zmax, 10000))
        #    self.z_i = interp1d(self.cosmo_cache['r'], self.cosmo_cache['z'], kind='linear', fill_value="extrapolate")
        #    from astropy.table import Table, vstack
        #    self.tiles = Table.read('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/tiles-DARK.fits')
        
        super().__init__(**kwargs)
        
            
        

    def _galcat_to_voidcat(self, data_positions,
                           full_cat = True, 
                           periodic_mode = 0,
                           ):
        
        
        periodic = self.is_box
        tic = time.time()
        if full_cat: periodic_mode = 0
        if periodic_mode == 0 and self.is_box:
            ngal = data_positions.shape[0] / np.prod(self.boxsize)
            free_path = ngal**(-1/3)
            cpy_range = 3.5 * free_path
        else:
            self.logger.info(f"No copy_range required with natively periodic DT.")
            cpy_range = 0
        max_r = np.inf
        while max_r > cpy_range:
            cpy_range *= 1.1
            if full_cat:
                voids, gal_dtfe = get_void_catalog_full(data_positions, 
                                                        periodic = periodic,
                                                        box_min = self.boxmin,
                                                        box_max = self.boxsize,
                                                        cpy_range = cpy_range,
                                                        )
            else:
                voids = get_void_catalog_cgal(data_positions, 
                            periodic=periodic, 
                            periodic_mode = periodic_mode,
                            box_min = self.boxmin,
                            box_max = self.boxsize,
                            cpy_range = cpy_range,
                            )
                gal_dtfe = None
            if periodic:
                box_mask = ((voids[:,:3] > 0) & (voids[:,:3] < np.asarray(self.boxsize))).all(axis = 1)
                voids = voids[box_mask]
                max_r = voids[:,3].max()
                self.logger.info(f"Biggest void is of size {max_r} Mpc/h")
                if max_r > cpy_range and periodic_mode != 0:                   
                    self.logger.info(f"Rerunning void finder since largest void was larger than periodic padding.")
            else:
                 max_r = 0   
        if self.full_catalog:
            voids = pd.DataFrame(voids, columns = ['x', 'y', 'z', 'R', 'vol', 'dtfe', 'area'])
            voids['sphericity'] = np.pi**(1/3) * (6 * voids['vol'])**(2/3) / voids['area']
        else:
            voids = pd.DataFrame(voids, columns = ['x', 'y', 'z', 'R'])
        
        
        self.logger.info(f"Got DT spheres in total time {time.time() - tic} s")
        return voids, gal_dtfe
    


    def compute_spheres(self, data_positions, full_catalog,  sample_function = _default_sample_function, sample_function_kwargs = {}):
        """
        Get the samples of the overdensity density field.

        Parameters
        ----------
        query_positions : array_like, optional
            Query positions.
        query_method : str, optional
            Method to generate query points. Options are 'lattice' or 'randoms'.
        nquery_factor : int, optional
            Factor to multiply the number of data points to get the number of query points.
        nsamples : int
            Number of samples.

        Returns
        -------
        samples : array_like
            Quantiles of the density field.
        samples_idx : array_like, optional
            Index of the sample of each query point.
        delta_query : array_like, optional
            Density contrast at the query points.
        """
        self.full_catalog = full_catalog
        void_cat, gal_dtfe = self._galcat_to_voidcat(data_positions,
                                                    full_cat = full_catalog, 
                                                    periodic_mode = 0,
                                                    )        
        self.samples = sample_function(void_cat, **sample_function_kwargs)
        return self.samples

    def sample_data_correlation(self, data_positions, **kwargs):
        """
        Compute the cross-correlation function between the density field
        samples and the data.

        Parameters
        ----------
        data_positions : array_like
            Positions of the data.
        kwargs : dict
            Additional arguments for pycorr.TwoPointCorrelationFunction.

        Returns
        -------
        s : array_like
            Pair separations.
        sample_data_ccf : array_like
            Cross-correlation function between samples and data.
        """
        from pycorr import TwoPointCorrelationFunction
        nsplits = kwargs.pop('nsplits', 1)
        self.logger.info(f"Using randoms split into {nsplits} parts.")
        if self.has_randoms:
            if 'randoms_positions' not in kwargs:
                raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
            randoms_positions = np.array_split(kwargs.pop('randoms_positions'), nsplits, axis = 0)
            if 'data_weights' in kwargs:
                kwargs['data_weights2'] = kwargs.pop('data_weights')
            
            randoms_weights = kwargs.pop('randoms_weights', [None] * nsplits)
            if randoms_weights[0] is not None:
                randoms_weights = np.array_split(randoms_weights, nsplits, axis = 0)
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.boxsize
            randoms_weights = randoms_positions = [None] * nsplits
        self._sample_data_correlation = []
        
        for i, sample in enumerate(self.samples):
            if self.has_randoms:
                split_rands = np.array_split(self.void_randoms[i][:,:3], nsplits, axis = 0)
            R1R2 = None
            result = 0
            for j in range(nsplits):
                result += TwoPointCorrelationFunction(
                            data_positions1=sample[:,:3],
                            data_positions2=data_positions,
                            randoms_positions1 = None if not self.has_randoms else split_rands[j],
                            randoms_positions2 = randoms_positions[j],
                            randoms_weights2 = randoms_weights[j],
                            mode='smu',
                            position_type='pos',
                            R1R2=R1R2,
                            **kwargs,
                            )
                R1R2 = result.R1R2
            self._sample_data_correlation.append(result)
            
        return self._sample_data_correlation

    def sample_correlation(self, **kwargs):
        """
        Compute the auto-correlation function of the density field samples.

        Parameters
        ----------
        kwargs : dict
            Additional arguments for pycorr.TwoPointCorrelationFunction.

        Returns
        -------
        s : array_like
            Pair separations.
        sample_acf : array_like
            Auto-correlation function of samples.
        """
        from pycorr import TwoPointCorrelationFunction
        if self.has_randoms:
            if 'randoms_positions' not in kwargs:
                raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.boxsize
        self._sample_correlation = []
        nsplits = kwargs.pop('nsplits', 1)
        self.logger.info(f"Using randoms split into {nsplits} parts.")
        for i, sample in enumerate(self.samples):
            if self.has_randoms:
                split_rands = np.array_split(self.void_randoms[i][:,:3], nsplits)
            R1R2 = None
            result = 0
            for j in range(nsplits):
                result += TwoPointCorrelationFunction(
                    data_positions1=sample[:,:3],
                    randoms_positions1 = None if not self.has_randoms else split_rands[j],
                    mode='smu',
                    position_type='pos',
                    R1R2 = R1R2,
                    **kwargs,
                )
                R1R2 = result.R1R2
            self._sample_correlation.append(result)
        return self._sample_correlation

    def sample_data_power(self, data_positions, **kwargs):
        """
        Compute the cross-power spectrum between the data and the density field samples.

        Parameters
        ----------
        data_positions : array_like
            Positions of the data.
        kwargs : dict
            Additional arguments for pypower.CatalogFFTPower.

        Returns
        -------
        k : array_like
            Wavenumbers.
        sample_data_power : array_like
            Cross-power spectrum between samples and data.
        """
        from pypower import CatalogFFTPower
        if self.has_randoms:
            if 'randoms_positions' not in kwargs:
                raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
            kwargs['randoms_positions1'] = kwargs['randoms_positions']
            kwargs['randoms_positions2'] = kwargs['randoms_positions']
            kwargs.pop('randoms_positions')
            if 'data_weights' in kwargs:
                kwargs['data_weights2'] = kwargs.pop('data_weights')
            if 'randoms_weights' in kwargs:
                kwargs['randoms_weights2'] = kwargs.pop('randoms_weights')
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.boxsize
        
        self._sample_data_power = []
        for sample in self.samples:
            result = CatalogFFTPower(
                data_positions1=sample[:,:3],
                data_positions2=data_positions,
                ells=(0, 2, 4),
                position_type='pos',
                **kwargs,
            ).poles
            self._sample_data_power.append(result)
        return self._sample_data_power

    def sample_power(self, **kwargs):
        """
        Compute the auto-power spectrum of the density field samples.

        Parameters
        ----------
        data_positions : array_like
            Positions of the data.
        kwargs : dict
            Additional arguments for pypower.CatalogFFTPower.

        Returns
        -------
        k : array_like
            Wavenumbers.
        sample_power : array_like
            Auto-power spectrum of samples.
        """
        from pypower import CatalogFFTPower
        if self.has_randoms:
            if 'randoms_positions' not in kwargs:
                raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
            kwargs['randoms_positions1'] = kwargs['randoms_positions']
            kwargs.pop('randoms_positions')
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.boxsize
        
        self._sample_power = []
        for sample in self.samples:
            result = CatalogFFTPower(
                data_positions1=sample[:,:3],
                ells=(0, 2, 4),
                position_type='pos',
                **kwargs,
            ).poles
            self._sample_power.append(result)
        return self._sample_power

    def plot_one_point(self):
        import matplotlib.pyplot as plt
        import matplotlib
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        if self.full_catalog:
            fig, ax = plt.subplots(figsize=(12, self.samples[0][:,3:].shape[1]), nrows = 1, ncols = self.samples[0][:,3:].shape[1])
        else:
            fig, ax = plt.subplots(figsize=(4, 4), nrows = 1, ncols = 1)
            ax = [ax]
        cmap = matplotlib.cm.get_cmap('coolwarm')
        colors = cmap(np.linspace(0.01, 0.99, len(self.samples)))
        for i, samp in enumerate(self.samples):
            hist, bin_edges, patches = ax[0].hist(self.samples[i][:,3], bins=200, density=True, lw=2.0, histtype = 'step')#, color='grey')
            if self.full_catalog:
                hist, bin_edges, patches = ax[1].hist(np.log10(np.where(self.samples[i][:,4] <= 0, 1e-8, self.samples[i][:,4])), bins=200, density=True, lw=2.0, histtype = 'step')#, color='grey')
                hist, bin_edges, patches = ax[2].hist(self.samples[i][:,5], bins=200, density=True, lw=2.0, histtype = 'step')#, color='grey')
                try:
                    hist, bin_edges, patches = ax[3].hist(np.log10(np.where(self.samples[i][:,6] <= 0, 1e-8, self.samples[i][:,6])), bins=200, density=True, lw=2.0, histtype = 'step')#, color='grey')
                except:
                    pass
        
        #for i in range(len(self.samples)):
        #    dmax = self.delta_query[self.samples_idx == i].max()
        #    imax = np.digitize([dmax], bin_edges)[0] - 1
        #    for index in range(imin, imax):
        #        patches[index].set_facecolor(colors[i])
        #    imin = imax
        #    ax.plot(np.nan, np.nan, color=colors[i], label=rf'${{\rm Q}}_{i}$', lw=4.0)
        ax[0].set_xlabel(r'$R \left(\, h^{-1}{\rm Mpc}\right)$', fontsize=15)
        if self.full_catalog:
            ax[1].set_xlabel(r'$\log(\Delta)$', fontsize=15)
            ax[2].set_xlabel(r'$\Phi$', fontsize=15)
            try:
                ax[3].set_xlabel(r'$\log(\Delta / n(z))$', fontsize=15)
            except:
                pass
        [a.set_ylabel('PDF', fontsize=15) for a in ax]
        #ax.set_xlim(-1.3, 3.0)
        [a.legend(handlelength=1.0) for a in ax]
        plt.tight_layout()
        plt.show()
        return fig

    def plot_sample_data_correlation(self, ell=0, save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.samples)):
            s, multipoles = self._sample_data_correlation[i](ells=(0, 2, 4), return_sep=True)
            ax.plot(s, s**2*multipoles[ell//2], lw=2.0, color=colors[i], label=rf'${{\rm DTS}}_{i}$')
        ax.set_xlabel(r'$s\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$s^2 \xi_\ell\, [h^{-2}{\rm Mpc^2}](s)$', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_sample_correlation(self, ell=0, save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.samples)):
            s, multipoles = self._sample_correlation[i](ells=(0, 2, 4), return_sep=True)
            ax.plot(s, s**2*multipoles[ell//2], lw=2.0, label=rf'${{\rm DTS}}_{i}$')
        ax.set_xlabel(r'$s\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$s^2 \xi_\ell\, [h^{-2}{\rm Mpc^2}](s)$', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig

    def plot_sample_data_power(self, ell=0, save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.samples)):
            k, poles = self._sample_data_power[i](ell=(0, 2, 4), return_k=True, complex=False)
            ax.plot(k, k * poles[ell//2], lw=2.0, label=rf'${{\rm DTS}}_{i}$')
        ax.set_xlabel(r'$k\, [h\,{\rm Mpc}^{-1}]$', fontsize=15)
        ax.set_ylabel(r'$k P_{\ell}(k)\, [h^{2}\,{\rm Mpc}^{-2}] $', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig

    def plot_sample_power(self, ell=0, save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.samples)):
            k, poles = self._sample_power[i](ell=(0, 2, 4), return_k=True, complex=False)
            ax.plot(k, k * poles[ell//2], lw=2.0, label=rf'${{\rm DTS}}_{i}$')
        ax.set_xlabel(r'$k\, [h\,{\rm Mpc}^{-1}]$', fontsize=15)
        ax.set_ylabel(r'$k P_{\ell}(k)\, [h^{2}\,{\rm Mpc}^{-2}] $', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig


