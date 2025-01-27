import numpy as np
import logging
import time
from pandas import qcut
from .base import BaseEnvironmentEstimator, BaseCatalogMeshEstimator


class DensitySplit(BaseEnvironmentEstimator):
    """
    Class to compute density-split clustering, as in http://arxiv.org/abs/2309.16541.
    """
    def __init__(self, **kwargs):
        self.logger = logging.getLogger('DensitySplit')
        self.logger.info('Initializing DensitySplit.')
        super().__init__(**kwargs)

    def set_quantiles(self, query_positions=None, query_method='randoms',
        nquery_factor=5, nquantiles=5):
        """
        Get the quantiles of the overdensity density field.

        Parameters
        ----------
        query_positions : array_like, optional
            Query positions.
        query_method : str, optional
            Method to generate query points. Options are 'lattice' or 'randoms'.
        nquery_factor : int, optional
            Factor to multiply the number of data points to get the number of query points.
        nquantiles : int
            Number of quantiles.

        Returns
        -------
        quantiles : array_like
            Quantiles of the density field.
        quantiles_idx : array_like, optional
            Index of the quantile of each query point.
        delta_query : array_like, optional
            Density contrast at the query points.
        """
        t0 = time.time()
        if query_positions is None:
            if self.has_randoms:
                raise ValueError('Query points must be provided when working with a non-uniform geometry.')
            else:
                query_positions = self.get_query_positions(self.delta_mesh, method=query_method,
                                                           nquery=nquery_factor*self._size_data)
        self.query_method = query_method
        self.query_positions = query_positions
        self.delta_query = self.delta_mesh.read_cic(query_positions)
        self.quantiles_idx = qcut(self.delta_query, nquantiles, labels=False)
        quantiles = []
        for i in range(nquantiles):
            quantiles.append(self.query_positions[self.quantiles_idx == i])
        self.quantiles = quantiles
        self.logger.info(f"Quantiles calculated in {time.time() - t0:.2f} seconds.")
        return self.quantiles, self.quantiles_idx, self.delta_query

    def to_healpy(self, positions, nside):
        import healpy as hp
        pix = hp.ang2pix(nside, positions[:, 0], positions[:, 1], lonlat=True)
        hpmap = np.bincount(pix, minlength=12*nside**2)
        hpmap = hp.ma(hpmap, badval=0)
        hpmap.mask = hpmap == 0
        return hpmap

    def to_treecorr(self, positions):
        from treecorr import Catalog
        return Catalog(ra=positions[:, 0], dec=positions[:, 1], ra_units='deg', dec_units='deg')



    def quantile_data_correlation(self, data_positions, **kwargs):
        """
        Compute the cross-correlation function between the density field
        quantiles and the data.

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
        quantile_data_ccf : array_like
            Cross-correlation function between quantiles and data.
        """
        from pycorr import TwoPointCorrelationFunction
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
                kwargs['boxsize'] = self.delta_mesh.boxsize
        self._quantile_data_correlation = []
        R1R2 = None
        for quantile in self.quantiles:
            result = TwoPointCorrelationFunction(
                data_positions1=quantile,
                data_positions2=data_positions,
                mode='smu',
                position_type='pos',
                R1R2=R1R2,
                **kwargs,
            )
            self._quantile_data_correlation.append(result)
            R1R2 = result.R1R2
        return self._quantile_data_correlation

    def quantile_correlation(self, **kwargs):
        """
        Compute the auto-correlation function of the density field quantiles.

        Parameters
        ----------
        kwargs : dict
            Additional arguments for pycorr.TwoPointCorrelationFunction.

        Returns
        -------
        s : array_like
            Pair separations.
        quantile_acf : array_like
            Auto-correlation function of quantiles.
        """
        from pycorr import TwoPointCorrelationFunction
        if self.has_randoms:
            if 'randoms_positions' not in kwargs:
                raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.delta_mesh.boxsize
        self._quantile_correlation = []
        for quantile in self.quantiles:
            result = TwoPointCorrelationFunction(
                data_positions1=quantile,
                mode='smu',
                position_type='pos',
                **kwargs,
            )
            self._quantile_correlation.append(result)
        return self._quantile_correlation

    def quantile_data_power(self, data_positions, **kwargs):
        """
        Compute the cross-power spectrum between the data and the density field quantiles.

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
        quantile_data_power : array_like
            Cross-power spectrum between quantiles and data.
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
                kwargs['boxsize'] = self.delta_mesh.boxsize
        if self.query_method == 'lattice':
            kwargs['shotnoise'] = 0.0
        self._quantile_data_power = []
        for quantile in self.quantiles:
            result = CatalogFFTPower(
                data_positions1=quantile,
                data_positions2=data_positions,
                ells=(0, 2, 4),
                position_type='pos',
                **kwargs,
            ).poles
            self._quantile_data_power.append(result)
        return self._quantile_data_power

    def quantile_power(self, **kwargs):
        """
        Compute the auto-power spectrum of the density field quantiles.

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
        quantile_power : array_like
            Auto-power spectrum of quantiles.
        """
        from pypower import CatalogFFTPower
        if self.has_randoms:
            if 'randoms_positions' not in kwargs:
                raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
            kwargs['randoms_positions1'] = kwargs['randoms_positions']
            kwargs.pop('randoms_positions')
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.delta_mesh.boxsize
        if self.query_method == 'lattice':
            kwargs['shotnoise'] = 0.0
        self._quantile_power = []
        for quantile in self.quantiles:
            result = CatalogFFTPower(
                data_positions1=quantile,
                ells=(0, 2, 4),
                position_type='pos',
                **kwargs,
            ).poles
            self._quantile_power.append(result)
        return self._quantile_power

    def plot_quantiles(self):
        import matplotlib.pyplot as plt
        import matplotlib
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        cmap = matplotlib.cm.get_cmap('coolwarm')
        colors = cmap(np.linspace(0.01, 0.99, 5))
        hist, bin_edges, patches = ax.hist(self.delta_query, bins=200, density=True, lw=2.0, color='grey')
        imin = 0
        for i in range(len(self.quantiles)):
            dmax = self.delta_query[self.quantiles_idx == i].max()
            imax = np.digitize([dmax], bin_edges)[0] - 1
            for index in range(imin, imax):
                patches[index].set_facecolor(colors[i])
            imin = imax
            ax.plot(np.nan, np.nan, color=colors[i], label=rf'${{\rm Q}}_{i}$', lw=4.0)
        ax.set_xlabel(r'$\Delta \left(R_s = 10\, h^{-1}{\rm Mpc}\right)$', fontsize=15)
        ax.set_ylabel('PDF', fontsize=15)
        ax.set_xlim(-1.3, 3.0)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_quantile_data_correlation(self, ell=0, save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            s, multipoles = self._quantile_data_correlation[i](ells=(0, 2, 4), return_sep=True)
            ax.plot(s, s**2*multipoles[ell//2], lw=2.0, color=colors[i], label=rf'${{\rm Q}}_{i}$')
        ax.set_xlabel(r'$s\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$s^2 \xi_\ell\, [h^{-2}{\rm Mpc^2}](s)$', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_quantile_correlation(self, ell=0, save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            s, multipoles = self._quantile_correlation[i](ells=(0, 2, 4), return_sep=True)
            ax.plot(s, s**2*multipoles[ell//2], lw=2.0, label=rf'${{\rm Q}}_{i}$')
        ax.set_xlabel(r'$s\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$s^2 \xi_\ell\, [h^{-2}{\rm Mpc^2}](s)$', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig

    def plot_quantile_data_power(self, ell=0, save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            k, poles = self._quantile_data_power[i](ell=(0, 2, 4), return_k=True, complex=False)
            ax.plot(k, k * poles[ell//2], lw=2.0, label=rf'${{\rm Q}}_{i}$')
        ax.set_xlabel(r'$k\, [h\,{\rm Mpc}^{-1}]$', fontsize=15)
        ax.set_ylabel(r'$k P_{\ell}(k)\, [h^{2}\,{\rm Mpc}^{-2}] $', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig

    def plot_quantile_power(self, ell=0, save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            k, poles = self._quantile_power[i](ell=(0, 2, 4), return_k=True, complex=False)
            ax.plot(k, k * poles[ell//2], lw=2.0, label=rf'${{\rm Q}}_{i}$')
        ax.set_xlabel(r'$k\, [h\,{\rm Mpc}^{-1}]$', fontsize=15)
        ax.set_ylabel(r'$k P_{\ell}(k)\, [h^{2}\,{\rm Mpc}^{-2}] $', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig

