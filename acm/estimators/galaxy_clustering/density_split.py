import time
import jax
from jaxpower import (
    MeshAttrs, ParticleField, FKPField,
    BinMesh2SpectrumPoles, get_mesh_attrs,
    compute_mesh2_spectrum, compute_fkp2_shotnoise,
    compute_box2_normalization
)
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pandas import qcut
from pypower import CatalogFFTPower
from pycorr import TwoPointCorrelationFunction
from .base import BaseEstimator
from acm.utils.plotting import set_plot_style


class DensitySplit(BaseEstimator):
    """
    Class to compute density-split clustering, as in http://arxiv.org/abs/2309.16541.
    Expects all positions passed in cartesian coordinates of shape (N, 3).
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
                raise ValueError('query_positions must be provided when working with a non-uniform geometry.')
            else:
                query_positions = self.get_query_positions(method=query_method, nquery=nquery_factor*self.size_data)
        self.query_method = query_method
        self.query_positions = query_positions
        self.delta_query = self.read_density_contrast(query_positions)
        self.quantiles_idx = qcut(self.delta_query, nquantiles, labels=False)
        quantiles = []
        for i in range(nquantiles):
            quantiles.append(self.query_positions[self.quantiles_idx == i])
        self.quantiles = quantiles
        self.logger.info(f"Quantiles calculated in {time.time() - t0:.2f} seconds.")
        return self.quantiles, self.quantiles_idx, self.delta_query

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
        if self.has_randoms:
            if 'randoms_positions' not in kwargs:
                raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
            kwargs['randoms_positions1'] = kwargs['randoms_positions']
            kwargs['randoms_positions2'] = kwargs['randoms_positions']
            kwargs.pop('randoms_positions')
            if 'data_weights' in kwargs:
                kwargs['data_weights1'] = None # setting default weights for quantiles
                kwargs['data_weights2'] = kwargs.pop('data_weights')
            if 'randoms_weights' in kwargs:
                kwargs['randoms_weights1'] = None
                kwargs['randoms_weights2'] = kwargs.pop('randoms_weights')
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.boxsize
        self._quantile_data_correlation = []
        R1R2 = None
        for quantile in self.quantiles: 
            result = TwoPointCorrelationFunction(
                data_positions1 = quantile,
                data_positions2 = data_positions,
                mode = 'smu',
                position_type = 'pos',
                R1R2 = R1R2,
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
        if self.has_randoms:
            if 'randoms_positions' not in kwargs:
                raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
            kwargs['randoms_positions1'] = kwargs.pop('randoms_positions')
            kwargs['data_weights1'] = None # setting default weights for quantiles
            kwargs['randoms_weights1'] = None
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.boxsize
        self._quantile_correlation = []
        R1R2 = None
        for quantile in self.quantiles: 
            result = TwoPointCorrelationFunction(
                data_positions1 = quantile,
                mode = 'smu',
                position_type = 'pos',
                R1R2 = R1R2,
                **kwargs,
            )
            self._quantile_correlation.append(result)
            R1R2 = result.R1R2

        return self._quantile_correlation

    def quantile_data_power(self, data_positions, edges={'step': 0.001}, ells=(0, 2, 4),
        los='z', resampler='tsc', interlacing=0, compensate=True, **kwargs):
        """
        Compute the cross-power spectrum between the data and the density field quantiles.

        Parameters
        ----------
        data_positions : array_like
            Positions of the data.
        edges : dict, optional
            Bin edges for the power spectrum.
        ells : tuple, optional
            Multipole moments to compute.
        los : str, optional
            Line-of-sight direction.
        resampler : str, optional
            Resampling scheme for the mesh painting.
        interlacing : int, optional
            Interlacing factor for the mesh painting.
        compensate : bool, optional
            Whether to apply compensation for the mass assignment scheme.
        kwargs : dict
            Additional arguments for pypower.CatalogFFTPower.

        Returns
        -------
        k : array_like
            Wavenumbers.
        quantile_data_power : array_like
            Cross-power spectrum between quantiles and data.
        """
        if self.has_randoms:
            if 'randoms_positions' not in kwargs:
                raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
            kwargs['randoms_positions1'] = kwargs['randoms_positions']
            kwargs['randoms_positions2'] = kwargs['randoms_positions']
            kwargs.pop('randoms_positions')
            if 'data_weights' in kwargs:
                kwargs['data_weights1'] = None # setting default weights for quantiles
                kwargs['data_weights2'] = kwargs.pop('data_weights')
            if 'randoms_weights' in kwargs:
                kwargs['randoms_weights1'] = None
                kwargs['randoms_weights2'] = kwargs.pop('randoms_weights')
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.boxsize

        # TODO handle survey-mode geometry with FKPField for data mesh

        jitted_compute_mesh2_spectrum = jax.jit(
            compute_mesh2_spectrum,
            static_argnames=['los'],
            donate_argnums=[0]
        )

        bin = BinMesh2SpectrumPoles(
            self.mattrs,
            edges=edges,
            ells=ells,
        )

        kw = dict(resampler=resampler, compensate=compensate, interlacing=interlacing)

        data = ParticleField(data_positions, attrs=self.mattrs, exchange=True, backend='jax')
        data_mesh = data.paint(**kw, out='real')
        data_mesh = data_mesh - data_mesh.mean()

        self._quantile_data_power = []
        for i, quantile_positions in enumerate(self.quantiles):
            t0 = time.time()

            quantile = ParticleField(quantile_positions, attrs=self.mattrs, exchange=True, backend='jax')

            norm = compute_box2_normalization(quantile, data, bin=bin)

            quantile_mesh = quantile.paint(**kw, out='real')
            quantile_mesh = quantile_mesh - quantile_mesh.mean()

            spectrum = jitted_compute_mesh2_spectrum(quantile_mesh, data_mesh, bin=bin, los=los)
            spectrum = spectrum.clone(norm=norm)

            self._quantile_data_power.append(spectrum)
            self.logger.info(f"Q{i}-galaxy spectrum calculated in {time.time() - t0:.2f} s.")
        return self._quantile_data_power

    def quantile_power(self, edges={'step': 0.001}, ells=(0, 2, 4),
        los='z', resampler='tsc', interlacing=0, compensate=True, **kwargs):
        """
        Compute the auto-power spectrum of the density field quantiles.

        Parameters
        ----------
        data_positions : array_like
            Positions of the data.
        edges : dict, optional
            Bin edges for the power spectrum.
        ells : tuple, optional
            Multipole moments to compute.
        los : str, optional
            Line-of-sight direction.
        resampler : str, optional
            Resampling scheme for the mesh painting.
        interlacing : int, optional
            Interlacing factor for the mesh painting.
        compensate : bool, optional
            Whether to apply compensation for the mass assignment scheme.
        kwargs : dict
            Additional arguments for pypower.CatalogFFTPower.

        Returns
        -------
        k : array_like
            Wavenumbers.
        quantile_power : array_like
            Auto-power spectrum of quantiles.
        """
        if self.has_randoms:
            if 'randoms_positions' not in kwargs:
                raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
            kwargs['randoms_positions1'] = kwargs.pop('randoms_positions')
            kwargs['data_weights1'] = None # setting default weights for quantiles
            kwargs['randoms_weights1'] = None
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.boxsize

        # TODO handle survey-mode geometry with FKPField for data mesh

        jitted_compute_mesh2_spectrum = jax.jit(
            compute_mesh2_spectrum,
            static_argnames=['los'],
            donate_argnums=[0]
        )

        bin = BinMesh2SpectrumPoles(
            self.mattrs,
            edges=edges,
            ells=ells,
        )

        kw = dict(resampler=resampler, compensate=compensate, interlacing=interlacing)

        self._quantile_power = []
        for i, quantile_positions in enumerate(self.quantiles):
            t0 = time.time()
            quantile = ParticleField(
                quantile_positions, attrs=self.mattrs, exchange=True, backend='jax'
            )

            norm = compute_box2_normalization(quantile, bin=bin)
            num_shotnoise = compute_fkp2_shotnoise(quantile, bin=bin)

            quantile_mesh = quantile.paint(**kw, out='real')
            # quantile_mesh = quantile_mesh / quantile_mesh.mean() - 1.
            quantile_mesh = quantile_mesh - quantile_mesh.mean()

            spectrum = jitted_compute_mesh2_spectrum(quantile_mesh, bin=bin, los=los)
            spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

            self._quantile_power.append(spectrum)
            self.logger.info(f"Q{i} auto-spectrum calculated in {time.time() - t0:.2f} s.")
        return self._quantile_power

    @set_plot_style
    def plot_quantiles(self, save_fn=None):
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
        if save_fn: plt.savefig(save_fn, bbox_inches='tight', dpi=300)
        return fig

    @set_plot_style
    def plot_quantile_data_correlation(self, ell=0, save_fn=None):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            s, multipoles = self._quantile_data_correlation[i](ells=(0, 2, 4), return_sep=True)
            ax.plot(s, s**2*multipoles[ell//2], lw=2.0, color=colors[i], label=rf'${{\rm Q}}_{i}$')
        ax.set_xlabel(r'$s\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$s^2 \xi_\ell\, [h^{-2}{\rm Mpc^2}](s)$', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight', dpi=300)
        return fig
    
    @set_plot_style
    def plot_quantile_correlation(self, ell=0, save_fn=None):
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

    @set_plot_style
    def plot_quantile_data_power(self, ell=0, save_fn=None):
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

    @set_plot_style
    def plot_quantile_power(self, ell=0, save_fn=None):
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