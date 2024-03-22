from pyrecon import RealMesh
import numpy as np
import logging
import time
from pandas import qcut
from .base import BaseEstimator


class DensitySplit(BaseEstimator):
    """
    Class to compute the wavelet scattering transform.
    """
    def __init__(self, **kwargs):

        self.logger = logging.getLogger('DensitySplit')
        self.logger.info('Initializing DensitySplit.')

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

    def set_density_contrast(self, query_positions=None, query_method='randoms', smoothing_radius=None, check=False, ran_min=0.01):
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
            threshold = ran_min * sum_randoms / self._size_randoms
            mask = self.randoms_mesh > threshold
            self.delta_mesh[mask] /= alpha * self.randoms_mesh[mask]
            self.delta_mesh[~mask] = 0.0
        else:
            self.delta_mesh = self.data_mesh / np.mean(self.data_mesh) - 1.
        if query_positions is None:
            if self.has_randoms:
                raise ValueError('Query points must be provided when working with a non-uniform geometry.')
            else:
                query_positions = self.get_query_positions(self.data_mesh, method=query_method)
        self.query_method = query_method
        self.delta_mesh = self.delta_mesh.read_cic(query_positions)
        return self.delta_mesh

    def get_query_positions(self, mesh, method='randoms', nquery=None, seed=42):
        """
        Get positions at the center of each mesh cell.

        Returns
        -------
        lattice_positions : array_like
            Lattice positions.
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
            self.query_positions = np.vstack((lattice_x, lattice_y, lattice_z)).T
        elif method == 'randoms':
            self.logger.info('Generating random query points within the box.')
            np.random.seed(seed)
            if nquery is None:
                nquery = 5 * self._size_data
            self.query_positions = np.random.rand(nquery, 3) * boxsize
        return self.query_positions


    def set_quantiles(self, nquantiles=5, return_index=False):
        """
        Get the quantiles of the overdensity density field.

        Parameters
        ----------
        nquantiles : int
            Number of quantiles.
        return_idx : bool, optional
            Whether to return index of the quantile of each query point.

        Returns
        -------
        quantiles : array_like
            Quantiles of the density field.
        quantiles_idx : array_like, optional
            Index of the quantile of each query point.
        """
        t0 = time.time()
        self.quantiles_idx = qcut(self.delta_mesh, nquantiles, labels=False)
        quantiles = []
        for i in range(nquantiles):
            quantiles.append(self.query_positions[self.quantiles_idx == i])
        self.quantiles = quantiles
        self.logger.info(f"Quantiles calculated in {time.time() - t0:.2f} seconds.")
        if return_index:
            return self.quantiles, self.quantiles_idx
        return quantiles

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
        ccf : array_like
            Cross-correlation function.
        """
        from pycorr import TwoPointCorrelationFunction
        if self.has_randoms and 'randoms_positions' not in kwargs:
            raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.data_mesh.boxsize
        self.quantile_data_ccf = []
        for quantile in self.quantiles:
            result = TwoPointCorrelationFunction(
                data_positions1=quantile,
                data_positions2=data_positions,
                mode='smu',
                position_type='pos',
                **kwargs,
            )
            self.s, multipoles = result(ells=(0, 2, 4), return_sep=True)
            self.quantile_data_ccf.append(multipoles)
        return self.s, self.quantile_data_ccf

    def quantile_correlation(self, **kwargs):
        """
        Compute the auto-correlation function of the density field quantiles.

        Parameters
        ----------
        kwargs : dict
            Additional arguments for pycorr.TwoPointCorrelationFunction.

        Returns
        -------
        acf : array_like
            Auto-correlation function.
        """
        from pycorr import TwoPointCorrelationFunction
        if self.has_randoms and 'randoms_positions' not in kwargs:
            raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.data_mesh.boxsize
        self.quantile_acf = []
        for quantile in self.quantiles:
            result = TwoPointCorrelationFunction(
                data_positions1=quantile,
                mode='smu',
                position_type='pos',
                **kwargs,
            )
            self.s, multipoles = result(ells=(0, 2, 4), return_sep=True)
            self.quantile_acf.append(multipoles)
        return self.s, self.quantile_acf

    def quantile_data_power(self, data_positions, **kwargs):
        """
        Compute the cross-power spectrum between the data and the density field quantiles.
        """
        from pypower import CatalogFFTPower
        if self.has_randoms and 'randoms_positions' not in kwargs:
            raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.data_mesh.boxsize
        if self.query_method == 'lattice':
            kwargs['shotnoise'] = 0.0
        self.quantile_data_power = []
        for quantile in self.quantiles:
            result = CatalogFFTPower(
                data_positions1=quantile,
                data_positions2=data_positions,
                ells=(0, 2, 4),
                position_type='pos',
                **kwargs,
            ).poles
            self.k, multipoles = result(ell=(0, 2, 4), return_k=True, complex=False)
            self.quantile_data_power.append(multipoles)
        return self.k, self.quantile_data_power

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
        power : array_like
            Power spectrum.
        """
        from pypower import CatalogFFTPower
        if self.has_randoms and 'randoms_positions' not in kwargs:
            raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.data_mesh.boxsize
        if self.query_method == 'lattice':
            kwargs['shotnoise'] = 0.0
        self.quantile_power = []
        for quantile in self.quantiles:
            result = CatalogFFTPower(
                data_positions1=quantile,
                ells=(0, 2, 4),
                position_type='pos',
                **kwargs,
            ).poles
            self.k, multipoles = result(ell=(0, 2, 4), return_k=True, complex=False)
            self.quantile_power.append(multipoles)
        return self.k, self.quantile_power

    def plot_quantiles(self):
        import matplotlib.pyplot as plt
        import matplotlib
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        cmap = matplotlib.cm.get_cmap('coolwarm')
        colors = cmap(np.linspace(0.01, 0.99, 5))
        hist, bin_edges, patches = ax.hist(self.delta_mesh, bins=200, density=True, lw=3.0, color='grey')
        imin = 0
        for i in range(5):
            dmax = self.delta_mesh[self.quantiles_idx == i].max()
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

    def plot_quantile_data_correlation(self):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            multipoles = self.quantile_data_ccf[i]
            ax.plot(self.s, self.s**2*multipoles[0], lw=3.0, label=rf'${{\rm Q}}_{i}$')
        ax.set_xlabel(r'$s\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$s^2 \xi_0(s)$', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        plt.show()
        return fig
    
    def plot_quantile_correlation(self):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            multipoles = self.quantile_acf[i]
            ax.plot(self.s, self.s**2*multipoles[0], lw=3.0, label=rf'${{\rm Q}}_{i}$')
        ax.set_xlabel(r'$s\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$s^2 \xi_0(s)$', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_quantile_data_power(self, save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            poles = self.quantile_data_power[i]
            ax.plot(self.k, self.k * poles[0], lw=3.0, label=rf'${{\rm Q}}_{i}$')
        ax.set_xlabel(r'$k\, [h\,{\rm Mpc}^{-1}]$', fontsize=15)
        ax.set_ylabel(r'$k P(k)\, [h^{2}\,{\rm Mpc}^{-2}] $', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig

    def plot_quantile_power(self, save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            poles = self.quantile_power[i]
            ax.plot(self.k, self.k * poles[0], lw=3.0, label=rf'${{\rm Q}}_{i}$')
        ax.set_xlabel(r'$k\, [h\,{\rm Mpc}^{-1}]$', fontsize=15)
        ax.set_ylabel(r'$k P(k)\, [h^{2}\,{\rm Mpc}^{-2}] $', fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig