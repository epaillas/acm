import xarray
import numpy as np
import glob
from pathlib import Path
from .base import BaseObservableEMC
import matplotlib.pyplot as plt
from jaxpower import read
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.xarray_data import dataset_to_dict
from acm.utils.plotting import set_plot_style
from acm.utils.decorators import temporary_class_state
from pycorr import TwoPointCorrelationFunction

class BaseVERSUSVoidSizeFunction(BaseObservableEMC):
    """
    Class for the Emulator Mock Challenge's VERSUS void size function.
    
    """
    def __init__(self, **kwargs):
        if not hasattr(self, "recon"):
            raise ValueError("Subclass must define a class attribute 'recon'")
        self.label = ("recon_" if self.recon else "") + 'vsf'
        self.filedir = ("recon_" if self.recon else "") + 'spherical_voids'
        super().__init__(stat_name=f'versus_{self.label}', n_test=6*100, **kwargs)
    
    def compress_covariance(
        self,
        save_to: str = None,
    ) -> xarray.DataArray:
        """
        Compress the covariance array from the raw measurement files.
        
        Parameters
        ----------
        save_to : str
            Path of the directory where to save the compressed covariance and bin_values. If None, it is not saved.
            Default is None.
            
        Returns
        -------
        xarray.DataArray
            Covariance array. 
        """
        base_dir = Path(self.paths['measurements_dir']) / 'small' / self.filedir
        data_fns = list(base_dir.glob(f'sv_{self.label}_ph*.npy')) # NOTE: File name format hardcoded !
        
        y = []
        for data_fn in data_fns:
            data = np.load(data_fn, allow_pickle=True)
            rv, vsf = data
            y.append(vsf)
        y = np.array(y)
        self.logger.info(f'Loaded covariance with shape: {y.shape}')
        
        cout = xarray.DataArray(
            data = y.reshape(y.shape[0], -1),
            coords = {
                "phase_idx": list(range(y.shape[0])),
                'rv': rv,
            },
            attrs = {
                "sample": ["phase_idx"],
                'features': ['rv'],
            },
            name = "covariance_y",
        )
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed covariance file to {save_fn}')
        return cout

    def compress_data(
        self, 
        add_covariance: bool = False,
        save_to: str = None,
        cosmos: list = cosmo_list,
        n_hod: int = 500,
        phase_idx: int = 0,
        seed_idx: int = 0,
    ) -> dict:
        """
        Compress the data from the VSF raw measurement files.
        
        Parameters
        ----------
        add_covariance : bool
            If True, add the covariance to the compressed data. Default is False.
        save_to : str
            Path of the directory where to save the compressed file. If None, it is not saved.
            Default is None.
        cosmos : list
            List of cosmological parameters to use. If None, use all cosmological parameters.
            Default is None.
        n_hod : int
            Number of HOD parameters to use. Default is 100.
        phase_idx : int
            TODO
        seed_idx : int
            TODO
            
        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'x' and 'y' DataArrays. 
            If add_covariance is True, also contains 'covariance_y' DataArray.
        """
        base_dir = Path(self.paths['measurements_dir']) / 'base' / self.filedir

        y = []
        hods = {}
        for cosmo_idx in cosmos:
            self.logger.info(f'Compressing c{cosmo_idx:03}')
            handle = f'c{cosmo_idx:03}_ph000/seed0/sv_{self.label}_c{cosmo_idx:03}_hod*.npy'
            filenames = sorted(base_dir.glob(handle))[:n_hod]
            for filename in filenames:
                data = np.load(filename, allow_pickle=True)
                rv, vsf = data
                y.append(vsf)
            hods[cosmo_idx] = self.get_raw_hod_idx(cosmo_idx)[:n_hod]
        y = np.array(y)

        y = xarray.DataArray(
            data = y.reshape(len(cosmos), n_hod, -1),
            coords = {
                'cosmo_idx': cosmos,
                'hod_idx': list(range(n_hod)),
                'rv': rv,
            },
            attrs = {
                'sample': ['cosmo_idx', 'hod_idx'],
                'features': ['rv'],
            },
            name = 'y',
        )
        x = self.compress_x(hods=hods, cosmos=cosmos)
        
        self.logger.info(f'Loaded data with shape: {x.shape}, {y.shape}')
        
        cout = xarray.Dataset(
            data_vars = {
                'x': x,
                'y': y,
            },
        )
        if add_covariance:
            cov_y = self.compress_covariance()
            cout = xarray.merge([cout, cov_y])
        
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed data to {save_fn}')
        return cout

    @set_plot_style
    def plot_training_set(self, save_fn: str = None):
        """
        Plot the training set for the observable.
        
        Parameters
        ----------
        save_fn : str
            Path to save the figure. If None, the figure is not saved.
            Default is None.
        """
        rv = self.rv.values

        fig, ax = plt.subplots(figsize=(5, 4))

        for data in self.y:
            ax.plot(rv, data, alpha=0.5, lw=0.3)

        ax.set_ylabel(r'$n_{\rm void}\,[h^3{\rm Mpc}^{-3}]$')
        ax.set_xlabel(r'$R_{\rm void}\, [h^{-1}{\rm Mpc}]$')

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving training set figure to {save_fn}')
            
        return fig, ax

    @set_plot_style
    def plot_covariance_set(self, save_fn: str = None):
        """
        Plot the covariance set for the observable.
        
        Parameters
        ----------
        save_fn : str
            Path to save the figure. If None, the figure is not saved.
            Default is None.
        """
        rv = self.rv.values

        fig, ax = plt.subplots(figsize=(5, 4))

        for data in self.covariance_y:
            ax.plot(rv, data, color='grey', alpha=0.5, lw=0.1)

        ax.set_ylabel(r'$n_{\rm void}\,[h^3{\rm Mpc}^{-3}]$')
        ax.set_xlabel(r'$R_{\rm void}\, [h^{-1}{\rm Mpc}]$')

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving training set figure to {save_fn}')
            
        return fig, ax

    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_observable(self, model_params: dict, save_fn: str = None):
        """
        Plot the data, model, and residuals.

        Parameters
        ----------
        model_params : dict
            Dictionary of model parameters to use for the prediction.
        save_fn : str
            Filename to save the plot. If None, the plot is not saved.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, numpy.ndarray
            Figure and axes of the plot.
        """

        height_ratios = [3, 1]
        figsize = (6, 1.5 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False,
            gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0.1)
        
        lax[0].set_ylabel(r'$n_{\rm void}\,[h^3{\rm Mpc}^{-3}]$')
        lax[-1].set_xlabel(r'$R_{\rm void}\, [h^{-1}{\rm Mpc}]$')

        rv = self.rv.values
        data = self.y
        model = self.get_model_prediction(model_params)
        
        cov = self.get_covariance_matrix(volume_factor=64)
        error = np.sqrt(np.diag(cov))
        lax[0].errorbar(rv, data, error, marker='o', ms=4, ls='', 
            color='C0', elinewidth=1.0, capsize=None)
        lax[0].plot(rv, model, ls='-', color='C0')
        lax[1].plot(rv, (data - model) / error, ls='-', color='C0')

        for offset in [-2, 2]: lax[1].axhline(offset, color='k', ls='--')
        lax[1].set_ylabel(r'$\Delta n_{\rm void} / \sigma_{ n_{\rm void} }$', fontsize=15)
        lax[1].set_ylim(-4, 4)

        for ax in lax:
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=14)

        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving plot to {save_fn}')
        return fig, lax


class BaseVERSUSCorrelationFunctionMultipoles(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge VERSUS void-galaxy correlation
    function multipoles observable.
    """
    def __init__(self, **kwargs):
        if not hasattr(self, "corr_type"):
            raise ValueError("Subclass must define a class attribute 'corr_type'")
        if not hasattr(self, "recon"):
            raise ValueError("Subclass must define a class attribute 'recon'")
        self.label = ("recon_" if self.recon else "") + self.corr_type
        self.filedir = ("recon_" if self.recon else "") + 'spherical_voids'
        super().__init__(stat_name=f'versus_{self.label}', 
                        n_test=6*100, **kwargs)
    
    def compress_covariance(
        self,
        save_to: str = None,
        rebin: int = 1,
        ells: list = [0, 2],
    ) -> xarray.DataArray:
        """
        Compress the covariance array from the raw measurement files.
        
        Parameters
        ----------
        save_to : str
            Path of the directory where to save the compressed covariance and bin_values. If None, it is not saved.
            Default is None.
        rebin : int
            Rebinning factor for the statistics. Default is 1.
        ells : list
            List of multipoles to compute the statistics for. Default is [0, 2].
            
        Returns
        -------
        xarray.DataArray
            Covariance array. 
        """
        base_dir = Path(self.paths['measurements_dir']) / 'small' / self.filedir
        data_fns = list(base_dir.glob(f'sv_{self.label}_ph*.npy')) # NOTE: File name format hardcoded !
        
        y = []
        for data_fn in data_fns:
            data = TwoPointCorrelationFunction.load(data_fn)#[:-1:rebin]
            s, multipoles = data(ells=ells, return_sep=True) 
            y.append(np.concatenate(multipoles))
        y = np.array(y)

        self.logger.info(f'Loaded covariance with shape: {y.shape}')
        
        cout = xarray.DataArray(
            data = y.reshape(y.shape[0], len(ells), -1),
            coords = {
                "phase_idx": list(range(y.shape[0])),
                'ells': ells,
                's': s,
            },
            attrs = {
                "sample": ["phase_idx"],
                'features': ['ells', 's'],
            },
            name = "covariance_y",
        )
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed covariance file to {save_fn}')
        return cout

    def compress_data(
        self, 
        add_covariance: bool = False,
        save_to: str = None,
        cosmos: list = cosmo_list,
        n_hod: int = 500,
        rebin: int = 1,
        ells: list = [0, 2],
        phase_idx: int = 0,
        seed_idx: int = 0,
    ) -> dict:
        """
        Compress the data from the tpcf raw measurement files.
        
        Parameters
        ----------
        add_covariance : bool
            If True, add the covariance to the compressed data. Default is False.
        save_to : str
            Path of the directory where to save the compressed file. If None, it is not saved.
            Default is None.
        cosmos : list
            List of cosmological parameters to use. If None, use all cosmological parameters.
            Default is None.
        n_hod : int
            Number of HOD parameters to use. Default is 100.
        rebin : int
            Rebinning factor for the statistics. Default is 1.
        ells : list
            List of multipoles to compute the statistics for. Default is [0, 2].
        phase_idx : int
            TODO
        seed_idx : int
            TODO
            
        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'x' and 'y' DataArrays. 
            If add_covariance is True, also contains 'covariance_y' DataArray.
        """
        base_dir = Path(self.paths['measurements_dir']) / 'base' / self.filedir

        y = []
        hods = {}
        for cosmo_idx in cosmos:
            self.logger.info(f'Compressing c{cosmo_idx:03}')
            handle = f'c{cosmo_idx:03}_ph000/seed0/sv_{self.label}_c{cosmo_idx:03}_hod*.npy'
            filenames = sorted(base_dir.glob(handle))[:n_hod]
            for filename in filenames:
                data = TwoPointCorrelationFunction.load(filename)#[::rebin]
                s, multipoles = data(ells=ells, return_sep=True) 
                y.append(np.concatenate(multipoles))
            hods[cosmo_idx] = self.get_raw_hod_idx(cosmo_idx)[:n_hod]
        
        y = np.array(y)
        y = xarray.DataArray(
            data = y.reshape(len(cosmos), n_hod, len(ells), -1),
            coords = {
                'cosmo_idx': cosmos,
                'hod_idx': list(range(n_hod)),
                'ells': ells,
                's': s,
            },
            attrs = {
                'sample': ['cosmo_idx', 'hod_idx'],
                'features': ['ells', 's'],
            },
            name = 'y',
        )
        x = self.compress_x(hods=hods, cosmos=cosmos)
        
        self.logger.info(f'Loaded data with shape: {x.shape}, {y.shape}')
        
        cout = xarray.Dataset(
            data_vars = {
                'x': x,
                'y': y,
            },
        )
        if add_covariance:
            cov_y = self.compress_covariance(ells=ells)
            cout = xarray.merge([cout, cov_y])
        
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed data to {save_fn}')
        return cout

    @set_plot_style
    def plot_training_set(self, save_fn: str = None):
        """
        Plot the training set for the observable.
        
        Parameters
        ----------
        save_fn : str
            Path to save the figure. If None, the figure is not saved.
            Default is None.
        """
        ells = self._dataset.y.coords['ells'].values.tolist()
        s = self.s.values

        fig, lax = plt.subplots(len(ells), 1, figsize=(4, 5), sharex=True)
        self.select_filters = {}
        for (l,ell) in enumerate(ells):
            self.select_filters.update({'ells': ell})
            for data in self.y:
                lax[ell//2].plot(s, data, color='C0', alpha=0.5, lw=0.1)
            lax[ell//2].set_ylabel(rf'$\xi_{ell}(s)$')
        lax[-1].set_xlabel(r'$s [h^{-1}{\rm Mpc}]$')

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving training set figure to {save_fn}')
            
        return fig, lax

    @set_plot_style
    def plot_covariance_set(self, save_fn: str = None):
        """
        Plot the covariance set for the observable.
        
        Parameters
        ----------
        save_fn : str
            Path to save the figure. If None, the figure is not saved.
            Default is None.
        """
        ells = self._dataset.y.coords['ells'].values.tolist()
        s = self.s.values

        fig, lax = plt.subplots(len(ells), 1, figsize=(4, 5), sharex=True)

        for ell in ells:
            self.select_filters.update({'ells': ell})
            for data in self.covariance_y:
                lax[ell//2].plot(s, data, color='C0', alpha=0.5, lw=0.1)
            lax[ell//2].set_ylabel(rf'$\xi_{ell}(s)$')
        lax[-1].set_xlabel(r'$s [h^{-1}{\rm Mpc}]$')

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving training set figure to {save_fn}')
            
        return fig, lax

    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_observable(self, model_params: dict, save_fn: str = None):
        """
        Plot the data, model, and residuals.

        Parameters
        ----------
        model_params : dict
            Dictionary of model parameters to use for the prediction.
        save_fn : str
            Filename to save the plot. If None, the plot is not saved.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, numpy.ndarray
            Figure and axes of the plot.
        """

        ells = self._dataset.y.coords['ells'].values.tolist()

        height_ratios = [max(len(ells), 3)] + [1] * len(ells)
        figsize = (6, 1.5 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False,
            gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0.1)
        show_legend = True
        
        for i, ell in enumerate(ells):
            lax[0].set_ylabel(rf'$\xi_{ell}(s)$', fontsize=15)
            lax[-1].set_xlabel(r'$s [h^{-1}{\rm Mpc}]$', fontsize=15)

            self.select_filters.update({'ells': ell})

            s = self.s.values
            data = self.y
            model = self.get_model_prediction(model_params)
            
            cov = self.get_covariance_matrix(volume_factor=64)
            error = np.sqrt(np.diag(cov))
            lax[0].errorbar(s, data, error, marker='o', ms=4, ls='', 
                color=f'C{i}', elinewidth=1.0, capsize=None, label=f'$\ell={ell}$')
            lax[0].plot(s, model, ls='-', color=f'C{i}')
            lax[i + 1].plot(s, (data - model) / error, ls='-', color=f'C{i}')

            for offset in [-2, 2]: lax[i + 1].axhline(offset, color='k', ls='--')
            lax[i + 1].set_ylabel(rf'$\Delta \xi_{{{ell:d}}} / \sigma_{{ \xi_{{{ell:d}}} }}$', fontsize=15)
            lax[i + 1].set_ylim(-4, 4)

        for ax in lax:
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=14)
        if show_legend: lax[0].legend(fontsize=15)

        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving plot to {save_fn}')
        return fig, lax


class VERSUSVoidSizeFunction(BaseVERSUSVoidSizeFunction):
    recon = False

    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/models/v1.2/best/versus_vsf/last.ckpt'


class ReconstructedVERSUSVoidSizeFunction(BaseVERSUSVoidSizeFunction):
    recon = True

    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return 'none'


class VERSUSVoidGalaxyCorrelationFunctionMultipoles(BaseVERSUSCorrelationFunctionMultipoles):
    corr_type = 'xivg'
    recon = False

    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/models/v1.2/best/versus_xivg/last.ckpt'


class VERSUSVoidAutoCorrelationFunctionMultipoles(BaseVERSUSCorrelationFunctionMultipoles):
    corr_type = 'xivv'
    recon = False

    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/models/v1.2/best/versus_xivv/last.ckpt'


class ReconstructedVERSUSVoidGalaxyCorrelationFunctionMultipoles(BaseVERSUSCorrelationFunctionMultipoles):
    corr_type = 'xivg'
    recon = True 

    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return 'none'
        

class ReconstructedVERSUSVoidAutoCorrelationFunctionMultipoles(BaseVERSUSCorrelationFunctionMultipoles):
    corr_type = 'xivv'
    recon = True 

    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return 'none'
