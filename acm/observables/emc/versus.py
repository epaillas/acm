import xarray
import numpy as np
import glob
from pathlib import Path
from .base import BaseObservableEMC
import matplotlib.pyplot as plt
from jaxpower import read
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.plotting import set_plot_style
from acm.utils.decorators import temporary_class_state
from acm.utils.xarray import dataset_to_dict, split_vars


class VERSUSVoidSizeFunction(BaseObservableEMC):
    """
    Class for the Emulator Mock Challenge's VERSUS void size function.
.
    """
    def __init__(self, **kwargs):
        super().__init__(stat_name='versus_vsf', n_test=6*1, **kwargs)
    
    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return 'none'
    
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
        # Directories
        base_dir = Path(self.paths['measurements_dir']) / 'small' / 'spherical_voids'
        data_fns = list(base_dir.glob('sv_ph*.npy')) # NOTE: File name format hardcoded !
        
        y = []
        for data_fn in data_fns:
            data = np.load(data_fn, allow_pickle=True)
            rv, vsf = data
            y.append(vsf)
        y = np.array(y)
        
        y = xarray.DataArray(
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
        
        self.logger.info(f'Loaded covariance with shape: {y.shape}')
        
        cout = xarray.Dataset(data_vars = {'covariance_y': y})
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
        ells: list = [0, 2, 4],
        phase_idx: int = 0,
        seed_idx: int = 0,
        test_filters: dict = None
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
        test_filters : dict, optional
            Dictionary of filters to split the dataset into training and test sets.
            Keys are the dimension names and values are the values to filter on for the test set.
            If None, no splitting is done. Default is None.
            
        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'x' and 'y' DataArrays. 
            If add_covariance is True, also contains 'covariance_y' DataArray.
        """
        base_dir = Path(self.paths['measurements_dir'],  f'base/spherical_voids/')

        y = []
        hods = {}
        for cosmo_idx in cosmos:
            self.logger.info(f'Compressing c{cosmo_idx:03}')
            handle = f'c{cosmo_idx:03}_ph000/seed0/sv_c{cosmo_idx:03}_hod*.npy'
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
            
        if test_filters is not None:
            for v_in, v_out in split_vars(cout.x, cout.y, **test_filters):
                v_in.name = v_in.name + '_test'
                v_out.name = v_out.name + '_train'
                v_in.attrs['nan_dims'] = list(test_filters.keys()) # Mark filtered dimensions that will be filled with NaNs
                v_out.attrs['nan_dims'] = list(test_filters.keys())
                cout = xarray.merge([cout, v_in, v_out])
        
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
        raise NotImplementedError()
    
# Alias
versus_vsf = VERSUSVoidSizeFunction