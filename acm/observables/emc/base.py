import torch
import xarray
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from acm.observables import Observable
from acm.utils import get_data_dirs
from acm.utils.abacus import load_abacus_cosmologies
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.xarray_data import dataset_to_dict
from acm.utils.plotting import set_plot_style

class BaseObservableEMC(Observable):
    """
    Base class for all the observables in the EMC project.
    """
    def __init__(self, flat_output_dims: int = 2, phase_correction: bool = False, **kwargs):
        if phase_correction and hasattr(self, 'compute_phase_correction'):
            self.logger.info('Computing phase correction.')
            self.phase_correction = self.compute_phase_correction()
            
        paths = kwargs.pop('paths', get_data_dirs('emc'))
        self.n_test = kwargs.pop('n_test', 6*500) # Default number of test samples for EMC
        super().__init__(paths=paths, flat_output_dims=flat_output_dims, **kwargs)

    def get_emulator_covariance_y(self, n_test: int|list = None, nofilters: bool = False) -> xarray.DataArray|np.ndarray:
        """
        Returns the unfiltered covariance array of the emulator error of the statistic, with shape (n_test, n_statistics).
        
        Parameters
        ----------
        n_test : int|list, optional
            Number of test samples or list of indices of the test samples. The default is None.
        nofilters : bool, optional
            If True, no filters are applied to the output and the full DataArray is returned. Defaults to False.
            
        Returns
        -------
        np.ndarray
            Array of the emulator covariance array, with shape (n_test, n_features).
        """
        n_test = n_test if n_test else self.n_test

        self.logger.info(f'Computing emulator covariance array for {n_test} test samples.')
        
        # Get unfiltered values
        x = self._dataset.x 
        y = self._dataset.y
        
        # Flatten on 2D for indexing
        x = self.stack_on_attribute('sample', x)
        x = self.stack_on_attribute('features', x)
        y = self.stack_on_attribute('sample', y)
        y = self.stack_on_attribute('features', y)
        
        if isinstance(n_test, int):
            idx_test = list(range(n_test))
        elif isinstance(n_test, list):
            idx_test = list(set(n_test))
        
        test_x = x[idx_test]
        test_y = y[idx_test]
        
        prediction = self.get_model_prediction(test_x, nofilters=True) # Unfiltered prediction !
        
        # Flatten on 2D for indexing
        prediction = self.stack_on_attribute('sample', prediction)
        prediction = self.stack_on_attribute('features', prediction)
        
        if isinstance(test_y, xarray.DataArray):
            test_y = test_y.values
        if isinstance(prediction, xarray.DataArray):
            prediction = prediction.values
            
        diff = test_y - prediction # NOTE: 2D flattening is done to ensure correct broadcasting here !

        y = y.unstack()
        shape = (len(idx_test),) + y.shape[len(y.attrs['sample']):]
        emulator_covariance_y = xarray.DataArray(
            diff.reshape(shape),
            coords = {
                'n_test': idx_test,
                **{k: y.coords[k] for k in y.dims if k in y.attrs['features']}
            },
            attrs = {
                'sample': ['n_test'],
                'features': y.attrs['features'],
            },
            name = 'emulator_covariance_y',
        )
        
        if nofilters:
            return emulator_covariance_y
        
        emulator_covariance_y = self.apply_filters(emulator_covariance_y)
        if 'emulator_covariance_y' in self.select_indices_on:
            emulator_covariance_y = self.apply_indices_selection(emulator_covariance_y)
        emulator_covariance_y = self.flatten_output(emulator_covariance_y, self.flat_output_dims)
        if self.squeeze_output:
            emulator_covariance_y = emulator_covariance_y.squeeze()
        if self.numpy_output:
            emulator_covariance_y = emulator_covariance_y.values
        return emulator_covariance_y

    def get_emulator_error(self, n_test: int|list = None) -> xarray.DataArray|np.ndarray:
        """
        Returns the unfiltered emulator error of the statistic, with shape (n_statistics, ).
        
        Parameters
        ----------
        n_test : int|list, optional
            Number of test samples or list of indices of the test samples. The default is None.
        
        Returns
        -------
        np.ndarray
           Emulator error, with shape (n_features, ).
        """
        n_test = n_test if n_test else self.n_test
        
        emulator_covariance_y = self.get_emulator_covariance_y(n_test, nofilters=True) # Unfiltered covariance array !
        
        # Flatten on 2D for indexing
        emulator_covariance_y = self.stack_on_attribute('sample', emulator_covariance_y)
        emulator_covariance_y = self.stack_on_attribute('features', emulator_covariance_y)
        
        emulator_error = np.median(np.abs(emulator_covariance_y), axis=0)

        y = self._dataset.y.unstack()
        shape = y.shape[len(y.attrs['sample']):]
        emulator_error = xarray.DataArray(
            emulator_error.reshape(shape),
            coords = {
                **{k: y.coords[k] for k in y.dims if k in y.attrs['features']}
            },
            attrs = {
                'sample': [],
                'features': y.attrs['features'],
            },
            name = 'emulator_error',
        )
        emulator_error = self.apply_filters(emulator_error)
        if 'emulator_error' in self.select_indices_on:
            emulator_error = self.apply_indices_selection(emulator_error)
        emulator_error = self.flatten_output(emulator_error, self.flat_output_dims)
        if self.squeeze_output:
            emulator_error = emulator_error.squeeze()
        if self.numpy_output:
            emulator_error = emulator_error.values
        return emulator_error

    # NOTE: Override Observable prediction to add the phase correction if needed !
    def get_model_prediction(self, x, model=None, coords: dict = None, attrs: dict = None, nofilters: bool = False):
        """
        Get the prediction from the model.
        
        Parameters
        ----------
        x : array_like, dict
            Input features for the model. 
            If an array, it should have shape (n_samples, n_params). 
            If a dict, it should have keys matching the model input names and values as lists/1d-arrays of shape (n_samples,).
        model : FCN
            Trained theory model. If None, the model attribute of the class is used. Defaults to None.
        coords : dict, optional
            Coordinates for the output DataArray. If None, the coordinates of _dataset.y are used. Defaults to None.
        attrs : dict, optional
            Attributes for the output DataArray. If None, the attributes of _dataset.y are used. Defaults to None.
        nofilters : bool, optional
            If True, no filters are applied to the output and the full DataArray is returned. Defaults to False.
        
        Returns
        -------
        xarray.DataArray | np.ndarray
            Model prediction.
        """
        if isinstance(x, dict):
            missing = set(self.x_names) - set(x.keys())
            extra = set(x.keys()) - set(self.x_names)
            if missing:
                raise ValueError(
                    "Input x dictionary keys do not match the model input names. "
                    f"Missing keys: {missing}"
                )
            if extra:
                self.logger.warning(
                    "Input x dictionary contains unexpected keys not used by the model. "
                    f"Unexpected keys: {extra}"
                )
            x = [x[name] for name in self.x_names]
            x = np.asarray(x).T  # Need to transpose to (n_samples, n_params)
        else:
            x = np.asarray(x)  # Ensure x is an array to make torch.Tensor faster
        
        if model is None:
            model = self.model
        
        with torch.no_grad():
            pred = model.get_prediction(torch.Tensor(x.copy()))
            pred = pred.numpy()

        if hasattr(self, 'phase_correction'):
            pred = self.apply_phase_correction(pred)
        
        if coords is None:
            coords = {
                **{k: self._dataset.y.coords[k] for k in self._dataset.y.dims if k in self._dataset.y.attrs['features']}
            }
        if attrs is None:
            attrs = {
                'sample': ['n_pred'],
                'features': self._dataset.y.attrs['features'],
            }

        n_pred = pred.shape[0] if len(pred.shape) > 1 else 1 # Edge case if only one prediction
        coords = {**{'n_pred': np.arange(n_pred)}, **coords} # Add extra coordinate for the number of predictions
        pred = pred.reshape([len(c) for c in coords.values()]) # reshape to the right shape
        pred = xarray.DataArray(
            pred, 
            coords = coords,
            attrs = attrs,
        )

        if nofilters:
            return pred
        
        pred = self.apply_filters(pred)
        pred = self.apply_indices_selection(pred)
        pred = self.flatten_output(pred, self.flat_output_dims)
        
        if self.squeeze_output:
            pred = pred.squeeze()
        if self.numpy_output:
            pred = pred.values
        return pred
        
    def compress_x(self, hods: dict, cosmos: list = cosmo_list) -> tuple:
        """
        Compress the x values from the parameters files.
        
        Parameters
        ----------
        hods : dict
            Dictionary of hods for each cosmology.
        cosmos : list, optional
            List of cosmologies to get from the files. The default is None, which means all cosmologies.
        
        Returns
        -------
        xarray.DataArray
            Compressed x values.
        """
        data_dir = self.paths['param_dir']

        filename = '/pscratch/sd/e/epaillas/emc/AbacusSummit.csv'
        cosmo_params = load_abacus_cosmologies(filename, cosmos, ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld'], mapping={'alpha_s': 'nrun'})
        
        n_hod = len(hods[cosmo_list[0]])  # assumes same number of hods for each cosmology
        x = []
        x_hods = np.load('/pscratch/sd/n/ntbfin/emulator/hods/hod_params.npy', allow_pickle=True).item()
        for cosmo_idx in cosmos:
            x_hod = x_hods[f'c{cosmo_idx:03}']
            x_hod_names = list(x_hod.keys())  # Store the HOD parameter names only once
            x_hod = np.array([x_hod[param] for param in x_hod.keys()]).T
            x_hod = x_hod[hods[cosmo_idx]]
            x_cosmo = [cosmo_params[f'c{cosmo_idx:03}'][param] for param in cosmo_params[f'c{cosmo_idx:03}'].keys()]
            x_cosmo = np.repeat(np.array(x_cosmo).reshape(1, -1), n_hod, axis=0)
            x.append(np.concatenate([x_cosmo, x_hod], axis=1))
            x_cosmo_names = [name for name in cosmo_params[f'c{cosmo_idx:03}'].keys()]
        x = np.concatenate(x)
        x_names = x_cosmo_names + x_hod_names
        order = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'nrun', 'N_ur', 'w0_fld', 'wa_fld', 'logM_cut', 'logM_1', 'sigma', 'alpha', 'kappa', 'alpha_c', 'alpha_s', 's', 'A_cen', 'A_sat', 'B_cen', 'B_sat']
        idx_sorted = [np.where(np.array(x_names) == name)[0][0] for name in order]
        x = x[:, idx_sorted]
        x_names = [x_names[i] for i in idx_sorted]
        x = xarray.DataArray(
            x.reshape(len(cosmos), n_hod, -1),
            coords = {
                'cosmo_idx': cosmos,
                'hod_idx': list(range(n_hod)),
                'parameters': x_names,
            },
            attrs= {
                'sample': ['cosmo_idx', 'hod_idx'],
                'features': ['parameters'],
            },
            name = 'x',
        )
        return x
    
    def compress_emulator_error(self, n_test: int|list, save_to: str = None):
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the emulator error file.
        
        Parameters
        ----------
        n_test : int|list
            Number of test samples or list of indices of the test samples.
        save_to : str
            Path of the directory where to save the emulator error file. If None, the emulator error file is not saved.
            Default is None.
        
        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'emulator_error' and 'emulator_covariance_y' DataArrays.
        """
        emulator_cov_y = self.get_emulator_covariance_y(n_test).unstack().squeeze()
        emulator_error = self.get_emulator_error(n_test).unstack().squeeze()

        emulator_error_dataset = xarray.Dataset(
            data_vars = {
                'emulator_error': emulator_error,
                'emulator_covariance_y': emulator_cov_y,
            }
        )

        if save_to:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(emulator_error_dataset))
        return emulator_error_dataset

    @set_plot_style
    def plot_observable(self, model_params: dict, sample_idx: int = 0, save_fn: str = None):
        """
        Plot the reconstructed galaxy power spectrum multipoles data, model, and residuals.

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
        show_legend = False

        lax[-1].set_xlabel(r'$\textrm{bin index}$', fontsize=15)
        lax[0].set_ylabel(r'${\rm X}$]', fontsize=15)

        data = self.flatten_output(self.y, flat_output_dims=2)[sample_idx] # Enforce 2D flattening and get a single sample
        bin_idx = np.arange(len(data))
        model = self.get_model_prediction(model_params)
        model = self.flatten_output(model, flat_output_dims=2)[0] # Enforce 2D flattening and get the first sample
        cov = self.get_covariance_matrix(volume_factor=64)
        error = np.sqrt(np.diag(cov))

        lax[0].errorbar(bin_idx, data, error, marker='o', ms=4, ls='', 
            color=f'C0', elinewidth=1.0, capsize=None)
        lax[0].plot(bin_idx, model, ls='-', color=f'C0')
        lax[1].plot(bin_idx, (data - model) / error, ls='-', color=f'C0')

        for offset in [-2, 2]: lax[1].axhline(offset, color='k', ls='--')
        lax[1].set_ylabel(r'$\Delta{\rm X} / \sigma_{\rm data}$', fontsize=15)
        lax[1].set_ylim(-4, 4)

        for ax in lax:
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=14)
        if show_legend: lax[0].legend(fontsize=15)

        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving plot to {save_fn}')
        return fig, lax

    @set_plot_style
    def plot_emulator_residuals(self, save_fn: str = None):
        """
        Plot the emulator residuals.
        
        Parameters
        ----------
        save_fn : str
            Filename to save the plot. If None, the plot is not saved.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, numpy.ndarray
            Figure and axes of the plot.
        """

        residuals = self.flatten_output(self.emulator_covariance_y, flat_output_dims=2) # Enforce 2D flattening
        data_cov = self.get_covariance_matrix(volume_factor=64)
        data_err = np.sqrt(np.diag(data_cov))

        fig, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True)

        for res in residuals:
            ax[0].plot(res / data_err, color='gray', alpha=0.3, lw=0.5)

        # summary statistics of the emulator residuals
        for method in ['median', 'stdev']:
            emu_cov = self.get_emulator_covariance_matrix(method=method, diag=True)
            emu_err = np.sqrt(np.diag(emu_cov))

            ax[1].plot(emu_err / data_err, lw=1.0, label=method)

        ax[1].axhline(1.0, color='k', ls=':', lw=0.7)
        ax[1].set_xlabel('bin index', fontsize=13)
        ax[0].set_ylabel(r'$\Delta X / \sigma_{\rm data}$', fontsize=13)
        ax[1].set_ylabel(r'$\textrm{statistic}$', fontsize=13)
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving plot to {save_fn}')
        return fig, ax