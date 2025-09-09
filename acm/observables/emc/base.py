import torch
import xarray
import numpy as np
import pandas as pd
from pathlib import Path
from acm.observables import Observable
from acm.utils import get_data_dirs
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.xarray_data import dataset_to_dict

class BaseObservableEMC(Observable):
    """
    Base class for all the observables in the EMC project.
    """
    def __init__(self, flat_output_dims: int = 2, phase_correction: bool = False, **kwargs):
        if phase_correction and hasattr(self, 'compute_phase_correction'):
            self.logger.info('Computing phase correction.')
            self.phase_correction = self.compute_phase_correction()
            
        paths = kwargs.pop('paths', get_data_dirs('emc'))
        self.n_test = kwargs.pop('n_test', 6*350) # Default number of test samples for EMC
        super().__init__(paths=paths, flat_output_dims=flat_output_dims, **kwargs)
    
    def get_emulator_covariance_y(self, n_test: int|list = None) -> np.ndarray:
        """
        Returns the unfiltered covariance array of the emulator error of the statistic, with shape (n_test, n_statistics).
        
        Parameters
        ----------
        n_test : int|list, optional
            Number of test samples or list of indices of the test samples. The default is None.
        
        Returns
        -------
        np.ndarray
            Array of the emulator covariance array, with shape (n_test, n_features).
        """
        n_test = n_test if n_test else self.n_test
        observable = self.__class__(
            # phase_correction = hasattr(self, 'phase_correction'), # TODO : check if this needs to be passed here ?
        ) # Unfiltered values !!
        x = observable.x
        y = observable.y
        
        if isinstance(n_test, int):
            idx_test = list(range(n_test))
        elif isinstance(n_test, list):
            idx_test = list(set(n_test))
        
        test_x = x[idx_test]
        test_y = y[idx_test]
        
        prediction = observable.get_model_prediction(test_x) # Unfiltered prediction !
        
        if isinstance(test_y, xarray.DataArray):
            test_y = test_y.values
        if isinstance(prediction, xarray.DataArray):
            prediction = prediction.values
            
        diff = test_y - prediction
        
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
        emulator_covariance_y = self.apply_filters(emulator_covariance_y)
        if 'emulator_covariance_y' in self.select_indices_on:
            emulator_covariance_y = self.apply_indices_selection(emulator_covariance_y)
        emulator_covariance_y = self.flatten_output(emulator_covariance_y)
        if self.squeeze_output:
            emulator_covariance_y = emulator_covariance_y.squeeze()
        if self.numpy_output:
            emulator_covariance_y = emulator_covariance_y.values
        return emulator_covariance_y
        
    def get_emulator_error(self, n_test: int|list = None) -> dict:
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
        observable = self.__class__() # Unfiltered values !!
        emulator_covariance_y = observable.get_emulator_covariance_y(n_test)
        emulator_error = np.median(np.abs(emulator_covariance_y), axis=0)

        y = observable.y.unstack()
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
        emulator_error = self.flatten_output(emulator_error)
        if self.squeeze_output:
            emulator_error = emulator_error.squeeze()
        if self.numpy_output:
            emulator_error = emulator_error.values
        return emulator_error
    
    # NOTE: Override Observable prediction to add the phase correction if needed !
    def get_model_prediction(self, x, model=None, coords=None, attrs=None):
        """
        Get the prediction from the model.
        
        Parameters
        ----------
        x : array_like
            Input features.
        model : FCN
            Trained theory model. If None, the model attribute of the class is used. Defaults to None.
        coords : dict, optional
            Coordinates for the output DataArray. If None, the coordinates of the _dataset y are used. Defaults to None.
        attrs : dict, optional
            Attributes for the output DataArray. If None, the attributes of the _dataset y are used. Defaults to None.
        
        Returns
        -------
        xarray.DataArray | np.ndarray
            Model prediction.
        """
        if model is None:
            model = self.model
        x = np.asarray(x) # Ensure x is an array to make torch.Tensor faster
        with torch.no_grad():
            pred = model.get_prediction(torch.Tensor(x))
            pred = pred.numpy()
            
        if hasattr(self, 'phase_correction'):
            pred = self.apply_phase_correction(pred)
    
        if coords is None:
            coords = {
                **{k: v for k, v in self._dataset.y.coords if k in self._dataset.y.attrs['features']}
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
        
        pred = self.apply_filters(pred)
        pred = self.apply_indices_selection(pred)
        pred = self.flatten_output(pred)
        
        if self.squeeze_output:
            pred = pred.squeeze()
        if self.numpy_output:
            pred = pred.values
        return pred
    
    def compress_x(self, cosmos: list = cosmo_list, n_hod: int = 100) -> tuple:
        """
        Compress the x values from the parameters files.
        
        Parameters
        ----------
        cosmos : list, optional
            List of cosmologies to get from the files. The default is None, which means all cosmologies.
        n_hod : int, optional
            Number of HODs to get from the files. The default is 100.
        
        Returns
        -------
        xarray.DataArray
            Compressed x values.
        """
        data_dir = self.paths['param_dir']
        
        x = []
        for cosmo_idx in cosmos:
            data_fn = data_dir + f'AbacusSummit_c{cosmo_idx:03}.csv' # NOTE: File name format hardcoded !
            x_i = pd.read_csv(data_fn)
            x_names = list(x_i.columns)
            x_names = [name.replace(' ', '').replace('#', '') for name in x_names]
            x.append(x_i.values[:n_hod, :])
        x = np.concatenate(x)
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