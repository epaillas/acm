import xarray
import numpy as np
import pandas as pd
from pathlib import Path
from acm.observables import Observable
from acm.utils import get_data_dirs
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.xarray_data import dataset_to_dict


class BaseObservableBGS(Observable):
    """
    Base class for the application of the ACM pipeline to the BGS dataset.
    """
    def __init__(self, flat_output_dims: int = 2, squeeze_output: bool = True, **kwargs):
        paths = kwargs.pop('paths', get_data_dirs('bgs'))
        paths['checkpoint_name'] = 'last.ckpt' # FIXME: Remove this on next model training
        self.n_test = kwargs.pop('n_test', 6*100) # Default number of test samples for BGS
        super().__init__(paths=paths, flat_output_dims=flat_output_dims, squeeze_output=squeeze_output, **kwargs)

    def get_emulator_covariance_y(self, n_test: int|list = None) -> xarray.DataArray|np.ndarray:
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
        observable = self.__class__() # Unfiltered values !!
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
    
    def compress_x(self, cosmos=cosmo_list, n_hod=100) -> xarray.DataArray:
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
        param_dir = self.paths['param_dir']
        
        x = []
        for cosmo_idx in cosmos:
            data_fn = Path(param_dir) / f'AbacusSummit_c{cosmo_idx:03}.csv'
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
            attrs = {
                'sample': ['cosmo_idx', 'hod_idx'],
                'features': ['parameters'],
            },
            name = 'x',
        )
        return x

    def compress_emulator_error(self, n_test: int|list, save_to: str = None) -> xarray.Dataset:
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
        n_test = n_test if n_test else self.n_test
        observable = self.__class__() # Unfiltered values !!
        
        emulator_cov_y = observable.get_emulator_covariance_y(n_test).unstack().squeeze()
        emulator_error = observable.get_emulator_error(n_test).unstack().squeeze()
        
        emulator_error_dataset = xarray.Dataset(
            data_vars = {
                'emulator_covariance_y': emulator_cov_y,
                'emulator_error': emulator_error,
            }
        )

        if save_to:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(emulator_error_dataset))
        return emulator_error_dataset