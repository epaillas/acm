import xarray
import numpy as np
import pandas as pd
from pathlib import Path
from acm.observables import Observable
from acm.utils import get_data_dirs
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.decorators import temporary_class_state
from acm.utils.xarray import dataset_to_dict


class BaseObservableBGS(Observable):
    """
    Base class for the application of the ACM pipeline to the BGS dataset.
    """
    def __init__(self, flat_output_dims: int = 2, squeeze_output: bool = True, **kwargs):
        paths = kwargs.pop('paths', get_data_dirs('bgs'))
        self.n_test = kwargs.pop('n_test', 6*100) # FIXME: Remove this on next file compression !
        super().__init__(paths=paths, flat_output_dims=flat_output_dims, squeeze_output=squeeze_output, **kwargs)

    def get_emulator_covariance_y(self, nofilters: bool = False) -> xarray.DataArray|np.ndarray:
        """
        Returns the unfiltered covariance array of the emulator error of the statistic, with shape (n_test, n_statistics).
        
        Parameters
        ----------
        nofilters : bool, optional
            If True, no filters are applied to the output and the full DataArray is returned. Defaults to False.
            
        Returns
        -------
        np.ndarray
            Array of the emulator covariance array, with shape (n_test, n_features).
        """        
        # Get unfiltered values
        x_test = self._dataset.get('x_test', None)
        y_test = self._dataset.get('y_test', None)
        
        if x_test is None or y_test is None:
            # For backward compatibility
            if hasattr(self, 'n_test'): 
                n_test = self.n_test
                idx_test = range(n_test) if isinstance(n_test, int) else n_test
                x_test = self.flatten_output(self._dataset.x, flat_output_dims=2)[idx_test]
                y_test = self.flatten_output(self._dataset.y, flat_output_dims=2)[idx_test]
            else:
                raise ValueError('x_test and y_test are not available in the dataset. Please provide them or set n_test in the class.')
        
        # Flatten on 2D for indexing
        x_test = self.flatten_output(x_test, flat_output_dims=2)
        y_test = self.flatten_output(y_test, flat_output_dims=2)
        
        prediction = self.get_model_prediction(x_test, nofilters=True) # Unfiltered prediction !
        
        # Flatten on 2D for indexing
        prediction = self.flatten_output(prediction, flat_output_dims=2)
        
        if isinstance(y_test, xarray.DataArray):
            y_test = y_test.values
        if isinstance(prediction, xarray.DataArray):
            prediction = prediction.values
            
        diff = y_test - prediction # NOTE: 2D flattening is done to ensure correct broadcasting here !

        n_test = y_test.shape[0] # Indexing on n_test to prevent filtering issues later on
        y = self._dataset.y.unstack()
        shape = (n_test, ) + y.shape[len(y.attrs['sample']):]
        emulator_covariance_y = xarray.DataArray(
            diff.reshape(shape),
            coords = {
                'n_test': range(n_test),
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
    
    def get_emulator_error(self) -> xarray.DataArray|np.ndarray:
        """
        Returns the unfiltered emulator error of the statistic, with shape (n_statistics, ).
        
        Returns
        -------
        np.ndarray
           Emulator error, with shape (n_features, ).
        """
        emulator_covariance_y = self.get_emulator_covariance_y(nofilters=True) # Unfiltered covariance array !
        
        # Flatten on 2D for indexing
        emulator_covariance_y = self.flatten_output(emulator_covariance_y, flat_output_dims=2)
        
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

    @temporary_class_state(
        flat_output_dims = 0,
        numpy_output = False,
        squeeze_output = False,
        select_filters = None,
        slice_filters = None,
        select_indices = None,
    )
    def compress_emulator_error(self, save_to: str = None) -> xarray.Dataset:
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the emulator error file.
        
        Parameters
        ----------
        save_to : str, optional
            Path of the directory where to save the emulator error file. If None, the emulator error file is not saved.
            Default is None.
        
        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'emulator_error' and 'emulator_covariance_y' DataArrays.
        """
        emulator_cov_y = self.get_emulator_covariance_y()
        emulator_error = self.get_emulator_error()
        
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