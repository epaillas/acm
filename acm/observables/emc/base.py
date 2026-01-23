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
from acm.utils.xarray import dataset_to_dict
from acm.utils.decorators import temporary_class_state

class BaseObservableEMC(Observable):
    """
    Base class for all the observables in the EMC project.
    """
    def __init__(self, flat_output_dims: int = 2, phase_correction: bool = False, **kwargs):
        if phase_correction and hasattr(self, 'compute_phase_correction'):
            self.logger.info('Computing phase correction.')
            self.phase_correction = self.compute_phase_correction()
            
        paths = kwargs.pop('paths', get_data_dirs('emc')) 
        self.n_test = kwargs.pop('n_test', 6*500) # FIXME: Remove this on next file compression ! (backward compatibility)
        super().__init__(paths=paths, flat_output_dims=flat_output_dims, **kwargs)

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
                self.logger.warning('n_test is deprecated. Please provide x_test and y_test in the dataset in the future.')
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
        shape = (n_test,) + y.shape[len(y.attrs['sample']):]
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
        emulator_covariance_y = self.flatten_output(emulator_covariance_y, self.flat_output_dims)
        if 'emulator_covariance_y' in self.select_indices_on:
            emulator_covariance_y = self.apply_indices_selection(emulator_covariance_y)
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
        emulator_error = self.flatten_output(emulator_error, self.flat_output_dims)
        if 'emulator_error' in self.select_indices_on:
            emulator_error = self.apply_indices_selection(emulator_error)
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
        pred = self.flatten_output(pred, self.flat_output_dims)
        pred = self.apply_indices_selection(pred)
        
        if self.squeeze_output:
            pred = pred.squeeze()
        if self.numpy_output:
            pred = pred.values
        return pred

    def get_hod_from_files(self, cosmo_idx: int, phase: int = 0, seed: int = 0) -> np.ndarray:
        """
        Get the HOD indexes from the statistic files for a given phase and seed.
        
        Parameters
        ----------
        cosmo_idx : int
            Cosmology index to read the HOD indexes from.
        phase : int, optional
            Phase index to read the HOD indexes from. Defaults to 0.
        seed : int, optional
            Seed index to read the HOD indexes from. Defaults to 0.
        statistic : str, optional
            Statistic to read the HOD indexes from. Defaults to 'density'.

        Returns
        -------
        np.ndarray
            Array of HOD indexes.
        """
        data_dir = self.paths['hod_dir']
        data_dir = Path(data_dir) / f'c{cosmo_idx:03d}_ph{phase:03d}' / f'seed{seed}'
        hod_idx = [int(fn.stem.lstrip('hod')) for fn in sorted(data_dir.glob('hod*'))]
        return np.array(hod_idx)
        
    def compress_x(
        self,  
        cosmos: list = cosmo_list,
        n_hod: int = None,
        **kwargs
    ) -> xarray.DataArray:
        """
        Compress the x values from the parameters files.
        
        Parameters
        ----------
        cosmos : list, optional
            List of cosmologies to get from the files. The default is None, which means all cosmologies.
        n_hod : int, optional
            Number of HODs to consider per cosmology.
            If None, it is determined from the first cosmology and restricted to that number for all cosmologies. 
            Defaults to None.
        kwargs : dict
            Additional arguments to pass to the get_hod_from_files method.
            
        Returns
        -------
        xarray.DataArray
            Compressed x values.
            
        Raises
        ------
        ValueError
            If the number of HODs for a cosmology is lower than the expected number, 
            as the compression requires all cosmologies to have the same number of HODs.
        """
        # NOTE: Hardcoded paths :/
        cosmo_file = '/pscratch/sd/e/epaillas/emc/AbacusSummit.csv' 
        hod_file = '/pscratch/sd/n/ntbfin/emulator/hods/hod_params.npy'
        
        cosmo_param_names = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
        cosmo_params_mapping = {'alpha_s': 'nrun'}
        cosmo_params = load_abacus_cosmologies(
            cosmo_file, 
            cosmologies = cosmos, 
            parameters = cosmo_param_names, 
            mapping=cosmo_params_mapping,
        )
        # Enforce parameters ordering after mapping
        x_cosmo_names = cosmo_param_names.copy()
        for key, value in cosmo_params_mapping.items():
            x_cosmo_names[x_cosmo_names.index(key)] = value
        hod_params = np.load(hod_file, allow_pickle=True).item()
        
        x = []
        for cosmo_idx in cosmos:
            # HOD parameters
            x_hod = hod_params[f'c{cosmo_idx:03}']
            x_hod_names = list(x_hod.keys())
            x_hod = np.array([x_hod[param] for param in x_hod.keys()]).T # Make x_hod into an array
            
            # Cosmo parameters
            x_cosmo = cosmo_params[f'c{cosmo_idx:03}']
            x_cosmo = np.array([x_cosmo[param] for param in x_cosmo_names]) # Enforce parameters ordering after mapping
            x_cosmo = np.repeat(x_cosmo.reshape(1, -1), x_hod.shape[0], axis=0)
            
            # Full parameters
            x_i = np.concatenate([x_cosmo, x_hod], axis=1)
            
            hod_idx = self.get_hod_from_files(cosmo_idx, **kwargs)
            if n_hod is None:
                n_hod = len(hod_idx) # Determine the number of HODs from the first cosmology
                self.logger.info(f'Number of HODs determined from c{cosmo_idx:03d}: {n_hod}')
            
            # Ensure the number of HODs is as expected
            if len(hod_idx) > n_hod:
                hod_idx = hod_idx[:n_hod] # Restrict to the expected number of HODs
                self.logger.info(f'Number of HODs for c{cosmo_idx:03d} is larger than expected ({len(hod_idx)} > {n_hod}). Restricting to the first {n_hod} HODs.')
            elif len(hod_idx) < n_hod:
                raise ValueError(f'Number of HODs for c{cosmo_idx:03d} is lower than expected ({len(hod_idx)} < {n_hod}). Cannot proceed with compression.')
            
            x.append(x_i[hod_idx, :])
            
        x = np.concatenate(x)
        x_names = x_cosmo_names + x_hod_names
        
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
    
    @temporary_class_state(
        flat_output_dims = 0,
        numpy_output = False,
        squeeze_output = False,
        select_filters = None,
        slice_filters = None,
        select_indices = None,
    )
    def compress_emulator_error(self, save_to: str = None):
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the emulator error file.
        
        Parameters
        ----------
        save_to : str
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
                'emulator_error': emulator_error,
                'emulator_covariance_y': emulator_cov_y,
            }
        )

        if save_to:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(emulator_error_dataset))
        return emulator_error_dataset
