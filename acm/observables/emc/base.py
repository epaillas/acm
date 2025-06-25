from acm.observables import Observable
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils import get_data_dirs
import numpy as np
import torch

class BaseObservableEMC(Observable):
    """
    Base class for all the observables in the EMC project.
    """
    def __init__(self, phase_correction: bool = False, **kwargs):
        if phase_correction and hasattr(self, 'compute_phase_correction'):
            self.logger.info('Computing phase correction.')
            self.phase_correction = self.compute_phase_correction()
        super().__init__(**kwargs)
        
    # NOTE: Define the stat name in the child class !
    # NOTE : Paths and _summary_coords_dict are mutable, so any modification in the child class will affect the parent class unless redefined.
    
    paths = get_data_dirs('emc')
    
    _summary_coords_dict = {
        'sample_features': {
            'cosmo_idx': cosmo_list,    # List of cosmologies index in AbacusSummit
            'hod_number': 350,          # Number of HODs sampled by cosmology
        },
        'param_number': 20,     # Number of parameters in x used to generate the simulations
        'phase_number': 1786,   # Number of phases in the small box simulations after removing outliers phases for any statistic
        'n_test': 350*6,        # List or number of test samples to compute the emulator error
    }
    
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
            Array of the emulator covariance array.
        """
        if n_test is None:
            n_test = self.summary_coords_dict['n_test']
            
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
        
        diff = test_y - prediction
        return diff
        
    
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
            Array of the emulator error.
        """
        if n_test is None:
            n_test = self.summary_coords_dict['n_test']
        emulator_cov_y = self.get_emulator_covariance_y(n_test)
        emulator_error = np.median(np.abs(emulator_cov_y), axis=0)
        return emulator_error
    
    # NOTE: Override BaseModelObservable prediction to add the phase correction if needed !
    def get_model_prediction(self, x, model=None)-> np.ndarray:
        if model is None:
            model = self.model
        x = np.asarray(x) # Ensure x is an array to make torch.Tensor faster
        with torch.no_grad():
            pred = model.get_prediction(torch.Tensor(x))
            pred = pred.numpy()
            
        if hasattr(self, 'phase_correction'):
            pred = self.apply_phase_correction(pred)
    
        # Expect output to be in unfiltered format, i.e. with the same dimensions as y
        if self.slice_filters or self.select_filters:
            flattened = bool(self.select_indices) # If we are using the bin_idx filter, we need to flatten the data
            coords = self.summary_coords(
                statistic = self.stat_name,
                coord_type = 'emulator_error',
                summary_coords_dict = self.summary_coords_dict,
                bin_values = self.unfiltered_bin_values, # Unfiltered bin values
                flattened = flattened,
            )
            n_pred = pred.shape[0] if len(pred.shape) > 1 else 1 # Edge case if only one prediction
            coords = {'n_pred': list(range(n_pred)), **coords} # Add extra coordinate for the number of predictions
            # Filter the data based on the filters provided
            pred = self.filter(
                y = pred, 
                coords = coords, 
                select_filters = self.select_filters, 
                slice_filters = self.slice_filters, 
                sample_keys = ['n_pred'],
            )
        return pred
    
    def compress_x(self, cosmos: list = None, n_hod: int = 100) -> tuple:
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
        x : array_like
            Compressed x values.
        x_names : list
            Names of the x values.
        """
        import pandas as pd
        import numpy as np
        
        if cosmos is None:
            cosmos = self.summary_coords_dict['sample_features']['cosmo_idx']
        data_dir = self.paths['param_dir']
        x = []
        for cosmo_idx in cosmos:
            data_fn = data_dir + f'AbacusSummit_c{cosmo_idx:03}.csv' # NOTE: File name format hardcoded !
            x_i = pd.read_csv(data_fn)
            x_names = list(x_i.columns)
            x_names = [name.replace(' ', '').replace('#', '') for name in x_names]
            x.append(x_i.values[:n_hod, :])
        x = np.concatenate(x)
        
        return x, x_names
    
        
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
        dict
            Dictionary containing the emulator error with the following keys:
            - 'bin_values' : Array of the bin values.
            - 'emulator_error' : Array of the emulator error.
            - 'emulator_cov_y' : Array of the emulator covariance matrix.
        """
        from pathlib import Path
        
        emulator_cov_y = self.get_emulator_covariance_y(n_test)
        emulator_error = self.get_emulator_error(n_test)
        bin_values = self.unfiltered_bin_values
        
        emulator_error_dict = {
            'bin_values': bin_values,
            'emulator_error': emulator_error,
            'emulator_cov_y': emulator_cov_y,
        }

        if save_to:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, emulator_error_dict)
        
        return emulator_error_dict