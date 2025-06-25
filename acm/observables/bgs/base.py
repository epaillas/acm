from acm.observables.observable import Observable
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils import get_data_dirs
import numpy as np

class BaseObservableBGS(Observable):
    """
    Base class for the application of the ACM pipeline to the BGS dataset.
    """
    # NOTE: Define the stat name in the child class !
    # NOTE : Paths and _summary_coords_dict are mutable, so any modification in the child class will affect the parent class unless redefined.
    
    paths = get_data_dirs('bgs')
    
    _summary_coords_dict = {
        'sample_features': {
            'cosmo_idx': cosmo_list,# List of cosmologies index in AbacusSummit
            'hod_number': 100,      # Number of HODs sampled by cosmology
        },
        'param_number': 17,     # Number of parameters in x used to generate the simulations
        'phase_number': 1639,   # Number of phases in the small box simulations after removing outliers phases for any statistic
        'n_test': 6*100,        # List or number of test samples to compute the emulator error
    }
    
    def get_emulator_covariance_y(self, n_test: int|list = None):
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
    
    #%% Compressed files creation
    def compress_x(self):
        """
        From the statistics files for the simulations, compress the x data.
        """
        import pandas as pd
        from pathlib import Path
        
        # Directories
        param_dir = self.paths['param_dir']
        
        # y & bin_values
        cosmos = self.summary_coords_dict['sample_features']['cosmo_idx']
        n_hod = self.summary_coords_dict['sample_features']['hod_number']
        
        x = []
        for cosmo_idx in cosmos:
            data_fn = Path(param_dir) / f'AbacusSummit_c{cosmo_idx:03}.csv'
            x_i = pd.read_csv(data_fn)
            x_names = list(x_i.columns)
            x_names = [name.replace(' ', '').replace('#', '') for name in x_names]
            x.append(x_i.values[:n_hod, :])
        x = np.concatenate(x)
        # assuming all x_names are the same
        
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