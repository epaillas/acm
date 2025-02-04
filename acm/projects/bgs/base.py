import numpy as np
import torch
from pathlib import Path

from acm.observables.base import BaseObservable
from .default import bgs_summary_coords_dict, bgs_paths

class BaseObservableBGS(BaseObservable):
    """
    Base class for the application of the ACM pipeline to the BGS dataset.
    """
    def __init__(self, slice_filters: dict = None, select_filters: dict = None):
        super().__init__(slice_filters=slice_filters, select_filters=select_filters)
        
    # NOTE: Define the stat name in the child class !
        
    @property
    def paths(self) -> dict:
        """
        Defines the default paths for the statistics results.
        
        Returns
        -------
        dict
            Dictionary with the paths for the statistics results.
            It must contain the following keys:
            - 'lhc_dir' : Directory containing the LHC data.
            - 'covariance_dir' : Directory containing the covariance array of the LHC data.
            - 'model_dir' : Directory where the model is saved.
        """
        return bgs_paths

    @property
    def summary_coords_dict(self):
        """
        Defines the default coordinates for the statistics results. 
        """
        return bgs_summary_coords_dict
    
    #%% Emulator error file
    def create_emulator_covariance(self, n_test: int|list):
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the emulator covariance file.
        """
        # Unfiltered lhc
        lhc_x, lhc_y, lhc_x_names = self.read_lhc() # Unfiltered lhc !
        
        if isinstance(n_test, int):
            idx_test = list(range(n_test))
        else:
            idx_test = n_test
        lhc_test_x = lhc_x[idx_test]
        lhc_test_y = lhc_y[idx_test]
        
        with torch.no_grad():
            pred = self.model.get_prediction(torch.Tensor(lhc_test_x)) # Unfiltered prediction !
            pred = pred.numpy()
        
        diff = lhc_test_y - pred
        return diff
    
    def create_emulator_error(
        self, 
        n_test: int|list, 
        save: bool = False,
        )-> dict:
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the emulator error file.
        """
        emulator_cov_y = self.create_emulator_covariance(n_test)
        emulator_error = np.median(np.abs(emulator_cov_y), axis=0)
        bin_values, lhc_x, lhc_y, lhc_x_names = self.read_lhc(return_sep=True)
        
        emulator_error_dict = {
            'bin_values': bin_values,
            'emulator_error': emulator_error,
            'emulator_cov_y': emulator_cov_y,
        }
        
        if save:
            save_dir = self.paths['error_dir'] + f'{self.stat_name}/'
            Path(save_dir).mkdir(parents=True, exist_ok=True) # Create directory if it does not exist
            save_fn = Path(save_dir) / f'{self.stat_name}_emulator_error.npy'
            np.save(save_fn, emulator_error_dict)
            
        return emulator_error_dict