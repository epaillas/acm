from acm.observables.base import BaseObservable
from .default import emc_summary_coords_dict, emc_paths

import numpy as np
from acm.data.io_tools import emulator_error_fnames, get_bin_values, summary_coords, filter

class GalaxyCorrelationFunctionMultipoles(BaseObservable):
    """
    Class for the Emulator's Mock Challenge galaxy correlation
    function multipoles.
    """
    def __init__(self, select_filters: dict = None, slice_filters: dict = None):
        super().__init__(select_filters, slice_filters)
        
    @property
    def stat_name(self) -> str:
        """
        Name of the statistic.
        """
        stat_name = 'tpcf'
        return stat_name
    
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
        paths = emc_paths
        # expecting model_fn = model_path/stat_name/checkpoint_name
        paths['checkpoint_name'] = 'cosmo+hod/optuna_log/last-v54.ckpt' 
        return paths

    @property
    def summary_coords_dict(self):
        """
        Defines the default coordinates for the statistics results. 
        """
        return emc_summary_coords_dict
    
    # NOTE: Right now, the emulator files don't contain the emulator covariance array
    # This will cause self.emulator_covariance_y and self.get_emulator_covariance_matrix() 
    # to raise an error
    
    # TODO : redefine the lhc and error files trough the creation functions
    #%% LHC creation : Methods to create the LHC data from statistics files
    # Not mandatory to implement, but can be useful to create the LHC data from the statistics files.
    def create_covariance(self):
        """
        From the statistics files for small AbacusSummit boxes, create the covariance array to store in the lhc file under the `cov_y` key.
        """
        raise NotImplementedError
    
    def create_lhc(self):
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the LHC data.
        """
        raise NotImplementedError
    
    #%% Emulator creation : Methods to create the emulator error file from the model and the LHC data
    
    def create_emulator_covariance(self):
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the emulator covariance file.
        """
        raise NotImplementedError
    
    def create_emulator_error(self):
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the emulator error file.
        """
        raise NotImplementedError