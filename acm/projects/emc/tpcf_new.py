from acm.observables.base import BaseObservable
from .default import emc_summary_coords_dict, emc_paths

# LHC creation imports
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from pycorr import TwoPointCorrelationFunction

import logging
from acm.utils import setup_logging


class GalaxyCorrelationFunctionMultipoles(BaseObservable):
    """
    Class for the Emulator's Mock Challenge galaxy correlation
    function multipoles.
    
    Note
    ----
    The bin_values are the separation bins of the correlation function (s, in Mpc/h).
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
        
        # To create the lhc files
        paths['param_dir'] = f'/pscratch/sd/e/epaillas/emc/cosmo+hod_params/'
        paths['covariance_statistic_dir'] = f'/pscratch/sd/e/epaillas/emc/covariance_sets/{self.stat_name}/z0.5/yuan23_prior'
        paths['statistic_dir'] = f'/pscratch/sd/e/epaillas/emc/training_sets/{self.stat_name}/cosmo+hod_bugfix/z0.5/yuan23_prior/'
        
        # Temporary : For the tests, the lhc data and error dir will be surcharged (TODO : remove this eventually)
        paths['lhc_dir'] = '/pscratch/sd/s/sbouchar/acm/emc/input_data/'
        paths['covariance_dir'] = '/pscratch/sd/s/sbouchar/acm/emc/input_data/'
        paths['error_dir'] = '/pscratch/sd/s/sbouchar/acm/emc/emulator_error/'
        paths['emulator_covariance_dir'] = '/pscratch/sd/s/sbouchar/acm/emc/emulator_error/'
        paths['save_dir'] = '/pscratch/sd/s/sbouchar/acm/emc/emulator_error/'
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
        y = []
        data_dir = Path(self.paths['covariance_statistic_dir'])
        data_fns = list(data_dir.glob('tpcf_ph*_hod466.npy')) # NOTE: Hardcoded ! 
        for data_fn in data_fns:
            data = TwoPointCorrelationFunction.load(data_fn)[::4]
            multipoles = data(ells=(0, 2))
            y.append(np.concatenate(multipoles))
        return np.asarray(y)
    
    def create_lhc(self, phase_idx: int = 0, save_to: str = None) -> dict:
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the LHC data.
        
        Parameters
        ----------
        phase_idx : int
            Index of the phase to consider in the statistics files. Default is 0.
        save_to : str
            Path of the directory where to save the LHC data. If None, the LHC data is not saved.
            Default is None.
            
        Returns
        -------
        dict
            Dictionary containing the LHC data with the following keys:
            - 'bin_values' : Array of the bin values.
            - 'lhc_x' : Array of the parameters used to generate the simulations.
            - 'lhc_y' : Array of the statistics values.
            - 'lhc_x_names' : List of the names of the parameters.
            - 'cov_y' : Array of the covariance matrix of the statistics values
        """
        # Logging
        setup_logging()
        logger = logging.getLogger(self.stat_name + '_lhc')
        
        # Directories
        param_dir = self.paths['param_dir']
        statistic_dir = self.paths['statistic_dir']
        
        # LHC_y & bin_values
        cosmos = self.summary_coords_dict['cosmo_idx']
        n_hod = self.summary_coords_dict['hod_number']
        
        lhc_y = []
        for cosmo_idx in cosmos:
            logger.info(f'Loading LHC data for cosmo {cosmo_idx}')
            data_dir = statistic_dir + f'c{cosmo_idx:03}_ph{phase_idx:03}/seed0/' # NOTE: Hardcoded !
            for hod in range(n_hod):
                data_fn = Path(data_dir) / f'tpcf_hod{hod:03}.npy'
                data = TwoPointCorrelationFunction.load(data_fn)[::4]
                s, multipoles = data(ells=(0, 2), return_sep=True)
                lhc_y.append(np.concatenate(multipoles))
        lhc_y = np.asarray(lhc_y)
        bin_values = s
        
        # LHC_x
        lhc_x = []
        for cosmo_idx in cosmos:
            data_fn = Path(param_dir) / f'AbacusSummit_c{cosmo_idx:03}.csv'
            lhc_x_i = pd.read_csv(data_fn)
            lhc_x_names = list(lhc_x_i.columns)
            lhc_x_names = [name.replace(' ', '').replace('#', '') for name in lhc_x_names]
            lhc_x.append(lhc_x_i.values[:n_hod, :])
        lhc_x = np.concatenate(lhc_x)
        # assuming all lhc_x_names are the same
        
        logger.info(f'Loaded 2PCF LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
        
        cov_y = self.create_covariance()
        print(f'Loaded 2PCF covariance with shape: {cov_y.shape}')

        cout = {'bin_values': bin_values, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names, 'cov_y': cov_y}
        
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}_lhc.npy'
            np.save(save_fn, cout)
            logger.info(f'Saving LHC data to {save_fn}')
        
        return cout
    
    #%% Emulator creation : Methods to create the emulator error file from the model and the LHC data
    def create_emulator_covariance(self, n_test: int|list):
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the emulator covariance file.
        Assuming the model is already trained and the LHC file is created.
        
        Parameters
        ----------
        n_test : int|list
            Number of test samples or list of indices of the test samples.
            
        Returns
        -------
        np.ndarray
            Array of the emulator covariance matrix.
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
    
    def create_emulator_error(self, n_test:int|list, save_to: str = None):
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
        emulator_cov_y = self.create_emulator_covariance(n_test)
        emulator_error = np.median(np.abs(emulator_cov_y), axis=0)
        bin_values, lhc_x, lhc_y, lhc_x_names = self.read_lhc(return_sep=True)
        
        emulator_error_dict = {
            'bin_values': bin_values,
            'emulator_error': emulator_error,
            'emulator_cov_y': emulator_cov_y,
        }

        if save_to:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}_emulator_error.npy'
            np.save(save_fn, emulator_error_dict)
        
        return emulator_error_dict
        