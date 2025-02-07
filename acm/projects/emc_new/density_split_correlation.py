from .base import BaseObservableEMC

# LHC creation imports
import numpy as np
import pandas as pd
from pathlib import Path

import logging

class DensitySplitCorrelationFunctionMultipoles(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge density-split correlation
    function multipoles.
    """
    def __init__(self, select_filters: dict = None, slice_filters: dict = None):
        super().__init__(select_filters=select_filters, slice_filters=slice_filters)
        
    @property
    def stat_name(self) -> str:
        """
        Name of the statistic.
        """
        stat_name = 'dsc_conf'
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
        paths = super().paths
        
        # To create the lhc files
        paths['covariance_statistic_dir'] = f'/pscratch/sd/e/epaillas/emc/covariance_sets/density_split/' # NOTE : Different stat name here
        paths['statistic_dir'] = f'/pscratch/sd/e/epaillas/emc/training_sets/{self.stat_name}/cosmo+hod/z0.5/yuan23_prior/'
        
        return paths

    @property
    def summary_coords_dict(self):
        """
        Defines the default coordinates for the statistics results. 
        """
        coords = super().summary_coords_dict
        coords['statistics'] = {
            self.stat_name: {
                'statistics': ['quantile_data_correlation', 'quantile_correlation'],
                'quantiles': [0, 1, 3, 4],
                'multipoles': [0, 2],
            },
        }
        return coords
    
    #%% LHC creation : Methods to create the LHC data from statistics files
    def create_covariance(self):
        """
        From the statistics files for small AbacusSummit boxes, create the covariance array to store in the lhc file under the `cov_y` key.
        """
        y = []
        for phase in range(3000, 5000):
            multipoles_stat = []
            for stat in ['quantile_data_correlation', 'quantile_correlation']:
                data_dir = Path(self.paths['covariance_statistic_dir']) / f'{stat}/z0.5/yuan23_prior/' # NOTE: Hardcoded !)
                data_fn = data_dir / f'{stat}_ph{phase:03}_hod466.npy' # NOTE: Hardcoded !
                if not data_fn.exists():
                    break
                data = np.load(data_fn, allow_pickle=True)
                multipoles_quantiles = []
                for q in [0, 1, 3, 4]:
                    result = data[q][::4]
                    multipoles = result(ells=(0, 2))
                    multipoles_quantiles.append(np.concatenate(multipoles))
                multipoles_stat.append(np.concatenate(multipoles_quantiles))
            else: # If the loop is not broken
                y.append(np.concatenate(multipoles_stat))
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
        logger = logging.getLogger(self.stat_name + '_lhc')
        
        # Directories
        statistic_dir = self.paths['statistic_dir']

        cosmos = self.summary_coords_dict['cosmo_idx']
        n_hod = self.summary_coords_dict['hod_number']
        
        # LHC_y & bin_values
        lhc_y = []
        for cosmo_idx in cosmos:
            logger.info(f'Loading LHC data for cosmo {cosmo_idx}')
            for hod in range(n_hod):
                multipoles_stat = []
                for stat in ['quantile_data_correlation', 'quantile_correlation']:
                    data_dir = statistic_dir + f'c{cosmo_idx:03}_ph{phase_idx:03}/seed0/' # NOTE: Hardcoded !
                    data_fn = Path(data_dir) / f'{stat}_hod{hod:03}.npy'
                    data = np.load(data_fn, allow_pickle=True)
                    multipoles_quantiles = []
                    for q in [0, 1, 3, 4]:
                        result = data[q][::4]
                        s, multipoles = result(ells=(0, 2), return_sep=True) # NOTE: Hardcoded !
                        multipoles_quantiles.append(np.concatenate(multipoles))
                    multipoles_stat.append(np.concatenate(multipoles_quantiles))
                lhc_y.append(np.concatenate(multipoles_stat))
        lhc_y = np.asarray(lhc_y)
        bin_values = s
        
        # LHC_x
        lhc_x, lhc_x_names = self.create_lhc_x()
        
        logger.info(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
        
        cov_y = self.create_covariance()
        logger.info(f'Loaded covariance with shape: {cov_y.shape}')

        cout = {'bin_values': bin_values, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names, 'cov_y': cov_y}
        
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}_lhc.npy'
            np.save(save_fn, cout)
            logger.info(f'Saving LHC data to {save_fn}')
        
        return cout