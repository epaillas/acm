from .base import BaseObservableBGS

# LHC creation imports
import numpy as np
from pathlib import Path

import logging

class GalaxyCorrelationFunctionMultipoles(BaseObservableBGS):
    """
    Class for the application of the Two-point correlation function statistic of the ACM pipeline 
    to the BGS dataset.
    """
    
    def __init__(self, slice_filters: dict = None, select_filters: dict = None):
        super().__init__(slice_filters=slice_filters, select_filters=select_filters)
        
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
        paths = super().paths
        
        # To create the lhc files
        paths['covariance_statistic_dir'] = f'/pscratch/sd/s/sbouchar/ACM_DS_small/{self.stat_name}/'
        paths['statistic_dir'] = f'/pscratch/sd/s/sbouchar/ACM_DS_data/{self.stat_name}/'
        
        return paths
    
    @property
    def summary_coords_dict(self):
        """
        Defines the default coordinates for the statistics results. 
        """
        coords = super().summary_coords_dict
        
        coords['statistics'] = {
            self.stat_name: {
                'multipoles': [0, 2],
            },
        }
        
        return coords

    #%% LHC creation : Methods to create the LHC data from statistics files
    def create_covariance(self):
        """
        Create the covariance array for the density split statistic.
        """
        outliers_path = Path(self.paths['covariance_statistic_dir']) / 'outliers_idx.npy' # NOTE: Hardcoded !
        outliers_phases = np.load(outliers_path)
        
        logger = logging.getLogger(self.stat_name + '_lhc')
        
        y = []
        for phase in range(3000, 5000):
            data_fn = Path(self.paths['covariance_statistic_dir']) / f'tpcf_c000_ph{phase:04}_hod096.npy' # NOTE: Hardcoded !
            if not data_fn.exists() or phase in outliers_phases:
                logger.warning(f'File {data_fn} not found or phase {phase} is an outlier')
                continue # Skip missing files or outliers
            # logger.info(f'Loading covariance data for phase {phase}') # Noisy
            data = np.load(data_fn, allow_pickle=True).item()
            multipoles = data(ells=(0, 2))
            y.append(np.concatenate(multipoles))
        return np.asarray(y)
    
    def create_lhc(self, save_to: str = None) -> dict:
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the LHC data.
        
        Parameters
        ----------
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
        
        lhc_y = []
        for cosmo_idx in cosmos:
            # logger.info(f'Loading LHC data for cosmo {cosmo_idx}') # Noisy
            for hod in range(n_hod):
                data_fn = Path(statistic_dir) / f'{self.stat_name}_c{cosmo_idx:03d}_hod{hod:03}.npy'
                data = np.load(data_fn, allow_pickle=True).item()
                s, multipoles = data(ells=(0, 2), return_sep=True) # NOTE: Hardcoded !
                lhc_y.append(np.concatenate(multipoles))
        lhc_y = np.array(lhc_y)
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