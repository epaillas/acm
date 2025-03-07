from .base import BaseObservableEMC

# LHC creation imports
import numpy as np
from pathlib import Path

import logging

class MinimumSpanningTree(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge minimum spanning tree.
    """
    def __init__(self, select_filters: dict = None, slice_filters: dict = None):
        super().__init__(select_filters=select_filters, slice_filters=slice_filters)
        
    @property
    def stat_name(self) -> str:
        """
        Name of the statistic.
        """
        stat_name = 'mst'
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
        paths['covariance_statistic_dir'] = f'/pscratch/sd/k/knaidoo/ACM/MockChallenge/Outputs/'
        paths['statistic_dir'] = f'/pscratch/sd/k/knaidoo/ACM/MockChallenge/Outputs/'
        
        return paths
    
    @property
    def summary_coords_dict(self):
        """
        Defines the default coordinates for the statistics results. 
        """
        coords = super().summary_coords_dict
        coords['hod_number'] = 350
        coords['statistics'] = {} # No coordinates for the statistic
        return coords
    
    
    #%% LHC creation : Methods to create the LHC data from statistics files
    def create_covariance(self):
        """
        From the statistics files for small AbacusSummit boxes, create the covariance array to store in the lhc file under the `cov_y` key.
        """
        y = []
        data_dir = Path(self.paths['covariance_statistic_dir'])
        data_fns = list(data_dir.glob('covariance_mocks_*_10p0.npz')) # NOTE: Hardcoded ! 
        for data_fn in data_fns:
            data = np.load(data_fn)
            tree = np.concatenate([data['yd'], data['yl'], data['yb'], data['ys']])
            y.append(np.concatenate(tree))
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
        data_dir = statistic_dir
        for cosmo_idx in cosmos:
            logger.info(f'Loading LHC data for cosmo {cosmo_idx}')
            for hod in range(n_hod):
                data_fn = Path(data_dir) / f'emulator_{cosmo_idx}_hod_{hod}_smooth_10p0.npz' # NOTE: Hardcoded !
                data = np.load(data_fn)
                tree = np.concatenate([data['yd'], data['yl'], data['yb'], data['ys']])
                lhc_y.append(np.concatenate(tree))
        lhc_y = np.asarray(lhc_y)
        bin_values = np.arange(0, lhc_y.shape[-1]) # Index of the statistic values
        
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