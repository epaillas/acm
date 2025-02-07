from .base import BaseObservableEMC

# read lhc import
from acm.data.io_tools import lhc_fnames, get_bin_values, summary_coords, filter

# LHC creation imports
import numpy as np
import pandas as pd
from pathlib import Path

import logging

class GalaxyBispectrumMultipoles(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge bispectrum.
    
    Note
    ----
    The bin_values have two definitions here ! 
    They are either the index of the bin_values array (when calling `self.get_bin_values(return_index=True)`), or the bin values themselves (k123).
    The data is computed using the bin indexes, to match the shape of the statistics, but calling `self.bin_values` will return the bin values.
    
    Warning
    -------
    The filters on the bin_values are applied on the bin index, not on the bin values themselves ! 
    (e.g. `{'bin_values': (0, 12)}` filter the bin indexes from 0 to 12, not the bin values)
    """
    def __init__(self, select_filters: dict = None, slice_filters: dict = None):
        super().__init__(select_filters=select_filters, slice_filters=slice_filters)
        
    @property
    def stat_name(self) -> str:
        """
        Name of the statistic.
        """
        stat_name = 'bispectrum'
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
        paths['covariance_statistic_dir'] = f'/pscratch/sd/e/epaillas/emc/v1.1/abacus/covariance_sets/small_box/raw/{self.stat_name}/kmax0.25_dk0.02/'
        paths['statistic_dir'] = f'/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/raw/{self.stat_name}/kmin0.013_kmax0.253_dk0.020/'
        
        return paths
    
    @property
    def summary_coords_dict(self):
        """
        Defines the default coordinates for the statistics results. 
        """        
        coords = super().summary_coords_dict
        coords['hod_number'] = 350
        return coords
    
    # Because of the shape of the bin_values, we need to redefine the read_lhc method
    def read_lhc(
        self,
        select_filters: dict = None,
        slice_filters: dict = None,
        return_sep: bool = False,
        return_true_sep: bool = False,
    )-> tuple:
        """
        Read the LHC data from the statistics files.

        Parameters
        ----------
        select_filters : dict, optional
            Filters to select values in coordinates. Defaults to None.
        slice_filters : dict, optional
            Filters to slice values in coordinates. Defaults to None.
        return_sep : bool, optional
            Wether to return the bin_values array. It will return the bin index array (the index of the bin_values array) if True. Defaults to False.
        return_true_sep : bool, optional
            Wether to return the bin_values array instead of the bin index array. It overrides `return_sep`. Defaults to False.


        Returns
        -------
        tuple
            Tuple of arrays with the input features, output features and bin_values array if `return_sep` is True : 
            `(bin_index), lhc_x, lhc_y, lhc_x_names`, 
            
            Tuple of arrays with the input features, output features and bin_values array if `return_true_sep` is True :
            `(bin_values), lhc_x, lhc_y, lhc_x_names`.
            
        Note
        ----
        The bin_index array is the index of the bin_values array. It is used to slice the lhc_x and lhc_y arrays, and is the one used in the filters !
        It is computed from `list(range(len(bin_values.prod(axis=0))))`.
            
        Example
        -------
        ::
        
            slice_filters = {'bin_values': (0, 0.5),} 
            select_filters = {'multipoles': [0, 2],}
        
        
        will return the summary statistics for `0 < bin_values < 0.5` and multipoles 0 and 2
        """
        # Class attributes
        statistic = self.stat_name
        data_dir = self.paths['lhc_dir']
        summary_coords_dict = self.summary_coords_dict
        
        data_fn = lhc_fnames(statistic, data_dir)
        data = np.load(data_fn, allow_pickle=True).item()
        
        bin_values = get_bin_values(data) # 1 line !
        bin_values_full = data['bin_values_full'] # 3 lines !
        lhc_x_names = data['lhc_x_names']
        lhc_x = data['lhc_x']
        lhc_y = data['lhc_y']
        
        # If filters are provided, filter the data
        if select_filters or slice_filters: 
            # Get the summary coordinates for the given statistic
            coords_x = summary_coords(statistic, coord_type='lhc_x', bin_values=bin_values, summary_coords_dict=summary_coords_dict)
            coords_y = summary_coords(statistic, coord_type='lhc_y', bin_values=bin_values, summary_coords_dict=summary_coords_dict)
            # lhc_x can also be filtered ! (for example, to select only some cosmologies)
            lhc_x = filter(lhc_x, coords_x, select_filters, slice_filters)
            lhc_y = filter(lhc_y, coords_y, select_filters, slice_filters)
            # Filter the bin_values too
            if bin_values is not None:
                unfiltered_bin_values = bin_values
                coords_bin = {'bin_values': bin_values}
                bin_values = filter(bin_values, coords_bin, select_filters, slice_filters)
                
                # TODO : Filter the true bin_values too
                mask = np.isin(unfiltered_bin_values, bin_values) # Mask to filter the bin_values_full
                new_bin_values_full = []
                for i in range(len(bin_values_full)): # Apply mask on each line of bin_values_full
                    new_bin_values_full.append(bin_values_full[i][mask])
                bin_values_full = np.asarray(new_bin_values_full)
                
                
        toret = (lhc_x, lhc_y, lhc_x_names)
    
        if return_true_sep:
            toret = (bin_values_full, *toret)
        elif return_sep:
            toret = (bin_values, *toret)
        return toret
            
    def get_bin_values(self, return_index: bool = False):
        """
        Return the bin values array or the bin index array.
        
        Parameters
        ----------
        return_index : bool
            If True, return the bin index array. If False, return the bin values array.
            Default is False.
        
        Returns
        -------
        array
            Array of the bin values or the bin index.
        """
        return_true_sep = not return_index # Inverse the return_sep
        bin_values, lhc_x, lhc_y, lhc_x_names = self.read_lhc(
            slice_filters=self.slice_filters,
            select_filters=self.select_filters,
            return_true_sep=return_true_sep,
            return_sep=True, 
            )
        return bin_values
    
    @property
    def bin_values(self):
        """
        Return the bin values array.
        """
        return self.get_bin_values()
    
    #%% LHC creation : Methods to create the LHC data from statistics files
    def create_covariance(self):
        """
        From the statistics files for small AbacusSummit boxes, create the covariance array to store in the lhc file under the `cov_y` key.
        """
        data_dir = Path(self.paths['covariance_statistic_dir'])
        data_fns = list(data_dir.glob('bispectrum_ph*_hod466.npy')) # NOTE: Hardcoded ! 
        y = []
        for data_fn in data_fns:
            data = np.load(data_fn, allow_pickle=True).item()
            k123 = data['k123']
            bk = data['bk']
            weight = k123.prod(axis=0) / 1e5
            multipoles = np.concatenate([weight * bk[f'b{i}'] for i in [0, 2]])
            y.append(multipoles)
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
        ells = [0, 2] # NOTE: Hardcoded !
        for cosmo_idx in cosmos:
            logger.info(f'Loading LHC data for cosmo {cosmo_idx}')
            data_dir = statistic_dir + f'c{cosmo_idx:03}_ph{phase_idx:03}/seed0/' # NOTE: Hardcoded ! 
            for hod in range(n_hod):
                data_fn = Path(data_dir) / f'bispectrum_hod{hod:03d}.npy'
                data = np.load(data_fn, allow_pickle=True).item()
                k123 = data['k123']
                bin_values = list(range(len(k123.prod(axis=0))))
                bk = data['bk']
                weight = k123.prod(axis=0) / 1e5
                multipoles = np.concatenate([weight * bk[f'b{i}'] for i in ells])
                bin_index = len(multipoles)
                lhc_y.append(multipoles)
        lhc_y = np.asarray(lhc_y)
        bin_values = np.asarray(bin_values)
        bin_values_full = k123
    
        # LHC_x
        lhc_x, lhc_x_names = self.create_lhc_x()
        
        logger.info(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
        
        cov_y = self.create_covariance()
        logger.info(f'Loaded covariance with shape: {cov_y.shape}')

        cout = {'bin_values': bin_values, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names, 'cov_y': cov_y, 'bin_values_full': bin_values_full}
        
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}_lhc.npy'
            np.save(save_fn, cout)
            logger.info(f'Saving LHC data to {save_fn}')
        
        return cout