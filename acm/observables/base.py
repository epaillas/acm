from abc import ABC, abstractmethod
from sunbird.data.data_utils import convert_to_summary
from pathlib import Path
import numpy as np

from acm.data.io_tools import read_lhc, read_covariance_y, read_covariance, read_model


class BaseObservable(ABC):
    """
    Base class for the statistics results handling in the ACM pipeline.
    """
    def __init__(self):
        """
        Expecting a self.stat_name to be defined here ! 
        """
        self.paths = self.set_paths()
        self.summary_coords_dict = self.set_coords()
    
    
    #%% Properties : Define the class properties (paths, statistics coordinates, etc.)
    @property  
    @abstractmethod
    def set_paths(self) -> dict:
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
        pass

    @property
    @abstractmethod
    def set_coords(self):
        """
        Defines the default coordinates for the statistics results. 
        """
        pass


    #%% Data : Methods to read the data
    def read_lhc(
        self,
        select_filters: dict = None,
        slice_filters: dict = None,
        return_sep: bool = False,
    )-> tuple:
        """
        Read the LHC file data. See acm.data.io_tools.read_lhc for more details.
        """
        data_dir = self.paths['lhc_dir']
        return read_lhc(
            statistics = [self.stat_name], 
            data_dir = data_dir,
            select_filters = select_filters,
            slice_filters = slice_filters,
            return_sep = return_sep,
            summary_coords_dict = self.summary_coords_dict
            )
    
    def lhc_x(self, select_filters=None, slice_filters=None):
        """
        Latin hypercube of input features (cosmological and/or HOD parameters)
        """
        lhc_x, lhc_y, lhc_x_names = self.read_lhc(select_filters=select_filters, slice_filters=slice_filters, return_sep=False)
        return lhc_x
    
    def lhc_y(self, select_filters=None, slice_filters=None):
        """
        Latin hypercube of output features (tpcf, power spectrum, etc).
        """
        lhc_x, lhc_y, lhc_x_names = self.read_lhc(select_filters=select_filters, slice_filters=slice_filters, return_sep=False)
        return lhc_y
    
    def bin_values(self, select_filters=None, slice_filters=None):
        """
        Bin values for the statistic. (e.g. separation bins for the correlation function)
        """
        bin_values, lhc_x, lhc_y, lhc_x_names = self.read_lhc(select_filters=select_filters, slice_filters=slice_filters, return_sep=True)
        return bin_values
    
    def covariance_y(self, select_filters=None, slice_filters=None):
        """
        Output features from the small AbacusSummit box for covariance
        estimation.
        """
        return read_covariance_y(
            statistic=self.stat_name,
            data_dir=self.paths['covariance_dir'],
            select_filters=select_filters,
            slice_filters=slice_filters,
            summary_coords_dict=self.summary_coords_dict,
            )
    
    def covariance_matrix(self, select_filters=None, slice_filters=None,  volume_factor=64, prefactor=1):
        """
        Covariance matrix for the statistic.
        """
        cov_y = self.covariance_y(select_filters=select_filters, slice_filters=slice_filters)
        prefactor = prefactor / volume_factor
        
        cov = prefactor * np.cov(cov_y, rowvar=False) # rowvar=False : each column is a variable and each row is an observation
        return cov
    
    def model(self):
        """
        Load trained theory model from checkpoint file.
        """
        return read_model([self.stat_name], self.paths['model_dir'])[0]
    
    
    #%% LHC creation : Methods to create the LHC data from statistics files
    @abstractmethod
    def create_covariance(self):
        """
        From the statistics files for small AbacusSummit boxes, create the covariance array to store in the lhc file under the `cov_y` key.
        """
        pass
    
    @abstractmethod
    def create_lhc(self):
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the LHC data.
        """
        pass