from abc import ABC, abstractmethod
import numpy as np
import torch

from sunbird.emulators import FCN
from acm.data.io_tools import (
    read_lhc, 
    read_covariance_y, 
    summary_coords, 
    read_model, 
    filter,
    read_emulator_error,
    read_emulator_covariance_y,
)

# TODO : improve the docstrings

class BaseObservable(ABC):
    """
    Base class for the statistics results handling in the ACM pipeline.
    """
    def __init__(self, select_filters: dict = None, slice_filters: dict = None):
        """
        Parameters
        ----------
        select_filters : dict, optional
            Filters to select values in coordinates. Defaults to None.
        slice_filters : dict, optional
            Filters to slice values in coordinates. Defaults to None.
        
        Example
        -------
        ::
        
            slice_filters = {'bin_values': (0, 0.5),} 
            select_filters = {'multipoles': [0, 2],}
        
        
        will return the summary statistics for `0 < bin_values < 0.5` and multipoles 0 and 2
        """
        self.select_filters = select_filters
        self.slice_filters = slice_filters
    
    #%% Properties : Define the class properties (name, paths, statistics coordinates, etc.)
    @property
    @abstractmethod
    def stat_name(self) -> str:
        """
        Name of the statistic.
        """
        pass
    
    @property  
    @abstractmethod
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
        pass

    @property
    @abstractmethod
    def summary_coords_dict(self):
        """
        Defines the default coordinates for the statistics results. 
        See `acm.data.default` for a more detailled example.
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
        Read the LHC file data. See `acm.data.io_tools.read_lhc` for more details.
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
    
    @property
    def lhc_x(self):
        """
        Latin hypercube of input features (cosmological and/or HOD parameters)
        """
        lhc_x, lhc_y, lhc_x_names = self.read_lhc(
            select_filters=self.select_filters, 
            slice_filters=self.slice_filters, 
            return_sep=False
            )
        return lhc_x
    
    @property
    def lhc_x_names(self):
        """
        Names of the input features (cosmological and/or HOD parameters).
        """
        lhc_x, lhc_y, lhc_x_names = self.read_lhc(
            select_filters=self.select_filters, 
            slice_filters=self.slice_filters, 
            return_sep=False
            )
        return lhc_x_names
    
    @property
    def lhc_y(self):
        """
        Latin hypercube of output features (tpcf, power spectrum, etc).
        """
        lhc_x, lhc_y, lhc_x_names = self.read_lhc(
            select_filters=self.select_filters, 
            slice_filters=self.slice_filters, 
            return_sep=False
            )
        return lhc_y
    
    @property
    def bin_values(self):
        """
        Bin values for the statistic. (e.g. separation bins for the correlation function)
        """
        bin_values, lhc_x, lhc_y, lhc_x_names = self.read_lhc(
            select_filters=self.select_filters, 
            slice_filters=self.slice_filters, 
            return_sep=True
            )
        return bin_values
    
    @property
    def covariance_y(self):
        """
        Output features from the small AbacusSummit box for covariance
        estimation.
        See `acm.data.io_tools.read_covariance_y` for more details.
        """
        data_dir = self.paths['covariance_dir']
        return read_covariance_y(
            statistic=self.stat_name,
            data_dir=data_dir,
            select_filters=self.select_filters,
            slice_filters=self.slice_filters,
            summary_coords_dict=self.summary_coords_dict,
            )
    
    def get_covariance_matrix(
        self,
        volume_factor: float = 64, 
        prefactor: float = 1):
        """
        Covariance matrix for the statistic. 
        The prefactor is here for corrections if needed, and the volume factor is the volume correction of the boxes.
        """   
        cov_y = self.covariance_y
        prefactor = prefactor / volume_factor
        
        cov = prefactor * np.cov(cov_y, rowvar=False) # rowvar=False : each column is a variable and each row is an observation
        return cov
    
    #%% Model : Methods to interract with the model
    @property
    def model(self):
        """
        Trained theory model.
        """
        return self.get_model(model_fn=None)
        
    def get_model(self, model_fn=None)-> FCN:
        """
        Load trained theory model from checkpoint file.
        
        Parameters
        ----------
        model_fn : str, optional
            Path to the model checkpoint file. If None, the path used is {model_dir}/{stat_name}/{checkpoint_name}. 
            Defaults to None.
        
        Returns
        -------
        FCN
            Trained theory model.
        """
        if model_fn is None:
            model_fn = self.paths['model_dir'] + f'{self.stat_name}/' + self.paths['checkpoint_name']
        return read_model([self.stat_name], model_fn)[0]
    
    def get_model_prediction(self, x, model=None)-> np.ndarray:
        """
        Get the prediction from the model.
        
        Parameters
        ----------
        x : array_like
            Input features.
        model : FCN
            Trained theory model. If None, the model attribute of the class is used. Defaults to None.
        
        Returns
        -------
        array_like
            Model prediction.
        """
        if model is None:
            model = self.model
        x = np.asarray(x) # Ensure x is an array to make torch.Tensor faster
        with torch.no_grad():
            pred = model.get_prediction(torch.Tensor(x))
            pred = pred.numpy()
    
        # Expect output to be in unfiltered format, i.e. with the same dimensions as summary_coords_dict
        bin_values, lhc_x, lhc_y, lhc_x_names = self.read_lhc(return_sep=True) # Get **unfiltered** bin values
        coords = summary_coords(
            self.stat_name, 
            coord_type='emulator_error', # To get only the statistics coordinates
            bin_values=bin_values,
            summary_coords_dict=self.summary_coords_dict
            )
        n_pred = pred.shape[0] if len(pred.shape) > 1 else 1 # Edge case if only one prediction
        coords = {'n_pred': list(range(n_pred)), **coords} # Add extra coordinate for the number of predictions
        pred = filter(pred, coords, self.select_filters, self.slice_filters, n_sim=n_pred)
        
        return pred
    
    @property
    def emulator_error(self):
        """
        Emulator error of the statistic. See `acm.data.io_tools.read_emulator_error` for more details.
        """
        error_dir = self.paths['error_dir']
        return read_emulator_error(
            statistics=[self.stat_name],
            error_dir=error_dir,
            select_filters=self.select_filters,
            slice_filters=self.slice_filters,
            summary_coords_dict=self.summary_coords_dict,
        )
        
    @property
    def emulator_covariance_y(self):
        """
        Emulator covariance of the statistic. See `acm.data.io_tools.read_emulator_covariance_y` for more details.
        """
        error_dir = self.paths['emulator_covariance_dir']
        return read_emulator_covariance_y(
            statistic=self.stat_name,
            error_dir=error_dir,
            select_filters=self.select_filters,
            slice_filters=self.slice_filters,
            summary_coords_dict=self.summary_coords_dict,
        )
    
    def get_emulator_covariance_matrix(self, prefactor: float = 1):
        """
        Emulator covariance matrix for the statistic. The prefactor is here for corrections if needed.
        """
        cov_y = self.emulator_covariance_y
        prefactor = prefactor
        
        cov = prefactor * np.cov(cov_y, rowvar=False)
        return cov
    
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