import numpy as np
import torch
from sunbird.emulators import FCN
from .base import BaseClass

class BaseModelObservable(BaseClass):
    """
    Base class to handle the loading and filtering of the models in the ACM pipeline.
    """
    def __init__(self, select_filters = None, slice_filters = None, select_indices = None):
        super().__init__(select_filters=select_filters, slice_filters=slice_filters, select_indices=select_indices)
        
        # Set the model here instead as property, to avoid loading it multiple times
        try:
            self.model = self.load_model() 
        except Exception as e: # handle the case where the model checkpoint is not found
            self.logger.warning(f"{e}, model will be undefined. If you are training a new model, this is expected.")
        
    #%% Properties
    @property
    def unfiltered_bin_values(self):
        """
        Unfiltered bin values for the statistic. (e.g. separation bins for the correlation function)
        """
        bin_values = self.read_file(
            statistic = self.stat_name,
            data_dir = self.paths['error_dir'],
            load_key = 'bin_values',
        )
        return bin_values
    
    @property
    def bin_values(self):
        """
        Bin values for the statistic, with filters applied. (e.g. separation bins for the correlation function)
        """
        if self.select_indices:
            self.logger.warning("Using flat_bin_idx filter, returning the unfiltered bin values")
            return self.unfiltered_bin_values
        
        load_key = 'bin_values'
        bin_values = self.read_file(
            statistic = self.stat_name,
            data_dir = self.paths['error_dir'],
            load_key = load_key,
        )
        
        if self.slice_filters or self.select_filters:
            coords = self.summary_coords(
                coord_type = load_key,
                summary_coords_dict = self.summary_coords_dict,
                bin_values = self.unfiltered_bin_values, # Unfiltered bin values (just in case)
            )
            # Filter the data based on the filters provided
            bin_values = self.filter(
                y = bin_values,
                coords = coords,
                select_filters = self.select_filters,
                slice_filters = self.slice_filters,
            )
        return bin_values
    
    @property
    def emulator_error(self):
        """
        Returns the emulator error of the statistic, with shape (n_statistics, ) and filters applied.
        Reads the emulator error from the error_dir if it is provided, otherwise uses the get_emulator_error method if implemented.
        """
        load_key = 'emulator_error'
        data_dir = self.paths.get('error_dir', None)
        if data_dir is not None:
            emulator_error = self.read_file(
                statistic = self.stat_name,
                data_dir = data_dir,
                load_key = load_key,
                ignore_key_check = True,
            )
        elif hasattr(self, 'get_emulator_error'):
            emulator_error = self.get_emulator_error()
        else:
            raise ValueError(f"Cannot find emulator error for {self.stat_name}. Please provide error_dir or implement get_emulator_error method.")
        
        if self.slice_filters or self.select_filters:
            flattened = bool(self.select_indices) # If we are using the bin_idx filter, we need to flatten the data
            coords = self.summary_coords(
                coord_type = load_key,
                summary_coords_dict = self.summary_coords_dict,
                bin_values = self.unfiltered_bin_values, # Unfiltered bin values
                flattened = flattened,
            )
            # Filter the data based on the filters provided
            emulator_error = self.filter(
                y = emulator_error,
                coords = coords,
                select_filters = self.select_filters,
                slice_filters = self.slice_filters,
            )
        return emulator_error
    
    @property
    def emulator_covariance_y(self):
        """
        Returns the covariance of the emulator error of the statistic, with shape (n_test, n_statistics) and filters applied.
        Reads the emulator covariance from the error_dir if it is provided, otherwise uses the get_emulator_covariance_y method if implemented.
        """
        data_dir = self.paths.get('error_dir', None)
        if data_dir is not None:
            y = self.read_file(
                statistic = self.stat_name,
                data_dir = data_dir,
                load_key = 'emulator_cov_y',
                ignore_key_check = True,
            )
        elif hasattr(self, 'get_emulator_covariance_y'):
            y = self.get_emulator_covariance_y()
        else:
            raise ValueError(f"Cannot find emulator covariance for {self.stat_name}. Please provide error_dir or implement get_emulator_covariance_y method.")
        
        if self.slice_filters or self.select_filters:
            flattened = bool(self.select_indices)
            coords = self.summary_coords(
                coord_type = 'emulator_error',
                summary_coords_dict = self.summary_coords_dict,
                bin_values = self.unfiltered_bin_values, # Unfiltered bin values
                flattened = flattened,
            )
            coords = {'n_test': list(range(len(y))), **coords} # Add the n_test dimension to the coords (to allow reshaping in filter)
            # Filter the data based on the filters provided
            y = self.filter(
                y = y,
                coords = coords,
                select_filters = self.select_filters,
                slice_filters = self.slice_filters,
                sample_keys = ['n_test'],
            )
        return y
    
    @property
    def checkpoint_fn(self):
        """
        Path to the checkpoint file of the model, constructed from the paths and the statistic name.
        """
        return self.paths['model_dir'] + f'{self.stat_name}/' + self.paths['checkpoint_name'] # FIXME : Update this format later
    
    #%% Methods
    def load_model(self, checkpoint_fn: str = None) -> FCN:
        """
        Trained theory model.
        """
        if checkpoint_fn is None:
            checkpoint_fn = self.checkpoint_fn
        
        # Load the model
        model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
        model.eval().to('cpu')
        if self.stat_name.startswith('minkowski'):
            from sunbird.data.transforms_array import WeiLiuInputTransform, WeiLiuOutputTransForm
            model.transform_output = WeiLiuOutputTransForm()
            model.transform_input = WeiLiuInputTransform()
        
        return model
    
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
    
        # Expect output to be in unfiltered format, i.e. with the same dimensions as y
        if self.slice_filters or self.select_filters:
            flattened = bool(self.select_indices) # If we are using the bin_idx filter, we need to flatten the data
            coords = self.summary_coords(
                coord_type = 'emulator_error',
                summary_coords_dict = self.summary_coords_dict,
                bin_values = self.unfiltered_bin_values, # Unfiltered bin values
                flattened = flattened,
            )
            n_pred = pred.shape[0] if len(pred.shape) > 1 else 1 # Edge case if only one prediction
            coords = {'n_pred': list(range(n_pred)), **coords} # Add extra coordinate for the number of predictions
            # Filter the data based on the filters provided
            pred = self.filter(
                y = pred, 
                coords = coords, 
                select_filters = self.select_filters, 
                slice_filters = self.slice_filters, 
                sample_keys = ['n_pred'],
            )
        return pred
    
    def get_emulator_covariance_matrix(self, prefactor: float = 1):
        """
        Emulator covariance matrix for the statistic. The prefactor is here for corrections if needed.
        """
        cov_y = self.emulator_covariance_y
        prefactor = prefactor
        
        cov = prefactor * np.cov(cov_y, rowvar=False)
        return cov
    
    #%% Compressed files creation
    def compress_emulator_covariance(self):
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the emulator covariance file.
        """
        raise NotImplementedError
    
    def compress_emulator_error(self):
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the emulator error file.
        """
        raise NotImplementedError