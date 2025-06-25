import numpy as np
from .base import BaseClass

class BaseDataObservable(BaseClass):
    """
    Base class to handle the loading and filtering of the data in the ACM pipeline.
    """
    #%% Properties
    @property
    def unfiltered_bin_values(self):
        """
        Unfiltered bin values for the statistic. (e.g. separation bins for the correlation function)
        """
        bin_values = self.read_file(
            statistic = self.stat_name,
            data_dir = self.paths['data_dir'],
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
            data_dir = self.paths['data_dir'],
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
    def x(self):
        """
        Returns the x values of the statistic with shape (n_features, n_params) and filters applied.
        """
        load_key = 'x'
        x = self.read_file(
            statistic = self.stat_name,
            data_dir = self.paths['data_dir'],
            load_key = load_key,
        )
        
        if self.slice_filters or self.select_filters:
            coords = self.summary_coords(
                coord_type = load_key,
                summary_coords_dict = self.summary_coords_dict,
            )
            sample_coords = self.summary_coords(
                coord_type = 'samples',
                summary_coords_dict = self.summary_coords_dict,
            )
            # Filter the data based on the filters provided
            x = self.filter(
                y = x,
                coords = coords,
                select_filters = self.select_filters,
                slice_filters = self.slice_filters,
                sample_keys = sample_coords,
            )
        return x
    
    @property
    def x_names(self):
        """
        Returns the names of the columns in the x values of the statistic.
        """
        x_names = self.read_file(
            statistic = self.stat_name,
            data_dir = self.paths['data_dir'],
            load_key = 'x_names',
        )
        # No filters on x_names for now
        return x_names
    
    @property
    def y(self):
        """
        Returns the y values of the statistic with shape (n_features, n_statistics) and filters applied.
        """
        load_key = 'y'
        y = self.read_file(
            statistic = self.stat_name,
            data_dir = self.paths['data_dir'],
            load_key = load_key,
        )
        
        if self.slice_filters or self.select_filters:
            flattened = bool(self.select_indices) # If we are using the bin_idx filter, we need to flatten the data
            coords = self.summary_coords(
                coord_type = load_key,
                summary_coords_dict = self.summary_coords_dict,
                bin_values = self.unfiltered_bin_values, # Unfiltered bin values
                flattened = flattened,
            )
            sample_coords = self.summary_coords(
                coord_type = 'samples',
                summary_coords_dict = self.summary_coords_dict,
            )
            # Filter the data based on the filters provided
            y = self.filter(
                y = y,
                coords = coords,
                select_filters = self.select_filters,
                slice_filters = self.slice_filters,
                sample_keys = sample_coords,
            )
        return y
  
    @property
    def covariance_y(self):
        """
        Returns the covariance of the y values of the statistic, with shape (n_phases, n_statistics) and filters applied.
        Reads the 'cov_y' key from the data file in 'covariance_dir' if it exists in the paths, otherwise reads from 'data_dir'.
        """
        load_key = 'cov_y'
        data_dir = self.paths.get('covariance_dir', self.paths['data_dir'])
        cov_y = self.read_file(
            statistic = self.stat_name,
            data_dir = data_dir,
            load_key = load_key,
        )
        
        if self.slice_filters or self.select_filters:
            flattened = bool(self.select_indices) # If we are using the bin_idx filter, we need to flatten the data
            coords = self.summary_coords(
                coord_type = load_key,
                summary_coords_dict = self.summary_coords_dict,
                bin_values = self.unfiltered_bin_values, # Unfiltered bin values
                flattened = flattened,
            )
            # Filter the data based on the filters provided
            cov_y = self.filter(
                y = cov_y,
                coords = coords,
                select_filters = self.select_filters,
                slice_filters = self.slice_filters,
                sample_keys = ['phase_idx'],
            )
        return cov_y

    #%% Methods
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

    #%% Compressed files creation
    # Not mandatory to implement, but can be useful to create the LHC data from the statistics files.
    def compress_covariance(self):
        """
        From the statistics files for small AbacusSummit boxes, create the covariance array to store in the compressed file under the `cov_y` key.
        """
        raise NotImplementedError
    
    def compress_data(self):
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the compressed data.
        """
        raise NotImplementedError