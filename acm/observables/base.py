from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from copy import deepcopy
import warnings
from sunbird.data.data_utils import convert_to_summary
import logging

# FIXME : bugs if bin_values is a dict

class BaseClass(ABC):
    """
    Base class for the statistics results handling in the ACM pipeline.
    """
    #%% Special methods
    def __init__(self, select_filters: dict = None, slice_filters: dict = None, select_indices: list = None):
        """
        Parameters
        ----------
        select_filters : dict, optional
            Filters to select values in coordinates. Defaults to None.
        slice_filters : dict, optional
            Filters to slice values in coordinates. Defaults to None.
        select_indices : list, optional
            Indices to select in the flattened data vector. Cannot be used with `select_filters` or `slice_filters`. Defaults to None.
        
        Example
        -------
        ::
        
            slice_filters = {'bin_values': (0, 0.5),} 
            select_filters = {'multipoles': [0, 2],}
        
        
        will return the summary statistics for `0 < bin_values < 0.5` and multipoles 0 and 2
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if bool((select_filters or slice_filters) and select_indices):
            self.logger.warning("Using select_indices with other filters, this will override filtering on the coordinates and flatten the y data arrays")
        
        self.select_filters = select_filters
        self.slice_filters = slice_filters
        
        if select_indices:
            assert type(select_indices) == list, "select_indices should be a list of indices"
            self.select_filters = {} if select_filters is None else select_filters # Otherwise update throws an error
            self.select_filters.update({'flat_bin_idx': select_indices})
        self.select_indices = select_indices
        
        # Checkup
        assert self.stat_name is not None, f"stat_name should be defined in the subclass {self.__class__.__name__}"
        assert self.paths is not None, f"paths should be defined in the subclass {self.__class__.__name__}"
        assert self.summary_coords_dict is not None, f"summary_coords_dict should be defined in the subclass {self.__class__.__name__}"
        
    def __str__(self):
        """
        Returns a string representation of the object (statistic names and slice filters).
        """
        return self.get_save_handle()
    
    #%% Class attributes
    
    stat_name = None
    paths = None
    summary_coords_dict = None
    
    #%% Static and class methods
    @staticmethod
    def get_bin_values(data: dict):
        """
        Returns the bin values from the data.
        
        Parameters
        ----------
        data : dict
            Dictionary with the data.
            If the dictionary contains a key 'bin_values', it will be used as the bin values.
            If not, the first unknown key will be used as the bin values.
            
        Returns
        -------
        array|dict
            Bin values from the data.
            
        Raises
        ------
        ValueError
            If more than one unknown key is found in the data.
        Warning
            If the bin values are assumed from an unknown key.
        """
        
        bin_values = data.get('bin_values', None)
        
        keys = list(data.keys())
        unknown_keys = [k for k in keys if k not in ['x', 'y', 'x_names', 'cov_y', 'bin_values']]
        if bin_values is None:
            if len(unknown_keys) > 1:
                raise ValueError(f"More than one unknown key in the data: {unknown_keys}")
            elif len(unknown_keys) == 1:
                warnings.warn(f"Assuming {unknown_keys[0]} is the bin values") # Can't use logger here...
                bin_values = data[unknown_keys[0]]
            
        # bin_values consistency check if it is a dict (all keys should have the same length)
        if isinstance(bin_values, dict):
            length_dict = {key: len(value) for key, value in bin_values.items()}
            if len(set(length_dict.values())) > 1: # Get the unique lengths
                raise ValueError(f"Bin values have different lengths: {length_dict}")
                
        return bin_values
    
    @classmethod
    def read_file(
        self, 
        statistic: str, 
        data_dir: str,
        load_key: str,
        ignore_key_check: bool = False,
    ) -> np.ndarray:
        """
        Reads the data file and returns the data.
        Expects a .npy file containing a dictionary with keys 'bin_values', 'x', 'y', 'x_names', and 'cov_y'.
        
        Parameters
        ----------
        statistic : str
            Statistic name (filename) to read.
        data_dir : str
            Directory containing the data. Expects to find a file named `{statistic}.npy` in this directory.
        load_key : str
            Which key to return from the file. Can be 'x', 'y', 'x_names', 'bin_values', or 'cov_y', unless ignore_key_check is True.
            Defaults to 'x'.
        ignore_key_check : bool, optional
            Allows to load a custom key from the data file (Not recommended). Defaults to False.
            
        Returns
        -------
        np.ndarray
            Key from the statistic file. 
            
        Raises
        ------
        AssertionError
            If the load_key is not one of ['x', 'y', 'x_names', 'bin_values', 'cov_y'] and
            ignore_key_check is False.
        """
        if not ignore_key_check:
            assert load_key in ['x', 'y', 'x_names', 'bin_values', 'cov_y'], "load_key should be one of ['x', 'y', 'x_names', 'bin_values', 'cov_y']"
        
        data_fn = f"{data_dir}/{statistic}.npy"
        data = np.load(data_fn, allow_pickle=True).item()
        
        if load_key == 'bin_values':
            values = self.get_bin_values(data) # Handle custom keys if they exist (not recommended)
        else:
            values = data[load_key]       
        
        return values
        
    @staticmethod
    def summary_coords(
        coord_type: str,
        summary_coords_dict: dict,
        bin_values = None,
        flattened: bool = False,
    ):
        """
        Finds the summary coordinates for the given statistic and coordinate type.
        Returns a dictionary containing the summary coordinates, in a format that can be used to reshape the data. (see `filter` method)	
        
        Parameters
        ----------
        coord_type : str
            Type of coordinates for which to find the coordinates.
            can be set to : 
            - `'y'` will return the summary coordinates for the LHC data (sample features, statistics and bin_values).
            - `'x'` will return the summary coordinates for the LHC data (sample features, param_idx).
            - `'cov_y'` will return the summary coordinates for the small box data (phase_idx, statistics and bin_values). 
            - `'emulator_error'` will return the summary coordinates for the emulator error data (statistics and bin_values).
            - `'statistic'` will return the summary coordinates for the statistic (statistics and bin_values).
        summary_coords_dict : dict
            Dictionary containing the summary coordinates for the statistic.
            It should also contain the sample features from the data, and the number of parameters and phases.
        bin_values : array_like|dict, optional
            Values of the bins on which the summary statistics are computed. 
            If set to None, the bin_values are not included in the summary coordinates. Defaults to None.
        flattened : bool, optional
            If True, the data is flattened on the statistics coordinates during the filtering. Defaults to False.
            
        Returns
        -------
        dict
            Dictionary containing the summary coordinates for the statistic.
            The keys are the names of the coordinates, and the values are the values of the coordinates.
            
        Raises
        ------
        ValueError
            If the bin_values are empty and flattened is True.
        ValueError
            If the coord_type is not one of ['x', 'y', 'cov_y', 'emulator_error', 'statistic'].

        Note
        ----
        In the sample features, the `hod_number` can be provided to indicate the number of HOD parameters instead of providing `hod_idx`.
        The `hod_number` is then removed from the sample features and the `hod_idx` is added to the sample features.
        """
        
        summary_coords_dict = deepcopy(summary_coords_dict) # Avoid modifying the original dictionary
        
        sample_dict = summary_coords_dict.get('sample_features', {}) # Sample features
        hod_number = sample_dict.pop('hod_number', None) # Number of HOD parameters
        if hod_number is not None:
            sample_dict['hod_idx'] = list(range(hod_number))
        
        param_number = summary_coords_dict.get('param_number', 0) # Number of parameters in x_names (unfiltered)
        param_dict = {
            'param_idx': list(range(param_number)),
        }
        
        phase_number = summary_coords_dict.get('phase_number', 0) # Number of phases in cov_y (unfiltered)
        phase_dict = {
            'phase_idx': list(range(phase_number)),
        }
        
        bin_dict = {}
        if isinstance(bin_values, np.ndarray):
            bin_dict = {'bin_values': bin_values}
        elif isinstance(bin_values, dict):
            bin_dict = bin_values
            
        coord_stat_dict = summary_coords_dict.get('data_features', {}) # Summary coordinates for the statistic
        unflattened_stat_dict = {
            **coord_stat_dict,
            **bin_dict,
        }
        
        dimensions = list(coord_stat_dict.keys())
        n_dim = np.prod([len(coord_stat_dict[d]) for d in dimensions], dtype=int)
        bin_length = len(list(bin_dict.values())[0]) if bin_dict else 0 # Detect empty bin_dict
        if bin_length == 0 and flattened:
            raise ValueError("bin_values is empty, cannot flatten the data")
        flattened_stat_dict = {
            'flat_bin_idx': list(range(n_dim*bin_length)), # Indexes of the flattened y array
        }
        
        stat_dict = flattened_stat_dict if flattened else unflattened_stat_dict
        
        # Handle each coord_type output
        if coord_type == 'x':
            cout = {**sample_dict, **param_dict}
        elif coord_type == 'y':
            cout = {**sample_dict, **stat_dict}
        elif coord_type == 'cov_y':
            cout = {**phase_dict, **stat_dict}
        elif coord_type == 'emulator_error' or coord_type == 'statistic':
            cout = {**stat_dict}
        elif coord_type == 'samples':
            cout = {**sample_dict}
        elif coord_type == 'bin_values':
            cout = {**bin_dict}
        else:
            raise ValueError(f'Unknown coord_type: {coord_type}')
        
        return cout
    
    @staticmethod
    def filter(
        y,
        coords: dict, # Order of the keys is VERY important here !!
        select_filters: dict = None,
        slice_filters: dict = None,
        sample_keys: list|dict = None,
        n_sim: int = None,
    ) -> np.ndarray:
        """
        Filters the data based on the filters provided.
        
        Parameters
        ----------
        y : array_like
            Data to filter.
        coords : dict
            Dictionary containing the summary coordinates for the data. Obtained from `summary_coords`.
        select_filters : dict, optional
            Filters to select values in coordinates. Defaults to None.
        slice_filters : dict, optional
            Filters to slice values in coordinates. Defaults to None.
        sample_keys : list|dict, optional
            Keys to select after filtering to reshape the first dimension of the output.
            A dict can be provided, in which case the keys are used to select the coordinates in the data.
            Defaults to None.
        n_sim : int, optional
            Number of simulations to reshape the first dimension of the output.
            Overriden by the sample_keys if provided. Defaults to None.
            
        Returns
        -------
        np.ndarray
            Filtered data, reshaped in the shape (n_sim, n_statistics), where n_statistics is the concatenated length of the statistics.
            This format is the expected format for the emulator.
            
        Example
        -------
        ::
        
            slice_filters = {'bin_values': (0, 0.5),} 
            select_filters = {'multipoles': [0, 2],}
        
        
        will return the summary statistics for `0 < bin_values < 0.5` and multipoles 0 and 2
        """
        if y is None:
            return None # Nothing to filter, avoid crashes here (handle them later)
        
        # Convert the data to a summary object
        dimensions = list(coords.keys())
        y = y.reshape([len(coords[d]) for d in dimensions])
        y = convert_to_summary(
            data = y,
            dimensions = dimensions, 
            coords = coords, 
            select_filters = select_filters,
            slice_filters = slice_filters
        ) # Filter the data
        
        # Figure out the number of samples after filtering (first dimension of the data)
        n_sim = 1 # assume 1 by default
        if sample_keys is not None:
            sample_keys = list(sample_keys) # just in case it is a dict
            if any(key in y.sizes for key in sample_keys): # Check that the sample keys are in the sizes of the xarray
                n_sim = np.prod([y.sizes[key] for key in sample_keys if key in y.sizes]) # Get the number of samples
        elif n_sim is not None and n_sim > 1: # > 1 because 1 is the default value
            n_sim = n_sim # NOTE : this is just in case we ever need to filter by hand (which should not happen ideally)
        
        # Reshape the data to the expected format 
        if n_sim >1 :
            y = y.values.reshape(n_sim, -1) # Concatenate the data on the last axis 
        else: # Edge case where there is only one simulation
            y = y.values.reshape(-1)
            
        return y 
        
    #%% Methods
    def get_save_handle(self, save_dir: str|Path = None):
        """
        Creates a handle that includes the statistics and filters used.
        This can be used to save anything related to this observable.

        Parameters
        ----------
        save_dir : str
            Directory where the results will be saved.
            If provided, the directory is created if it does not exist.
            If None, the handle is returned as a string.
            Default is None.

        Returns
        -------
        str|Path
            The handle for saving the results, to be completed with the file extension.
            Returned as a Path instance if save_dir is provided as a Path.
        """
        slice_filters = self.slice_filters
        
        statistic_handle = self.stat_name
        if slice_filters:
            for key, value in slice_filters.items():
                statistic_handle += f'_{key}_{value[0]:.2f}-{value[1]:.2f}'
            # TODO : add select filters to the handle ?
        
        if save_dir is None:
            return statistic_handle
        
        # If save_path is provided, make sure it exists
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        cout = Path(save_dir) / f'{statistic_handle}'
        
        if isinstance(save_dir, str):
            return cout.as_posix() # Return as string if save_dir is a string
        return Path(save_dir) / f'{statistic_handle}'

   
