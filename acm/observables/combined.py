import numpy as np
from pathlib import Path
from .observable import Observable

class CombinedModel():
    """
    Class for the combination of theory models.
    """
    def __init__(self, observables: list[Observable]):
        """
        Parameters
        ----------
        observables : list[Observable]
            List of observables to be combined, initialized with their respective filters.
        """
        self.observables = observables
        self.models = [obs.model for obs in self.observables]

    def get_prediction(self, x):
        """
        Get the prediction from the model.
        
        Parameters
        ----------
        x : array_like
            Input features.
        
        Returns
        -------
        array_like
            Model prediction, with respective filters applied to each observable.
        """
        return np.concatenate([obs.get_model_prediction(x) for obs in self.observables], axis=-1)
    
    

class CombinedObservable():
    """
    Class for the combination of observables.
    It has list properties, that allow to access easily self.observables for readibility.
    """
    def __init__(self, observables: list[Observable]):
        """
        Parameters
        ----------
        observables : list[Observable]
            List of observables to be combined, initialized with their respective filters.
        """
        self.observables = observables
        self.slice_filters = [obs.slice_filters for obs in self.observables]
        self.select_filters = [obs.select_filters for obs in self.observables]
        
    def __str__(self):
        """
        Returns a string representation of the object (statistic names and slice filters).
        """
        return self.get_save_handle()
    
    def __getitem__(self, item):
        """
        Allows to access the observables by their statistic name or their index.
        """
        if isinstance(item, int):
            return self.observables[item]
        elif isinstance(item, str):
            try:
                idx = self.stat_name.index(item)
                return self.observables[idx]
            except ValueError: # If the item is not found in the list, raise an error
                KeyError(f"Observable with name {item} not found.")
        else:
            raise TypeError(f"Item must be an int or str, not {type(item)}.")
        
    def __setitem__(self, item, value):
        """
        Allows to set the observable at the given index.
        """
        if not isinstance(value, Observable):
            raise TypeError(f"Value must be a Observable, not {type(value)}.")
        if not isinstance(item, int):
            raise TypeError(f"Item must be an int, not {type(item)}.")
        self.observables[item] = value
            
    def __len__(self):
        """
        Returns the number of observables in the combination.
        """
        return len(self.observables)
    
    def __iter__(self):
        """
        Returns an iterator over the observables in the combination.
        """
        return iter(self.observables)
    
    def __contains__(self, item):
        """
        Checks if the observable with the given statistic name is in the combination.
        """
        return item in self.stat_name
    
    def __reversed__(self):
        """
        Returns a reversed iterator over the observables in the combination.
        """
        return reversed(self.observables)
    
    def __add__(self, other):
        """
        Allows to add two CombinedObservable objects together or to add a new observable to the existing Observable.
        """
        if isinstance(other, CombinedObservable):
            return CombinedObservable(self.observables + other.observables)
        elif isinstance(other, Observable):
            return CombinedObservable(self.observables + [other])
        else:
            raise TypeError(f"Cannot add {type(other)} to CombinedObservable.")
        
        

    @property
    def stat_name(self):
        """
        Name of the statistic.
        """
        return [obs.stat_name for obs in self.observables]

    @property
    def x(self):
        """
        Input features (samples).

        Note: We assume all observable have the same input features, so we just
        return the first from the list.
        """
        return [obs.x for obs in self.observables][0]

    @property
    def x_names(self):
        """
        Names of the input features.

        Note: We assume all observable have the same input features, so we just
        return the first from the list.
        """
        return [obs.x_names for obs in self.observables][0]

    @property
    def y(self):
        """
        Output features (y coordinate of the data), concatenated along the last axis for all observables.
        """
        return np.concatenate([obs.y for obs in self.observables], axis=-1)
    
    @property
    def unfiltered_bin_values(self):
        """
        Unfiltered bin values for the statistics. (e.g. separation bins for the correlation function), 
        concatenated along the last axis for all observables.
        """
        return np.concatenate([obs.unfiltered_bin_values for obs in self.observables], axis=-1)
    
    @property
    def bin_values(self):
        """
        Bin values for the statistics. (e.g. separation bins for the correlation function), 
        concatenated along the last axis for all observables.
        """
        return np.concatenate([obs.bin_values for obs in self.observables], axis=-1)

    @property
    def covariance_y(self):
        """
        Features from small AbacusSummit box for covariance estimation.
        """
        return np.concatenate([obs.covariance_y for obs in self.observables], axis=-1)

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
    
    @property
    def model(self):
        """
        Theory model of the combination of observables. 
        `model.get_prediction(x)` returns the prediction of the combination of observables, 
        with the respective filters applied to each observable.
        """
        return CombinedModel(self.observables)
    
    def get_model_prediction(self, x)-> np.ndarray:
        """
        Get the prediction from the model.
        
        Parameters
        ----------
        x : array_like
            Input features.
            
        Returns
        -------
        array_like
            Model prediction.
        """
        return np.concatenate([obs.get_model_prediction(x) for obs in self.observables], axis=-1)

    @property
    def emulator_error(self):
        """
        Emulator error of the combination of observables.
        """
        return np.concatenate([obs.emulator_error for obs in self.observables], axis=-1)
    
    @property
    def emulator_covariance_y(self):
        """
        Emulator covariance of the combination of observables.
        """
        return np.concatenate([obs.emulator_covariance_y for obs in self.observables], axis=-1)
    
    def get_emulator_covariance_matrix(self, prefactor: float = 1, block_diag: bool = False):
        """
        Emulator covariance matrix for the statistic. The prefactor is here for corrections if needed.
        """
        prefactor = prefactor

        if block_diag:
            from scipy.linalg import block_diag
            cov_blocks = [prefactor * np.cov(obs.emulator_covariance_y, rowvar=False) for obs in self.observables]
            cov = block_diag(*cov_blocks)
            return cov

        cov_y = self.emulator_covariance_y
        cov = prefactor * np.cov(cov_y, rowvar=False)
        return cov
    
    def get_save_handle(self, save_dir: str|Path = None):
        """
        Creates a handle that combines the handles of the observables,
        separated by a '+'. They contain the statistic name and the filters used.
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
        statistic_handles = [
            observable.get_save_handle() for observable in self.observables
        ]
        statistic_handle = '+'.join(statistic_handles)
        
        if save_dir is None:
            return statistic_handle
        
        # If save_path is provided, make sure it exists
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        cout = Path(save_dir) / f'{statistic_handle}'
        
        if isinstance(save_dir, str):
            return cout.as_posix() # Return as string if save_dir is a string
        return Path(save_dir) / f'{statistic_handle}'