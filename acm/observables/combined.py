import numpy as np
from .base import BaseObservable

class CombinedModel():
    """
    Class for the combination of theory models.
    """
    def __init__(self, observables: list[BaseObservable]):
        """
        Parameters
        ----------
        observables : list[BaseObservable]
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

class BaseCombinedObservable(BaseObservable):
    """
    Class for the combination of observables.
    """
    def __init__(self, observables: list[BaseObservable]):
        self.observables = observables
        self.slice_filters = [obs.slice_filters for obs in self.observables]
        self.select_filters = [obs.select_filters for obs in self.observables]

    @property
    def stat_name(self):
        """
        Name of the statistic.
        """
        return [obs.stat_name for obs in self.observables]
    
    @property
    def paths(self):
        """
        Defines the default paths for the statistics results.
        """
        return {obs.stat_name: obs.paths for obs in self.observables} 
    
    @property
    def summary_coords_dict(self):
        """
        Defines the default coordinates for the statistics results.
        """
        return {obs.stat_name: obs.summary_coords_dict for obs in self.observables} 

    @property
    def lhc_x(self):
        """
        Latin hypercube of input features (cosmological and/or HOD parameters).

        Note: We assume all observable have the same input features, so we just
        return the first from the list.
        """
        return [obs.lhc_x for obs in self.observables][0]

    @property
    def lhc_x_names(self):
        """
        Names of the input features (cosmological and/or HOD parameters).

        Note: We assume all observable have the same input features, so we just
        return the first from the list.
        """
        return [obs.lhc_x_names for obs in self.observables][0]

    @property
    def lhc_y(self):
        """
        Latin hypercube of output features (tpcf, power spectrum, etc).
        """
        return np.concatenate([obs.lhc_y for obs in self.observables], axis=-1)
    
    @property
    def bin_values(self):
        """
        Bin values for the statistics. (e.g. separation bins for the correlation function).
        """
        return np.concatenate([obs.bin_values for obs in self.observables], axis=-1)

    @property
    def covariance_y(self):
        """
        Features from small AbacusSummit box for covariance estimation.
        """
        return np.concatenate([obs.covariance_y for obs in self.observables], axis=-1)

    @property
    def models(self):
        """
        Dict of theory models of the combination of observables.
        """
        return {obs.stat_name: obs.model for obs in self.observables}
    
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