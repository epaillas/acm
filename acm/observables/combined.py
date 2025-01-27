import numpy as np
from .base import BaseObservable


class BaseCombinedObservable(BaseObservable):
    """
    Class for the combination of observables.
    """
    def __init__(self, observables: list[BaseObservable]):
        self.observables = observables
        self.slice_filters = [obs.slice_filters for obs in self.observables]
        self.select_filters = [obs.select_filters for obs in self.observables]
        super().__init__()

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
        return [obs.paths for obs in self.observables]
    
    @property
    def summary_coords_dict(self):
        """
        Defines the default coordinates for the statistics results.
        """
        return [obs.summary_coords_dict for obs in self.observables]

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
        Bin values for the statistic. (e.g. separation bins for the correlation function).
        
        Note: We assume all observable have the same input features, so we just
        return the first from the list. 
        """
        return [obs.bin_values for obs in self.observables][0]

    @property
    def covariance_y(self):
        """
        Features from small AbacusSummit box for covariance estimation.
        """
        return np.concatenate([obs.covariance_y for obs in self.observables], axis=-1)

    @property
    def model(self):
        """
        Theory model of the combination of observables.
        """
        return [obs.model for obs in self.observables]
    
    def get_model_prediction(self, x)-> np.ndarray:
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