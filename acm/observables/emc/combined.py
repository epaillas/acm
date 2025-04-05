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


class CombinedObservable:
    """
    Class for the Emulator's Mock Challenge combination of observables.
    """
    def __init__(self, observables: list):
        self.observables = observables
        self.stat_name = [obs.stat_name for obs in self.observables]

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
        return np.concatenate([obs.lhc_y for obs in self.observables], axis=0)

    @property
    def small_box_y(self):
        """
        Features from small AbacusSummit box for covariance estimation.
        """
        return np.concatenate([obs.small_box_y for obs in self.observables], axis=1)

    def diffsky_y(self, **kwargs):
        """
        Measurements from Diffsky simulations.
        """
        return np.concatenate([obs.diffsky_y(**kwargs) for obs in self.observables])

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
        return np.concatenate([obs.emulator_error for obs in self.observables], axis=0)

    @property
    def coords_model(self):
        """
        Coordinates of the model.
        """
        return [obs.coords_model for obs in self.observables]

    @property
    def coords_small_box(self):
        """
        Coordinates of the small box.
        """
        return [obs.coords_small_box for obs in self.observables]

    @property
    def coords_lhc_x(self):
        """
        Coordinates of the Latin hypercube of input features.
        """
        return [obs.coords_lhc_x for obs in self.observables]

    @property
    def coords_lhc_y(self):
        """
        Coordinates of the Latin hypercube of output features.
        """
        return [obs.coords_lhc_y for obs in self.observables]

    @property
    def model_fn(self):
        """
        Filename of the model.
        """
        return [obs.model_fn for obs in self.observables]

    def get_emulator_error(self, method: ['mae', 'cov'] = 'mae'):
        """
        Calculate the emulator error from a subset of the Latin hypercube,
        which we treat as the test set.

        We make a new instance of the class with the test set filters and
        compare the emulator prediction to the true values.
        """
        return np.concatenate([obs.get_emulator_error(method) for obs in self.observables], axis=0)

    def get_covariance_matrix(self, divide_factor=64):
        """
        Covariance matrix of the combination of observables.

        Args:
            divide_factor (int): Divide the covariance matrix by this value
            to account for the volume difference between the small boxes and the target
            simulation.
        """
        return np.cov(self.small_box_y.T) / divide_factor

    def get_covariance_correction(self, n_s, n_d, n_theta=None, method='percival'):
        """
        Correction factor to debias de inverse covariance matrix.

        Args:
            n_s (int): Number of simulations.
            n_d (int): Number of bins of the data vector.
            n_theta (int): Number of free parameters.
            method (str): Method to compute the correction factor.

        Returns:
            float: Correction factor
        """
        if method == 'percival':
            B = (n_s - n_d - 2) / ((n_s - n_d - 1)*(n_s - n_d - 4))
            return (n_s - 1)*(1 + B*(n_d - n_theta))/(n_s - n_d + n_theta - 1)
        elif _method == 'hartlap':
            return (n_s - 1)/(n_s - n_d - 2)

