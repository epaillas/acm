import numpy as np
from .base import BaseObservable


class CombinedObservable(BaseObservable):
    """
    Class for the Emulator's Mock Challenge combination of observables.
    """
    def __init__(self, observables: list):
        self.observables = observables
        self.stat_name = [obs.stat_name for obs in self.observables]
        self.slice_filters = [obs.slice_filters for obs in self.observables]
        self.select_filters = [obs.select_filters for obs in self.observables]
        super().__init__()

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
    def lhc_y(self, select_filters=None, slice_filters=None):
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
        """
        return [obs.model for obs in self.observables]

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

    def get_emulator_error(self, select_filters=None, slice_filters=None):
        """
        Calculate the emulator error from a subset of the Latin hypercube,
        which we treat as the test set.

        We make a new instance of the class with the test set filters and
        compare the emulator prediction to the true values.
        """
        return np.concatenate([obs.get_emulator_error(select_filters=select_filters,
            slice_filters=slice_filters) for obs in self.observables], axis=0)

