from .base import BaseObservable


class DTVoidGalaxyCorrelationFunctionMultipoles(BaseObservable):
    """
    Class for the Emulator's Mock Challenge void-galaxy correlation
    function multipoles using the DT void finder.
    """
    def __init__(self, select_filters: dict = None, slice_filters: dict = None):
        self.stat_name = 'dt_gv'
        self.sep_name = 's'
        self.select_filters = select_filters
        self.slice_filters = slice_filters
        super().__init__()
        
    @property
    def coords_lhc_x(self):
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(100)),
            'param_idx': list(range(20))
        }

    @property
    def coords_lhc_y(self):
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(100)),
            'multipoles': [0, 2],
            's': self.separation,
        }

    @property
    def coords_small_box(self):
        return {
            'phase_idx': list(range(1786)),
            'multipoles': [0, 2],
            's': self.separation,
        }

    @property
    def coords_model(self):
        return {
            'multipoles': [0, 2],
            's': self.separation,
        }

    @property
    def model_fn(self):
        return '/pscratch/sd/d/dforero/emc/trained_models/DTvoids_gv/cosmo+hod/optuna/last.ckpt'

    def get_emulator_error(self, select_filters=None, slice_filters=None):
        """
        Calculate the emulator error from a subset of the Latin hypercube,
        which we treat as the test set.
        
        We make a new instance of the class with the test set filters and
        compare the emulator prediction to the true values.
        """
        import numpy as np
        if select_filters is None:
            select_filters = self.select_filters
            select_filters['cosmo_idx'] = list(range(0, 5)) + list(range(13, 14))
            select_filters['hod_idx'] = list(range(100))
        if slice_filters is None:
            slice_filters = self.slice_filters
        # instantiate class with test set filters
        observable = self.__class__(select_filters=select_filters, slice_filters=slice_filters)
        test_x = observable.lhc_x
        test_y = observable.lhc_y
        # reshape to (n_samples, n_features)
        n_samples = len(select_filters['cosmo_idx']) * len(select_filters['hod_idx'])
        test_x = test_x.reshape(n_samples, -1)
        test_y = test_y.reshape(n_samples, -1)
        pred_y = observable.get_model_prediction(test_x, batch=True)
        return np.median(np.abs(test_y - pred_y), axis=0)