from .base import BaseObservable


class DensitySplitPowerSpectrumMultipoles(BaseObservable):
    """
    Class for the Emulator's Mock Challenge density-split power spectrum
    multipoles.
    """
    def __init__(self, select_filters: dict = None, slice_filters: dict = None):
        self.stat_name = 'dsc_pk'
        self.sep_name = 'k'
        self.select_filters = select_filters
        self.slice_filters = slice_filters
        super().__init__()
        
    @property
    def coords_lhc_x(self):
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(250)),
            'param_idx': list(range(20))
        }

    @property
    def coords_lhc_y(self):
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(250)),
            'statistics': ['quantile_data_power', 'quantile_power'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            'k': self.separation,
        }

    @property
    def coords_small_box(self):
        return {
            'phase_idx': list(range(1786)),
            'statistics': ['quantile_data_power', 'quantile_power'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            'k': self.separation,
        }

    @property
    def coords_model(self):
        return {
            'statistics': ['quantile_data_power', 'quantile_power'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            'k': self.separation,
        }

    @property
    def model_fn(self):
        return f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_fourier/cosmo+hod/optuna/last-v25.ckpt'