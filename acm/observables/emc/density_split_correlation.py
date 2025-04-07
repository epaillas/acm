from .base import BaseObservable
import logging

class DensitySplitCorrelationFunctionMultipoles(BaseObservable):
    """
    Class for the Emulator's Mock Challenge density-split correlation
    function multipoles.
    """
    def __init__(self, phase_correction=False, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stat_name = 'dsc_xi'
        self.sep_name = 's'

        if phase_correction and hasattr(self, 'compute_phase_correction'):
            self.logger.info('Computing phase correction.')
            self.phase_correction = self.compute_phase_correction()

        super().__init__(**kwargs)

    @property
    def lhc_indices(self):
        """
        Indices of the Latin hypercube samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(100)),
        }

    @property
    def test_set_indices(self):
        """
        Indices of the test set samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)),
            'hod_idx': list(range(100)),
        }

    @property
    def small_box_indices(self):
        """
        Indices of the covariance samples, including variations in phase and HOD parameters.
        """
        return {
            'phase_idx': list(range(1786)),
        }

    @property
    def coordinates(self):
        """
        Coordinates of the data and model vectors.
        """
        return{
            'statistics': ['quantile_data_power', 'quantile_power'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            's': self.separation,
        }
    
    @property
    def coordinates_indices(self):
        """
        Indices of the (flat) coordinates of the data and model vectors.
        """
        return{'bin_idx': list(range(2 * 4 * 2 * len(self.separation)))}

    @property
    def model_fn(self):
        return f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_conf/cosmo+hod/aug9/last-v1.ckpt'