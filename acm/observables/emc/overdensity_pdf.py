from .base import BaseObservable
import logging


class GalaxyOverdensityPDF(BaseObservable):
    """
    Class for the Emulator's Mock Challenge galaxy overdensity PDF.
    """
    def __init__(self, phase_correction=False, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stat_name = 'pdf_r10'
        self.sep_name = 'delta'

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
            'delta': self.separation,
        }
    
    @property
    def coordinates_indices(self):
        """
        Indices of the (flat) coordinates of the data and model vectors.
        """
        return{'bin_idx': list(range(len(self.separation)))}

    @property
    def model_fn(self):
        return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/pdf_r10/cosmo+hod/optuna/last-v13.ckpt'