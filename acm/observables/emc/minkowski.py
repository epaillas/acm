from .base import BaseObservable
import logging


class MinkowskiFunctionals(BaseObservable):
    """
    Class for the Emulator's Mock Challenge Minkowski functionals.
    """
    def __init__(self, phase_correction=False, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stat_name = 'minkowski'
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
            'hod_idx': list(range(200)),
        }

    @property
    def test_set_indices(self):
        """
        Indices of the test set samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)),
            'hod_idx': list(range(200)),
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
        return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/minkowski/cosmo+hod/best-model-epoch=132-val_loss=0.0319.ckpt'

    def get_emulator_error(self, select_filters=None, slice_filters=None):
        from sunbird.data.data_utils import convert_to_summary
        from pathlib import Path
        import numpy as np
        error_dir = '/pscratch/sd/e/epaillas/emc/v1.1/emulator_error/'
        error_fn = Path(error_dir) / 'minkowski.npy'
        error = np.load(error_fn, allow_pickle=True).item()['emulator_error']
        coords = self.coordinates_indices if self.select_indices else self.coordinates
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(coords.keys())
        error = error.reshape(*coords_shape)
        select_filters = self.select_coordinates if self.select_coordinates else self.select_indices
        slice_filters = self.slice_coordinates
        return convert_to_summary(
            data=error, dimensions=dimensions, coords=coords,
            select_filters=select_filters, slice_filters=slice_filters
        ).values.reshape(-1)