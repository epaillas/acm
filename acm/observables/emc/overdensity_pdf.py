from .base import BaseObservable


class GalaxyOverdensityPDF(BaseObservable):
    """
    Class for the Emulator's Mock Challenge galaxy overdensity PDF.
    """
    def __init__(self, select_filters: dict = None, slice_filters: dict = None):
        self.stat_name = 'pdf_r10'
        self.sep_name = 'delta'
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
            self.sep_name: self.separation,
        }

    @property
    def coords_small_box(self):
        return {
            'phase_idx': list(range(1786)),
            self.sep_name: self.separation,
        }

    @property
    def coords_model(self):
        return {
            self.sep_name: self.separation,
        }

    @property
    def model_fn(self):
        return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/pdf_r10/cosmo+hod/optuna/last-v13.ckpt'