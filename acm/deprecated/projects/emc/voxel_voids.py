from .base import BaseObservable


class VoxelVoidGalaxyCorrelationFunctionMultipoles(BaseObservable):
    """
    Class for the Emulator's Mock Challenge void-galaxy correlation
    function multipoles using the voxel void finder.
    """
    def __init__(self, select_filters: dict = None, slice_filters: dict = None):
        self.stat_name = 'voxel_voids'
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
        return f'/pscratch/sd/e/epaillas/emc/trained_models/voxel_voids/cosmo+hod/sep16/last.ckpt'