from .base import BaseObservableEMC

class GalaxyPowerSpectrumMultipoles(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge bispectrum.
    """
    def __init__(self, select_filters: dict = None, slice_filters: dict = None):
        super().__init__(select_filters=select_filters, slice_filters=slice_filters)
        
    @property
    def stat_name(self) -> str:
        """
        Name of the statistic.
        """
        stat_name = 'pk'
        return stat_name
    
    @property
    def summary_coords_dict(self):
        """
        Defines the default coordinates for the statistics results. 
        """        
        coords = super().summary_coords_dict
        coords['hod_number'] = 250
        coords['statistics'] = {
            self.stat_name: {
                'multipoles': [0, 2],
            },
        }
        return coords