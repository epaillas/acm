from acm.observables.base import BaseObservable
from .default import emc_summary_coords_dict, emc_paths

class BaseObservableEMC(BaseObservable):
    """
    Base class for the application of the ACM pipeline to the BGS dataset.
    """
    def __init__(self, slice_filters: dict = None, select_filters: dict = None):
        super().__init__(slice_filters=slice_filters, select_filters=select_filters)
        
    # NOTE: Define the stat name in the child class !
        
    @property
    def paths(self) -> dict:
        """
        Defines the default paths for the statistics results.
        
        Returns
        -------
        dict
            Dictionary with the paths for the statistics results.
            It must contain the following keys:
            - 'lhc_dir' : Directory containing the LHC data.
            - 'covariance_dir' : Directory containing the covariance array of the LHC data.
            - 'model_dir' : Directory where the model is saved.
        """
        return emc_paths

    @property
    def summary_coords_dict(self):
        """
        Defines the default coordinates for the statistics results. 
        """
        return emc_summary_coords_dict