import numpy as np
import torch
from pathlib import Path

from .base import BaseObservableBGS

class DensitySplitConfSpace(BaseObservableBGS):
    """
    Class for the application of the densitysplit statistic of the ACM pipeline to the BGS dataset.
    """
    
    def __init__(self, slice_filters: dict = None, select_filters: dict = None):
        super().__init__(slice_filters=slice_filters, select_filters=select_filters)
        
    @property
    def stat_name(self) -> str:
        """
        Name of the statistic.
        """
        stat_name = 'dsc_conf'
        return stat_name

    #%% LHC file (TODO)
    def create_covariance(
        self,
        
        )-> np.ndarray:
        """
        Create the covariance array for the density split statistic.
        """
        pass
    
    def create_lhc(self):
        pass