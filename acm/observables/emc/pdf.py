import xarray
import numpy as np
import glob
from pathlib import Path
from .base import BaseObservableEMC
import matplotlib.pyplot as plt
from jaxpower import read
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.plotting import set_plot_style
from acm.utils.decorators import temporary_class_state
from acm.utils.xarray import dataset_to_dict, split_vars


class GalaxyOverdensityPDF(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge galaxy overdensity PDF.
    """
    def __init__(self, **kwargs):
        super().__init__(stat_name='pdf', **kwargs)

# Alias
pdf = GalaxyOverdensityPDF