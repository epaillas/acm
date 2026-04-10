import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray
from jaxpower import read

from acm.utils.decorators import temporary_class_state
from acm.utils.default import cosmo_list  # List of cosmologies in AbacusSummit
from acm.utils.plotting import set_plot_style
from acm.utils.xarray import dataset_to_dict, split_vars

from .base import BaseObservableEMC


class GalaxyOverdensityPDF(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge galaxy overdensity PDF.
    """

    def __init__(self, **kwargs):
        super().__init__(stat_name="pdf", **kwargs)


# Alias
pdf = GalaxyOverdensityPDF
