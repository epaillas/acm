from .base import BaseObservableEMC


class GalaxyOverdensityPDF(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge galaxy overdensity PDF.
    """

    def __init__(self, **kwargs):
        super().__init__(stat_name="pdf", **kwargs)


# Alias
pdf = GalaxyOverdensityPDF
