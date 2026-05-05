from .base import BaseObservableEMC


class GalaxyOverdensityPDF(BaseObservableEMC):
    """Class for the Emulator's Mock Challenge galaxy overdensity PDF."""

    def __init__(self, stat_name: str = "pdf", **kwargs) -> None:
        super().__init__(stat_name=stat_name, **kwargs)


# Alias
pdf = GalaxyOverdensityPDF
