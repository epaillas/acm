import logging
import time

from PolyBin3D import BSpec, PolyBin3D, PSpec

from .base import BaseDensityMeshEstimator

logger = logging.getLogger(__name__)


class PolyBinEstimator(BaseDensityMeshEstimator):
    """
    PolyBin class.

    Inherits from the PolyBin3D code developed by Oliver Philcox & Thomas Flöss
    (https://github.com/oliverphilcox/PolyBin3D).
    """

    def __init__(self, sightline: str = "global", **kwargs) -> None:
        super().__init__(**kwargs)

        self.base = PolyBin3D(
            sightline=sightline,
            gridsize=self.data_mesh.nmesh,
            boxsize=self.data_mesh.boxsize,
            boxcenter=self.data_mesh.boxcenter,
            backend="jax",
        )


class Bispectrum(PolyBinEstimator, BSpec):
    """Bispectrum class that inherits from the PolyBin3D code."""

    def __init__(self, **kwargs) -> None:
        logger.info("Initializing Bispectrum.")
        PolyBinEstimator.__init__(self, **kwargs)

    def set_binning(self, **kwargs) -> None:
        """Set up the binning for the bispectrum estimation."""
        BSpec.__init__(self, base=self.base, **kwargs)

    def bk_ideal(self, **kwargs) -> dict:
        """Compute the ideal bispectrum from the density mesh."""
        t0 = time.time()
        bk = BSpec.Bk_ideal(self, data=self.delta_mesh.value, **kwargs)
        logger.info(f"Computed ideal bispectrum in {time.time() - t0:.2f} seconds.")
        return bk


class PowerSpectrum(PolyBinEstimator, PSpec):
    """Power spectrum class that inherits from the PolyBin3D code."""

    def __init__(self, **kwargs) -> None:
        logger.info("Initializing PowerSpectrum.")
        PolyBinEstimator.__init__(self, **kwargs)

    def set_binning(self, **kwargs) -> None:
        """Set up the binning for the power spectrum estimation."""
        PSpec.__init__(self, base=self.base, **kwargs)

    def pk_ideal(self, **kwargs) -> dict:
        """Compute the ideal power spectrum from the density mesh."""
        t0 = time.time()
        pk = PSpec.Pk_ideal(self, data=self.delta_mesh.value, **kwargs)
        logger.info(f"Computed ideal power spectrum in {time.time() - t0:.2f} seconds.")
        return pk
