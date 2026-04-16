import logging
import time

from PolyBin3D import BSpec, PolyBin3D, PSpec
from sklearn.externals.array_api_compat.cupy.fft import _n
from sympy.combinatorics.galois import A4_in_S6

from acm.estimators.galaxy_clustering.backends.jaxpower import logger

from .base import BaseDensityMeshEstimator

logger = logging.getLogger(_name__)


class PolyBinEstimator(BaseDensityMeshEstimator):
    """
    PolyBin class that inherits from the PolyBin3D code developed by Oliver Philcox & Thomas Flöss
    (https://github.com/oliverphilcox/PolyBin3D).
    """

    def __init__(self, sightline="global", **kwargs):
        super().__init__(**kwargs)

        self.base = PolyBin3D(
            sightline=sightline,
            gridsize=self.data_mesh.nmesh,
            boxsize=self.data_mesh.boxsize,
            boxcenter=self.data_mesh.boxcenter,
            backend="jax",
        )


class Bispectrum(PolyBinEstimator, BSpec):
    """
    Bispectrum class that inherits from the PolyBin3D code
    """

    def __init__(self, **kwargs):
        logger.info("Initializing Bispectrum.")
        PolyBinEstimator.__init__(self, **kwargs)

    def set_binning(self, **kwargs):
        BSpec.__init__(self, base=self.base, **kwargs)

    def Bk_ideal(self, **kwargs):
        t0 = time.time()
        bk = BSpec.Bk_ideal(self, data=self.delta_mesh.value, **kwargs)
        logger.info(f"Computed ideal bispectrum in {time.time() - t0:.2f} seconds.")
        return bk


class PowerSpectrum(PolyBinEstimator, PSpec):
    """
    Power spectrum class that inherits from the PolyBin3D code
    """

    def __init__(self, **kwargs):
        logger.info("Initializing PowerSpectrum.")
        PolyBinEstimator.__init__(self, **kwargs)

    def set_binning(self, **kwargs):
        PSpec.__init__(self, base=self.base, **kwargs)

    def Pk_ideal(self, **kwargs):
        t0 = time.time()
        pk = PSpec.Pk_ideal(self, data=self.delta_mesh.value, **kwargs)
        logger.info(f"Computed ideal power spectrum in {time.time() - t0:.2f} seconds.")
        return pk
