from .base import BaseEnvironmentEstimator
from PolyBin3D import PolyBin3D, BSpec
import logging
import time


# class Bispectrum(BSpec):
#     """
#     Bispectrum class that inherits from the PolyBin3D code developed by Oliver Philcox & Thomas Flöss
#     (https://github.com/oliverphilcox/PolyBin3D). 
#     """
#     def __init__(self, **kwargs):
#         self.base = PolyBin3D(**kwargs)

#     def set_binning(self, **kwargs):
#         super().__init__(base=self.base, **kwargs)

class PolyBinEstimator(BaseEnvironmentEstimator):
    """
    PolyBin class that inherits from the PolyBin3D code developed by Oliver Philcox & Thomas Flöss
    (https://github.com/oliverphilcox/PolyBin3D). 
    """
    def __init__(self, sightline='global', **kwargs):
        super().__init__(**kwargs)

        self.base = PolyBin3D(
            sightline=sightline,
            gridsize=self.data_mesh.nmesh,
            boxsize=self.data_mesh.boxsize,
            boxcenter=self.data_mesh.boxcenter,
            backend='jax',

        )


class Bispectrum(PolyBinEstimator, BSpec):
    """
    Bispectrum class that inherits from the PolyBin3D code developed by Oliver Philcox & Thomas Flöss
    (https://github.com/oliverphilcox/PolyBin3D). 
    """
    def __init__(self,  **kwargs):
        self.logger = logging.getLogger('Bispectrum')
        self.logger.info('Initializing Bispectrum.')
        PolyBinEstimator.__init__(self, **kwargs)

    def set_binning(self, **kwargs):
        BSpec.__init__(self, base=self.base, **kwargs)

    def Bk_ideal(self, **kwargs):
        t0 = time.time()
        bk = BSpec.Bk_ideal(self, data=self.delta_mesh.value, **kwargs)
        self.logger.info(f'Computed ideal bispectrum in {time.time() - t0:.2f} seconds.')
        return bk