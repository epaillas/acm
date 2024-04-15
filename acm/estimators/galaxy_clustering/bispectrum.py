from PolyBin3D import BSpec


class Bispectrum(BSpec):
    """
    Bispectrum class that inherits from the PolyBin3D code developed by Oliver Philcox & Thomas Flöss
    (https://github.com/oliverphilcox/PolyBin3D). 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)