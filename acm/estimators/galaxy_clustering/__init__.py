"""Module for Galaxy clustering estimators."""

from acm.utils.modules import check_installed

if check_installed("pycorr"):
    from .tpcf import TwoPointCorrelationFunctionEstimator  # pragma: no cover
if check_installed("pycorr", "jaxpower"):
    from .density_split import DensitySplit  # pragma: no cover
if check_installed("jaxpower"):
    from .spectrum import PowerSpectrumMultipoles  # pragma: no cover
if check_installed("kymatio", "torch"):
    from .wst import WaveletScatteringTransform  # pragma: no cover
