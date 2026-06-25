"""
Backends for Galaxy Clustering Estimators.

This module handles the different backends used in the estimators to create mesh fields from galaxy catalogs and compute density contrasts
"""

from acm.utils.modules import check_installed

from .base import EstimatorBackend, load_backend

if check_installed("jaxpower"):
    from .jaxpower import JaxpowerBackend
if check_installed("pypower"):
    from .pypower import PypowerBackend
