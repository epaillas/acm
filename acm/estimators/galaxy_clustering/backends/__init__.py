"""
Backends for Galaxy Clustering Estimators.

This module handles the different backends used in the estimators to create mesh fields from galaxy catalogs and compute density contrasts
"""

from importlib import import_module

from acm.utils.modules import check_installed

from .base import EstimatorBackend, load_backend

if check_installed("jaxpower"):
    from .jaxpower import JaxpowerBackend
if check_installed("pypower"):
    from .pypower import PypowerBackend
if check_installed("pyrecon") and hasattr(import_module("pyrecon"), "RealMesh"):
    from .pyrecon import PyreconBackend  # NOTE: RealMesh exists only on main branch
