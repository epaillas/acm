"""
Abstract backend interfaces for dark matter simulations.

A backend is responsible for two things:
  1. Loading a dark matter halo catalog from a simulation (get_dark_matter_catalog)
  2. Populating it with galaxies via an HOD or similar model (make_galaxy_catalog)

To implement a new backend, subclass SnapshotBackend or LightconeBackend
and register it with @register_backend("<name>").
"""
from acm.utils.modules import check_installed

from .base import (
  DarkMatterBackend,
  SnapshotBackend,
  load_backend,
)

# Register backends if possible:
if check_installed('abacusnbody'):
  from .abacus import AbacusHODBackend
