"""
Module generating galaxy catalogs.

The pipeline is organized in three layers:
  1. backends       — load simulation data and generate galaxy catalogs from it
  2. products       — store and provide access to per-tracer galaxy data
  3. factories      — orchestrate the backend and catalog classes
"""
from .dataclasses import Tracer, Transform
from .factories.snapshot import GalaxyCatalogFactory
from .products.snapshot import SnapshotCatalog, RandomSnapshotCatalog
