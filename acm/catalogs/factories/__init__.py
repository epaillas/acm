"""
Concrete catalog factories for snapshot-based pipelines.

GalaxyCatalogFactory is the main entry point for users: it loads a dark matter
backend, runs the HOD model at each requested redshift, and stores the resulting
GalaxyCatalog instances for downstream use.
"""
from .base import BaseCatalogFactory
from .snapshot import SnapshotCatalogFactory
