"""
Base classes for the galaxy catalog pipeline.

The pipeline is organized in three layers:
  1. DarkMatterBackend   — loads simulation data and generates galaxy catalogs from it
  2. GalaxyCatalog       — stores and provides access to per-tracer galaxy data
  3. BaseCatalogFactory  — orchestrates the backend and catalog classes

Two geometry-specific branches are provided:
  - Snapshot-based (SnapshotBackend / SnapshotCatalogFactory)
  - Lightcone-based (LightconeBackend / LightconeCatalogFactory) - TODO
"""

import logging
from abc import ABC, abstractmethod

from cosmoprimo import Cosmology
from cosmoprimo.fiducial import DESI
from pandas import DataFrame

from acm.catalogs.backends import DarkMatterBackend, load_backend
from acm.catalogs.dataclasses import Tracer

logger = logging.getLogger(__name__)


class BaseGalaxyCatalog:
    """
    Stores galaxy data for multiple tracers.

    GalaxyCatalog is geometry-agnostic: it does not know how the data was
    produced or what columns it contains. Subclasses (CubicGalaxyCatalog,
    CutskyGalaxyCatalog, etc.) may add geometry-specific behaviour, but the
    base storage and retrieval interface is defined here.

    Cosmology is passed in from the factory and stored as references, so all
    snapshots in a factory share the same cosmo and cosmo_fid objects.
    """

    def __init__(
        self,
        cosmo: Cosmology,
        cosmo_fid: Cosmology,
    ) -> None:
        """
        Initialize the galaxy catalog with the given cosmologies.

        Parameters
        ----------
        cosmo : cosmoprimo.Cosmology, optional
            The cosmology for the simulation.
        cosmo_fid : cosmoprimo.Cosmology, optional
            The fiducial cosmology.
        """
        self.cosmo = cosmo
        self.cosmo_fid = cosmo_fid
        self.tracers: dict[str, Tracer] = {}
        self._data: dict[str, DataFrame] = {}

    def __repr__(self) -> str:
        """Provide a string representation of the galaxy catalog, including tracer information."""
        return (
            f"{self.__class__.__name__}("
            f"tracers={list(self.tracers.keys())})"
        )

    def register_tracer(self, tracer: Tracer) -> None:
        """Register a tracer in the catalog."""
        if tracer.name in self.tracers:
            logger.warning(f"Tracer '{tracer.name}' already exists.")
        self.tracers[tracer.name] = tracer

    def set_tracer_data(self, tracer: Tracer, data: DataFrame) -> None:
        """Set the galaxy data for a given tracer."""
        self.register_tracer(tracer)  # Ensure tracer is registered before setting data
        self._data[tracer.name] = data

    def get_tracer_data(self, tracer_name: str) -> DataFrame:
        """Get the galaxy data for a given tracer."""
        if tracer_name not in self._data:
            raise KeyError(f"No data loaded for tracer '{tracer_name}'.")
        return self._data[tracer_name]

    def __getitem__(self, tracer_name: str) -> DataFrame:
        """Allow direct indexing to get tracer data, e.g. catalog['ELG']."""
        return self.get_tracer_data(tracer_name)


class BaseCatalogFactory(ABC):
    """
    Abstract base class for all catalog factories.

    Holds the shared infrastructure: backend loading, cosmology objects,
    catalog class selection, and the internal catalog store. Geometry-specific
    orchestration (how and when catalogs are created) is left to subclasses.

    All factories share:
        - cosmo     : simulation cosmology (cosmoprimo)
        - cosmo_fid : fiducial cosmology, defaults to DESI()
        - catalog_class : the GalaxyCatalog subclass to instantiate
    """

    def __init__(
        self,
        backend: str | DarkMatterBackend,
        catalog_class: type[BaseGalaxyCatalog],
        cosmo: Cosmology,
        cosmo_fid: Cosmology | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the catalog factory with the specified backend, catalog class, and cosmology.

        Parameters
        ----------
        backend : str | DarkMatterBackend
            The dark matter backend to load catalogs from.
        catalog_class : type[BaseGalaxyCatalog]
            The galaxy catalog class to instantiate.
        cosmo : cosmoprimo.Cosmology
            Simulation cosmology, passed down to every catalog.
        cosmo_fid : cosmoprimo.Cosmology, optional
            Fiducial cosmology. Defaults to DESI().
        **kwargs
            Keyword arguments to pass to the backend constructor.
        """
        self.backend = load_backend(backend, **kwargs)
        self.catalog_class = catalog_class
        self._catalogs = {}
        self.cosmo = cosmo
        self.cosmo_fid = cosmo_fid if cosmo_fid is not None else DESI()

    def __repr__(self) -> str:
        """Provide a string representation of the catalog factory, including backend and catalog class information"""
        return (
            f"{self.__class__.__name__}("
            f"backend={self.backend.__class__.__name__}, "
            f"catalog_class={self.catalog_class.__name__})"
        )

    @property
    def catalogs(self) -> dict:
        """Dictionary of all loaded galaxy catalogs, keyed by redshift."""
        return dict(self._catalogs)

    @abstractmethod
    def get_catalog(self, *args, **kwargs) -> BaseGalaxyCatalog: ... #noqa: D102