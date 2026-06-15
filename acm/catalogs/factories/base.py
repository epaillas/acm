import logging
from abc import ABC, abstractmethod
from pathlib import Path

from cosmoprimo import Cosmology
from cosmoprimo.fiducial import DESI

from acm.catalogs.backends import DarkMatterBackend, load_backend
from acm.catalogs.products import BaseGalaxyCatalog

logger = logging.getLogger(__name__)


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

    def __repr__(self) -> str:  # pragma: no cover
        """Provide a string representation of the catalog factory, including backend and catalog class information."""
        return (
            f"{self.__class__.__name__}("
            f"backend={self.backend.__class__.__name__}, "
            f"catalog_class={self.catalog_class.__name__})"
        )

    @property
    def catalogs(self) -> dict:
        """Dictionary of all loaded galaxy catalogs."""
        return dict(self._catalogs)

    @abstractmethod
    def get_catalog(self, *args, **kwargs) -> BaseGalaxyCatalog: ...  # noqa: D102

    @abstractmethod
    def save(self, path: str | Path) -> None: ...  # noqa: D102

    @abstractmethod
    def load_catalogs(self, path: str | Path) -> None: ...  # noqa: D102
