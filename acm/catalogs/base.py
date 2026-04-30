import logging
from abc import ABC, abstractmethod
from typing import Any

from cosmoprimo import Cosmology
from cosmoprimo.fiducial import DESI

from .dataclasses import Tracer
from .backends import DarkMatterBackend, load_backend

logger = logging.getLogger(__name__)


class GalaxyCatalog:
    """
    Abstract base class for galaxy catalogs at a fixed redshift.

    It provides common functionalities for loading and processing galaxy catalogs,
    which can be extended by child classes for specific use cases.
    The class is designed to handle a multi-tracer galaxy catalog at a fixed redshift.
    """
    def __init__(
        self, 
        redshift: float, 
        cosmo:Cosmology = None, 
        cosmo_fid: Cosmology = None,
    ) -> None:
        """
        Parameters
        ----------
        redshift : float
            Redshift of the snapshot.
        cosmo : cosmoprimo.Cosmology, optional
            The cosmology for the simulation.
        cosmo_fid : cosmoprimo.Cosmology, optional
            The fiducial cosmology.
        """
        self.redshift = redshift
        self.cosmo = cosmo
        self.cosmo_fid = cosmo_fid
        self.tracers: dict[str, Tracer] = {}
        self._data: dict[str, Any] = {}

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"redshift={self.redshift}, "
            f"tracers={list(self.tracers.keys())})"
        )
    
    @property
    def az(self) -> float:
        """Scale factor at this snapshot's redshift."""
        return 1.0 / (1.0 + self.redshift)

    @property
    def hubble(self) -> float:
        """H(z) in km/s/(Mpc/h) for the simulation cosmology."""
        return 100.0 * self.cosmo.efunc(self.redshift)

    @property
    def hubble_fid(self) -> float:
        """H(z) in km/s/(Mpc/h) for the fiducial cosmology."""
        return 100.0 * self.cosmo_fid.efunc(self.redshift)

    @property
    def q_par(self) -> float:
        """AP parallel scaling factor."""
        return self.hubble_fid / self.hubble

    @property
    def q_perp(self) -> float:
        """AP perpendicular scaling factor."""
        return (
            self.cosmo.angular_diameter_distance(self.redshift)
            / self.cosmo_fid.angular_diameter_distance(self.redshift)
        )
    
    def register_tracer(self, tracer: Tracer) -> None:
        if tracer.name in self.tracers:
            logger.warning(f"Tracer '{tracer.name}' already exists.")
        self.tracers[tracer.name] = tracer

    def set_tracer_data(self, tracer: Tracer, data: Any) -> None:
        self.register_tracer(tracer) # Ensure tracer is registered before setting data
        self._data[tracer.name] = data

    def get_tracer_data(self, tracer_name: str) -> Any:
        if tracer_name not in self._data:
            raise KeyError(f"No data loaded for tracer '{tracer_name}'.")
        return self._data[tracer_name]


class BaseCatalogFactory(ABC):
    """
    Abstract base class for galaxy catalog factories.

    This class defines the interface for loading dark matter catalogs and
    populating galaxy catalogs.
    Child classes should implement the specific logic for loading and processing
    the catalogs based on the chosen backend and galaxy catalog structure.
    """
    
    def __init__(
        self,
        backend: str | DarkMatterBackend,
        cosmo: Cosmology = None,
        catalog_class: type[GalaxyCatalog] = GalaxyCatalog,
        cosmo_fid: Cosmology = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        backend : str | DarkMatterBackend
            The dark matter backend to load catalogs from.
        cosmo : cosmoprimo.Cosmology, optional
            Simulation cosmology, passed down to every catalog.
        catalog_class : type[GalaxyCatalog]
            The galaxy catalog class to instantiate. Defaults to GalaxyCatalog.
        cosmo_fid : cosmoprimo.Cosmology, optional
            Fiducial cosmology. Defaults to DESI().
        **kwargs
            Keyword arguments to pass to the backend constructor.
        """
        self.backend = load_backend(backend, **kwargs)
        self.cosmo = cosmo
        self.cosmo_fid = cosmo_fid if cosmo_fid is not None else DESI()
        self.catalog_class = catalog_class
        self._catalogs: dict = {}
    
    def __repr__(self) -> str:
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
    def make_catalogs(self): ...

    @abstractmethod
    def get_catalog(self): ...

