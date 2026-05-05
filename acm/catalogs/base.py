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

from acm.catalogs.backends import DarkMatterBackend, SnapshotBackend, load_backend
from acm.catalogs.dataclasses import Tracer

logger = logging.getLogger(__name__)


class BaseGalaxyCatalog:
    """
    Stores galaxy data for multiple tracers at a fixed redshift.

    GalaxyCatalog is geometry-agnostic: it does not know how the data was
    produced or what columns it contains. Subclasses (CubicGalaxyCatalog,
    CutskyGalaxyCatalog, etc.) may add geometry-specific behaviour, but the
    base storage and retrieval interface is defined here.

    Cosmology is passed in from the factory and stored as references, so all
    snapshots in a factory share the same cosmo and cosmo_fid objects.
    """

    def __init__(
        self,
        redshift: float,
        cosmo: Cosmology,
        cosmo_fid: Cosmology,
    ) -> None:
        """
        Initialize the galaxy catalog with the given redshift and cosmologies.

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
        self._data: dict[str, DataFrame] = {}

    def __repr__(self) -> str:
        """Provide a string representation of the galaxy catalog, including redshift and tracer information."""
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
        return self.cosmo.angular_diameter_distance(
            self.redshift
        ) / self.cosmo_fid.angular_diameter_distance(self.redshift)

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
    def get_catalog(self, *args, **kwargs) -> BaseGalaxyCatalog: ...


class SnapshotCatalogFactory(BaseCatalogFactory):
    """
    Abstract base class for snapshot-based catalog factories.

    Subclasses must implement make_catalogs and get_catalog.
    """

    def __init__(
        self,
        backend: str | SnapshotBackend,
        catalog_class: type[BaseGalaxyCatalog],
        cosmo: Cosmology,
        cosmo_fid: Cosmology | None = None,
        **kwargs,
    ) -> None:
        super().__init__(backend, catalog_class, cosmo, cosmo_fid, **kwargs)
        self.backend: SnapshotBackend  # type hint for better autocompletion

    @abstractmethod
    def make_catalogs(
        self,
        redshifts: list[float],
        tracers: list[Tracer] | dict[float, list[Tracer]],
        **kwargs,
    ) -> None:
        """
        Load dark matter snapshots and populate galaxy catalogs for each redshift.

        Parameters
        ----------
        redshifts : list[float]
            List of redshifts at which to load dark matter snapshots.
        tracers : list[Tracer] | dict[float, list[Tracer]]
            Tracers to populate for each redshift. Can be a single list applied to all redshifts
            or a dictionary mapping each redshift to its own list of tracers.
        dark_matter_kwargs : dict, optional
            Keyword arguments forwarded to the backend when loading the dark matter catalog (e.g. default tracer parameters).
        **kwargs
            Extra arguments forwarded to the backend.
        """
        ...

    @abstractmethod
    def get_catalog(self, redshift: float) -> BaseGalaxyCatalog:
        """
        Retrieve the galaxy catalog at a given redshift.

        Parameters
        ----------
        redshift : float
            The redshift of the desired snapshot.
        """
        ...
