import logging
from abc import ABC, abstractmethod

from pandas import DataFrame

from acm.catalogs.dataclasses import Tracer
from acm.utils.backends import BackendRegistry

logger = logging.getLogger(__name__)


class DarkMatterBackend(ABC):
    """
    Root abstract class for all dark matter backends.

    Defines the one method common to all geometries: make_galaxy_catalog,
    which converts a loaded dark matter catalog into per-tracer galaxy data.
    get_dark_matter_catalog is geometry-dependent and is declared in the respective subclasses.
    """

    @abstractmethod
    def make_galaxy_catalog(
        self,
        dm_catalog: object,
        tracers: list[Tracer],
        **kwargs,
    ) -> dict[Tracer, DataFrame]:
        """
        Generate galaxy data for each tracer from a dark matter catalog.

        This part is common to all backends regardless of geometry.

        Parameters
        ----------
        dm_catalog : object
            The dark matter catalog to use for generating the galaxy catalog.
        tracers : list[Tracer]
            The list of tracers to generate the galaxy catalog for.
        **kwargs
            Additional keyword for backend-specific options.

        Returns
        -------
        dict[Tracer, DataFrame]
            A dictionary mapping each tracer to its corresponding galaxy catalog as a DataFrame.
        """
        ...

    @abstractmethod
    def load_dark_matter_catalog(self, *args, **kwargs) -> None:
        """Load the dark matter catalog from disk, to be implemented by geometry-specific subclasses."""
        ...

    @abstractmethod
    def get_dark_matter_catalog(self, *args, **kwargs) -> object:
        """Get the dark matter catalog from memory, to be implemented by geometry-specific subclasses."""
        ...

class SnapshotBackend(DarkMatterBackend):
    """
    Base for snapshot-based backends.

    Snapshot backends load one dark matter catalog per redshift, which maps
    naturally to N-body simulation suites.
    """

    def __init__(self) -> None:
        self._cache: dict[float, object] = {}  # Cache for dark matter catalogs

    @abstractmethod
    def load_dark_matter_catalog(
        self,
        redshift: float,
        **kwargs,
    ) -> None:
        """
        Load the dark matter catalog from disk for the specified redshift and tracers.

        Parameters
        ----------
        redshift : float
            Redshift at which to load the dark matter catalog.
        **kwargs
            Extra parameters to pass to the loader.
        """
        ...

    @abstractmethod
    def get_dark_matter_catalog(
        self,
        redshift: float,
    ) -> object:
        """
        Get the dark matter catalog from memory for the specified redshift.

        Parameters
        ----------
        redshift : float
            Redshift at which to retrieve the dark matter catalog.
        **kwargs
            Extra parameters to pass to the getter.

        Returns
        -------
        object
            The dark matter catalog for the specified redshift.
        """
        ...

    @property
    @abstractmethod
    def boxsize(self) -> float | list[float]:
        """Size of the simulation box in Mpc/h. Can be a single float (same for all dimensions) or a list of three floats for each dimension."""
        ...


# Create a registry for dark matter backends
_registry = BackendRegistry(DarkMatterBackend)
register_backend = _registry.register
load_backend = _registry.load
