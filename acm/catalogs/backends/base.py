"""
Abstract backend interfaces for dark matter simulations.

A backend is responsible for two things:
  1. Loading a dark matter halo catalog from a simulation (get_dark_matter_catalog)
  2. Populating it with galaxies via an HOD or similar model (make_galaxy_catalog)

To implement a new backend, subclass SnapshotBackend or LightconeBackend
and register it with @register_backend("<name>").
"""
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

from pandas import DataFrame

from ..dataclasses import Tracer

logger = logging.getLogger(__name__)

_BACKEND_REGISTRY = {}

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
    def get_dark_matter_catalog(self, *args, **kwargs) -> object: ...

class SnapshotBackend(DarkMatterBackend):
    """
    Base for snapshot-based backends.
    
    Snapshot backends load one dark matter catalog per redshift, which maps
    naturally to N-body simulation suites.
    """

    @abstractmethod
    def get_dark_matter_catalog(self, redshift: float, **kwargs) -> object:
        """
        Load the dark matter catalog for the specified redshift and tracers.

        Parameters
        ----------
        redshift : float
            Redshift at which to load the dark matter catalog.
        **kwargs
            Extra parameters to pass to the loader.

        Returns
        -------
        object
            The loaded dark matter catalog, in a format specific to the backend.
        """
        ...

def register_backend(
    name: str,
) -> Callable[[type[DarkMatterBackend]], type[DarkMatterBackend]]:
    """
    Decorator to register a dark matter backend class with a given name.
    This allows for easy retrieval of the backend class by name later on.

    Parameters
    ----------
    name : str
        The name to register the backend class under.
    """

    def decorator(cls: type[DarkMatterBackend]) -> type[DarkMatterBackend]:
        if not issubclass(cls, DarkMatterBackend):
            raise TypeError(
                f"Class {cls.__name__} must inherit from DarkMatterBackend to be registered."
            )
        if name in _BACKEND_REGISTRY:
            logger.warning(
                f"Overwriting existing backend registration for name '{name}'."
            )
        _BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def load_backend(
    backend: str | DarkMatterBackend, *args, **kwargs
) -> DarkMatterBackend:
    """
    Load a registered dark matter backend by name or pass trough an existing instance.

    Parameters
    ----------
    backend : str | DarkMatterBackend
        The name of the backend to load or an existing backend instance.
    *args
        Positional arguments to pass to the backend constructor.
    **kwargs
        Keyword arguments to pass to the backend constructor.

    Returns
    -------
    DarkMatterBackend
        An instance of the requested dark matter backend.

    Raises
    ------
    ValueError
        If no backend is registered under the given name.
    """
    if isinstance(backend, DarkMatterBackend):
        logger.info(f"Using provided backend instance: {backend.__class__.__name__}")
        return backend

    if isinstance(backend, str):
        if backend not in _BACKEND_REGISTRY:
            available = list(_BACKEND_REGISTRY.keys())
            raise KeyError(
                f"Unknown backend '{backend}'. Available backends: {available}"
            )
        logger.info(f"Loading backend '{backend}'")
        return _BACKEND_REGISTRY[backend](*args, **kwargs)

    raise TypeError(
        f"backend must be a string or a DarkMatterBackend instance, got {type(backend)}"
    )
