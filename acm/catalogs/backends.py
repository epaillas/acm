import logging
from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Callable

from .dataclasses import Tracer

logger = logging.getLogger(__name__)

_BACKEND_REGISTRY = {}


class DarkMatterBackend(ABC):
    """
    Backend to load the dark matter catalog and populate the galaxy catalog.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the dark matter backend with any necessary parameters.

        This method can be used to set up connections, load configuration files,
        or perform any other setup required to access the dark matter snapshots.
        DO NOT load the dark matter snapshots here, as this should be done
        in the `get_dark_matter_catalog` method to allow for lazy loading of the data.
        """
        pass

    @abstractmethod
    def get_dark_matter_catalog(self, **kwargs):
        """
        Get the dark matter catalog based on the provided parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def make_galaxy_catalog(
        self,
        dm_catalog: Any,
        tracers: list[Tracer], 
        **kwargs,
    ) -> list[Tracer, Any]:
        """
        Populate the galaxy catalog based on the provided parameters.
        """
        raise NotImplementedError


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
    backend: str | DarkMatterBackend, 
    *args, 
    **kwargs
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


@register_backend("AbacusHOD")
class AbacusHODBackend(DarkMatterBackend):
    pass
