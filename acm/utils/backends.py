import logging
from typing import Callable, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BackendRegistry[T]:
    """
    A generic registry for loading backend classes by name.

    This class provides a reusable pattern for registering and loading
    backend implementations across different parts of the pipeline
    (e.g. dark matter backends, survey backends, etc.).

    Usage
    -----
    # Define a registry for a given base class
    backend_registry = BackendRegistry(MyBaseClass)

    # Register a backend
    @backend_registry.register("my_backend")
    class MyBackend(MyBaseClass):
        ...

    # Load a backend by name or pass through an existing instance
    backend = backend_registry.load("my_backend", *args, **kwargs)
    """

    def __init__(self, base_class: type[T]) -> None:
        """
        Initialize the registry with a specified base class.

        Parameters
        ----------
        base_class : type[T]
            The base class that all registered backends must inherit from.
        """
        self.base_class = base_class
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """
        Register a backend class under a given name.

        Parameters
        ----------
        name : str
            The name to register the backend class under.

        Raises
        ------
        TypeError
            If the class does not inherit from the base class.
        """
        def decorator(cls: type[T]) -> type[T]:
            if not issubclass(cls, self.base_class):
                raise TypeError(
                    f"{cls.__name__} must inherit from {self.base_class.__name__} to be registered."
                )
            if name in self._registry:
                logger.warning(f"Overwriting existing backend registration for name '{name}'.")
            self._registry[name] = cls
            return cls
        return decorator

    def load(self, backend: str | T, *args, **kwargs) -> T:
        """
        Load a registered backend by name or pass through an existing instance.

        Parameters
        ----------
        backend : str | T
            The name of the backend to load, or an existing instance.
        *args, **kwargs
            Forwarded to the backend constructor when backend is a string.

        Returns
        -------
        T
            An instance of the requested backend.

        Raises
        ------
        KeyError
            If no backend is registered under the given name.
        TypeError
            If backend is neither a string nor an instance of the base class.
        """
        if isinstance(backend, self.base_class):
            logger.info(f"Using provided backend instance: {backend.__class__.__name__}")
            return backend

        if isinstance(backend, str):
            if backend not in self._registry:
                raise KeyError(
                    f"Unknown backend '{backend}'. Available backends: {list(self._registry)}"
                )
            logger.info(f"Loading backend '{backend}'")
            return self._registry[backend](*args, **kwargs)

        raise TypeError(
            f"backend must be a string or a {self.base_class.__name__} instance, got {type(backend)}"
        )

    @property
    def available(self) -> list[str]:
        """List of registered backend names."""
        return list(self._registry)