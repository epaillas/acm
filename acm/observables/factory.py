"""
Factory pattern implementation for observables in the ACM package.

See https://realpython.com/factory-method-python/.
"""
from pathlib import Path

from .base import BaseObservable
from .lsstypes import LsstypesObservable
from .xarray import XarrayObservable


#%% Creator components
class ObservableFactory[S: BaseObservable]:
    """Factory class for identifying which observable class to use."""

    def __init__(self) -> None:
        """Initialize the ObservableFactory."""
        self._creators: dict[str, type[S]] = {}

    def register_observable(self, backend: str, creator: type[S]) -> None:
        """Register a new observable creator for a specific backend."""
        self._creators[backend] = creator

    def get_observable(self, backend: str) -> type[S]:
        """Return the observable class based on the specified backend."""
        creator = self._creators.get(backend)
        if not creator:
            raise ValueError(f"Unsupported backend: {backend}")
        return creator

    def get_loader(self, filename: str | Path) -> type[S]:
        """Return the observable class that can load the given file."""
        for creator in self._creators.values():
            if creator.can_load(filename):
                return creator
        raise ValueError(f"Unsupported file extension for: {filename}")

factory = ObservableFactory()
factory.register_observable("xarray", XarrayObservable)
factory.register_observable("lsstypes", LsstypesObservable)

#%% Client components
class Observable[S: BaseObservable]:
    """Factory class for creating observables based on the backend choice."""

    def __new__(cls, *args, backend: str = "xarray", **kwargs) -> S:
        """Create an observable instance based on the specified backend."""
        obs_cls: type[S] = factory.get_observable(backend)
        return obs_cls(*args, **kwargs)

    @classmethod
    def load(cls, filename: str | Path, **kwargs) -> S:
        """Load an observable instance from a file, automatically determining the appropriate class."""
        obs_cls: type[S] = factory.get_loader(filename)
        return obs_cls.load(filename, **kwargs)
