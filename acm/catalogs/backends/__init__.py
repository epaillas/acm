"""
Abstract backend interfaces for dark matter simulations.

A backend is responsible for two things:
  1. Loading a dark matter halo catalog from a simulation (get_dark_matter_catalog)
  2. Populating it with galaxies via an HOD or similar model (make_galaxy_catalog)

To implement a new backend, subclass SnapshotBackend or LightconeBackend
and register it with @register_backend("<name>").

Concrete backends depend on optional, heavyweight packages and are imported
lazily when load_backend is first called with their name.
"""

import importlib

from .base import DarkMatterBackend, SnapshotBackend, register_backend
from .base import load_backend as _load_backend

__all__ = [
    "DarkMatterBackend",
    "SnapshotBackend",
    "load_backend",
    "register_backend",
]

# Registry name -> module implementing it. Modules are imported on first use.
_LAZY_BACKENDS = {
    "AbacusHOD": f"{__name__}.abacus",
    "HODDIES": f"{__name__}.hoddies",
}


def load_backend(
    backend: str | DarkMatterBackend,
    *args,
    **kwargs,
) -> DarkMatterBackend:
    """Wrap :func:`base.load_backend` with lazy imports for concrete backends.

    Parameters
    ----------
    backend : str or DarkMatterBackend
        Registered backend name, or an already constructed backend instance.
    *args, **kwargs
        Forwarded to the backend constructor when ``backend`` is a string.

    Returns
    -------
    DarkMatterBackend
        The backend instance.
    """
    if isinstance(backend, str):
        module_name = _LAZY_BACKENDS.get(backend)
        if module_name is not None:
            importlib.import_module(module_name)
    return _load_backend(backend, *args, **kwargs)
