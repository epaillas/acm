import os
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from acm.utils.default import is_nersc

T = TypeVar("T")  # Type variable for class methods

def temporary_class_state(**attrs) -> Callable:
    """
    Temporarily modify class attributes during a method call.

    Restores original values after method execution (even if exceptions occur).
    """
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self: T, *args, **kwargs) -> T:
            # Save original values
            original_attrs = {key: getattr(self, key) for key in attrs}
            for key, value in attrs.items():
                setattr(self, key, value)

            try:
                return method(self, *args, **kwargs)
            finally:
                # Restore originals
                for key, value in original_attrs.items():
                    setattr(self, key, value)
        return wrapper
    return decorator

# Provides a global toggle for NERSC-only function restrictions, defaulting to enabled.
ENABLE_NERSC = os.getenv("ACM_ENABLE_NERSC_ONLY", "1") == "1"

def require_nersc(enabled: bool = ENABLE_NERSC) -> Callable:
    """Restrict a function execution to NERSC environments."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> object:
            if enabled and not is_nersc:
                fname = getattr(func, "__name__", "unknown")
                raise OSError(
                    f"The function '{fname}' can only be executed in a NERSC environment."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def kwargs_alias(**aliases: str) -> Callable:
    """
    Resolve keyword argument aliases before passing them to a function.

    Parameters
    ----------
    **aliases: str
        Mapping of canonical names to alias strings.

    Examples
    --------
    >>> @kwargs_aliases(canonical="old_alias")
    ... def make_galaxy_catalog(self, ..., old_alias=2, **kwargs):
    ...    var = canonical  # 'canonical' will be set to the value of 'old_alias' if provided
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> object:
            for canonical, alias in aliases.items():
                if alias in kwargs and canonical in kwargs:
                    raise ValueError(
                        f"{func.__name__} cannot use both '{canonical}' and '{alias}' as arguments."
                    )
                elif alias in kwargs:
                    kwargs[canonical] = kwargs.pop(alias)
            return func(*args, **kwargs)
        return wrapper
    return decorator