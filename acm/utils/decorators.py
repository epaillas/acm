from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from acm.utils.default import is_nersc

T = TypeVar('T') # Type variable for class methods

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


def require_nersc(enabled: bool = True) -> Callable:
    """Restrict a function execution to NERSC environments."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> object:
            if enabled and not is_nersc:
                fname = getattr(func, '__name__', 'unknown')
                raise OSError(
                    f"The function '{fname}' can only be executed in a NERSC environment."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
