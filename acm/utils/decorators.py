from functools import wraps
from acm.utils.default import is_nersc

def temporary_class_state(**attrs):
    """
    Decorator factory to temporarily modify class attributes during a method call.
    Restores original values after method execution (even if exceptions occur).
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
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

def require_nersc(enabled: bool = True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enabled and not is_nersc:
                raise EnvironmentError(
                    f"The function '{func.__name__}' can only be executed in a NERSC environment."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator