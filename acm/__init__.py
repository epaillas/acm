from importlib.metadata import PackageNotFoundError, version

from .utils.logging import get_logger_for_script, setup_logging

try:
    __version__ = version("acm")
except PackageNotFoundError:
    __version__ = "unknown"


def f(x: int = "hello") -> int:  # This should be flagged by ty
    return x
