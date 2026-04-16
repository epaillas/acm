from importlib.metadata import PackageNotFoundError, version

from .utils.logging import setup_logging

try:
    __version__ = version("acm")
except PackageNotFoundError:
    __version__ = "unknown"
