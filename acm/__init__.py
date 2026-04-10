from .utils.logging import setup_logging

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("acm")
except PackageNotFoundError:
    __version__ = "unknown"