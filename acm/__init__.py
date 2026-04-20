from importlib.metadata import PackageNotFoundError, version

from .utils.logging import get_logger_for_script, setup_logging

try:
    __version__ = version("acm")
except PackageNotFoundError:
    __version__ = "unknown"

# This is a change to test the CI workflows with paths, it should be removed after testing is complete 