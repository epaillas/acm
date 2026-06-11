"""Utilities for working with ACM."""

from importlib.metadata import PackageNotFoundError, version

from .utils.logging import get_logger_for_script, setup_logging

try:
    __version__ = version("acm")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"