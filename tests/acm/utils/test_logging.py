"""Tests for logging utility functions."""
import logging
 
import pytest
 
from acm.utils.logging import (
    _get_logger_path,
    get_logger_for_script,
    suppress_logging,
)

class TestSuppressLogging:
    
    def test_suppresses_below_critical(self, caplog):
        """Messages below CRITICAL are silenced inside the context manager."""
        with caplog.at_level(logging.WARNING):
            with suppress_logging(enabled=True):
                logging.getLogger("test").warning("should be suppressed")
        assert caplog.records == []
        
    def test_restores_level_after_exit(self):
        """Root logger level is restored to its original value after the block."""
        root = logging.getLogger()
        original = root.getEffectiveLevel()
        with suppress_logging(enabled=True):
            pass
        assert root.getEffectiveLevel() == original
        
    def test_disabled_does_not_suppress(self, caplog):
        """enabled=False leaves logging untouched."""
        with caplog.at_level(logging.WARNING):
            with suppress_logging(enabled=False):
                logging.getLogger("test").warning("should appear")
        assert len(caplog.records) == 1
        
    def test_custom_highest_level(self, caplog):
        """highest_level=WARNING suppresses only DEBUG and INFO."""
        with caplog.at_level(logging.DEBUG):
            with suppress_logging(enabled=True, highest_level=logging.WARNING):
                logging.getLogger("test").debug("suppressed")
                logging.getLogger("test").info("suppressed")
        assert caplog.records == []


class TestGetLoggerPath:
 
    def test_path_inside_package(self):
        """A path containing the package name is converted to dotted module notation."""
        result = _get_logger_path("/home/user/mypackage/utils/logging.py", pkg_name="mypackage")
        assert result == "mypackage.utils.logging.py"
 
    def test_path_outside_package(self):
        """A path not containing the package name falls back to the script filename only."""
        result = _get_logger_path("/home/user/other_project/script.py", pkg_name="mypackage")
        assert result == "script.py"
 
    def test_nested_path_inside_package(self):
        """Deeply nested paths inside the package are fully resolved."""
        result = _get_logger_path("/home/user/mypackage/module/sub/script.py", pkg_name="mypackage")
        assert result == "mypackage.module.sub.script.py"
 
    def test_accepts_path_object(self, tmp_path):
        """Accepts a pathlib.Path object, not just a string."""
        p = tmp_path / "mypackage" / "utils" / "mymodule.py"
        result = _get_logger_path(p, pkg_name="mypackage")
        assert result == "mypackage.utils.mymodule.py"


class TestGetLoggerForScript:
 
    def test_returns_logger_instance(self, tmp_path):
        """Always returns a logging.Logger object."""
        p = tmp_path / "acm" / "myscript.py"
        assert isinstance(get_logger_for_script(p), logging.Logger)
 
    def test_logger_name_matches_path(self, tmp_path):
        """Logger name is derived from the file path via _get_logger_path."""
        p = tmp_path / "acm" / "myscript.py"
        logger = get_logger_for_script(p)
        assert "acm" in logger.name
        assert "myscript" in logger.name
 
    def test_different_scripts_get_different_loggers(self, tmp_path):
        """Two different script paths produce two distinct loggers."""
        p1 = tmp_path / "acm" / "script_a.py"
        p2 = tmp_path / "acm" / "script_b.py"
        assert get_logger_for_script(p1).name != get_logger_for_script(p2).name