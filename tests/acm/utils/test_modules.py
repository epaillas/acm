from unittest.mock import MagicMock, patch

import pytest

from acm.utils.modules import check_installed, get_class_from_module

# ruff: noqa: ANN201, D101, D102, INP001, S101


class TestGetClassFromModule:

    def test_returns_builtin_class(self):
        """Test that the function can retrieve a built-in class."""
        cls = get_class_from_module("builtins", "int")
        assert cls is int

    def test_raises_on_missing_module(self):
        """Test that the function raises an error when the module cannot be found."""
        with pytest.raises(ModuleNotFoundError):
            get_class_from_module("totally.fake.module", "SomeClass")

    def test_raises_on_missing_attribute(self):
        """Test that the function raises an error when the class cannot be found in the module."""
        with pytest.raises(AttributeError):
            get_class_from_module("builtins", "NonExistentClass")

    def test_returns_class_from_mocked_module(self):
        """Class returned is exactly the one attached to the mock module."""
        mock_module = MagicMock()
        mock_class = MagicMock()
        mock_module.MyClass = mock_class

        with patch("importlib.import_module", return_value=mock_module):
            result = get_class_from_module("fake.module", "MyClass")

        assert result is mock_class

    def test_import_module_called_with_correct_path(self):
        """importlib.import_module is called with the exact module path provided."""
        mock_module = MagicMock()

        with patch("importlib.import_module", return_value=mock_module) as mock_import:
            get_class_from_module("some.module.path", "AnyAttr")

        mock_import.assert_called_once_with("some.module.path")

class TestCheckInstalled:
    def test_installed_package(self):
        assert check_installed("os") is True

    def test_missing_package(self):
        assert check_installed("definitely_not_a_real_package") is False

    def test_several_packages(self):
        assert check_installed("os", "math", "csv")

    def test_missing_among_list(self):
        assert check_installed("os", "iswearimapackage", 'csv') is False
