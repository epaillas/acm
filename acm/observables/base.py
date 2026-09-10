"""Definition of the Observable product interface."""

import logging
from abc import ABC, abstractmethod
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self, overload

import numpy as np

from acm.utils.default import short_hash

from .model import ObservableModel

type Array2D = np.ndarray[tuple[int, int]]  # Short type alias for 2D NumPy arrays

logger = logging.getLogger(__name__)


def _format_filter_value(value) -> str:  # noqa: ANN001
    """Format a filter value for string representation."""
    if isinstance(value, slice):
        parts = [value.start, value.stop]
        if value.step is not None:
            parts.append(value.step)
        return "-".join([str(p) for p in parts])  # (start, stop, step)
    if isinstance(value, (list, tuple)):
        vals = list(value)
        if len(vals) > 2 and all(isinstance(v, (int, float)) for v in vals):
            steps = {b - a for a, b in pairwise(vals)}
            if len(steps) == 1:  # Arithmetic sequence --> (start, stop, step)
                parts = [f"{vals[0]:.3g}", f"{vals[-1]:.3g}"]
                step = steps.pop()
                if step != 1:
                    parts.append(f"{step:.3g}")
                return "-".join(parts)
        return ",".join(str(v) for v in vals)
    return str(value)


def make_handle(filters: dict, hlength: int | None = None) -> str:
    """Make a unique handle string from the given filters, optionally hashing if too long."""
    items = sorted(filters.items())
    parts = [f"{k}={_format_filter_value(v)}" for k, v in items]
    handle = "_".join(parts)
    if hlength is not None and len(handle) > hlength:
        handle = short_hash(handle, length=hlength)
    return handle


class Formatter[R](ABC):  # NOTE: splitting interface for clarity
    """
    Class handling the output formatting of the Observable product interface.

    R: Dynamic type parameter for raw data type, with or without selections applied.
    """

    def __init__(self) -> None:
        self._filters: dict = {}
        self._select: dict[str, list[int]] = {}

    @property
    def filters(self) -> dict:
        """Get the current filters."""
        return self._filters

    @filters.setter
    def filters(self, filters: dict) -> None:
        """Set the filters and update related values."""
        logger.debug(f"Filters set: {filters}")
        self._filters = filters

    def set_filters(self, **kwargs) -> None:
        """
        Explicitly set filters trough keyword arguments.

        Parameters
        ----------
        **kwargs
            Keyword arguments representing the filters to set.
            Allows slices, list of values or single values for filtering.
        """
        self.filters = kwargs  # Uses setter

    def set_selection(self, *names: str, indices: list[int]) -> None:
        """
        Set the indice selection on the last dimension of the 2D NumPy array output.

        Only applied on elements matching the registered names.

        Parameters
        ----------
        names: str
            Names of the elements to which the selection will be applied.
        indices: list[int]
            Indices to select from the last dimension of the 2D NumPy array output.

        Notes
        -----
        Calling this method with the same name will overwrite the previous selection.
        Calling this method with a new name will add it to the existing selections.
        Use :meth:`clear_filters` to reset all selections at once.
        """
        if len(indices) == 0:
            raise ValueError("Selection indices cannot be empty.")
        for n in names:
            self._select[n] = indices
        logger.debug(f"Selection set: {indices=}, {names=}")

    def clear_filters(self) -> None:
        """Clear all filters and selection indices."""
        self.filters = {}  # Uses setter
        self._select = {}
        logger.debug("All filters and selection indices cleared.")

    def get_handle(self, prefix: str | None = None, hlength: int | None = None) -> str:
        """
        Get a unique handle for the current instance based on its filters and selection.

        Parameters
        ----------
        prefix: str | None, optional
            An optional prefix to prepend to the handle. Defaults to None.
        hlength: int | None, optional
            The maximum length of the filter values before hashing. If the filter value
            string exceeds this length, it will be hashed and truncated to this length.
            Defaults to None.

        Returns
        -------
        str
            A unique string handle representing the current instance's filters and selection.

        Notes
        -----
        The handle is constructed by sorting the filters, formatting their values, and joining them.
        If the resulting handle exceeds the specified hash length, the filter values are hashed.
        """
        handle = make_handle(self.filters, hlength)
        if prefix is not None:
            handle = f"{prefix}_{handle}"
        return handle

    @abstractmethod
    def _apply_filters(self, data: R) -> R:
        """Apply the set filters to the provided data."""

    def _apply_selection(self, name: str, data: Array2D) -> Array2D:
        """
        Select specific indices from the last dimension of the array.

        Applicable only if the array is 1D or 2D, and on the names
        specified in the selection setup (see :meth:`set_select`).

        Parameters
        ----------
        name: str
            The name of the data variable, to elect the correct selection indices.
        data : np.ndarray[tuple[int, int]]
            The 2D NumPy array from which to select indices.

        Returns
        -------
        np.ndarray[tuple[int, int]]
            The selected NumPy array.

        Raises
        ------
        ValueError
            If the selection indices exceed the size of the last dimension of the array.
        """
        idx = self._select.get(name, None)
        if idx is not None and data.ndim < 3:
            ls = data.shape[-1]
            if ls <= max(idx):
                raise ValueError(f"Indices number exceed last dimension size {ls}.")
            logger.debug(f"Applying selection on {name}")
            return data[..., idx]
        return data

    @overload
    @staticmethod
    def _to_numpy(data: R, nested: Literal[False]) -> Array2D: ...
    @overload
    @staticmethod
    def _to_numpy(data: R, nested: bool = False) -> np.ndarray: ...
    @staticmethod
    @abstractmethod
    def _to_numpy(data: R, nested: bool = False):
        """
        Cast the provided data to a NumPy array.

        Parameters
        ----------
        data: R
            The object to cast on a numpy array.
        nested: bool
            If True, returns an unflattened array. Defaults to False (2D array)

        Returns
        -------
        np.ndarray
            The data cast to a 2D NumPy array, unless nested=True.
        """

    @overload
    def _format_data(self, data: R, name: str, nested: Literal[False]) -> Array2D: ...
    @overload
    def _format_data(self, data: R, name: str, nested: bool = False) -> np.ndarray: ...
    def _format_data(self, data: R, name: str, nested: bool = False):
        """
        Format the provided data by applying filters, casting to numpy and applying selection.

        Parameters
        ----------
        data: R
            The data to format.
        name: str
            The name of the data variable, used for selection.
        nested: bool, optional
            If True, the data is nested and will not be flattened. Default is False.

        Returns
        -------
        np.ndarray
            The formatted data as a 2D NumPy array, unless nested=True.
        """
        _data = self._apply_filters(data)
        _data = self._to_numpy(_data, nested=nested)
        if nested is False:
            _data = self._apply_selection(name, _data)
        return _data

    @overload
    def get_data(self, name: str, raw: Literal[True], nested: bool = False) -> R: ...
    @overload
    def get_data(
        self,
        name: str,
        raw: Literal[False] = False,
        nested: Literal[False] = False,
    ) -> Array2D: ...
    @overload
    def get_data(
        self,
        name: str,
        raw: Literal[False] = False,
        nested: bool = False,
    ) -> np.ndarray: ...
    @abstractmethod
    def get_data(self, name: str, raw: bool = False, nested: bool = False):
        """
        Get the data for the given name.

        Parameters
        ----------
        name: str
            The name of the data to retrieve.
        raw: bool, optional
            If True, return the raw unfiltered and unflattened data. Default is False.
        nested: bool, optional
            If True, return the data in its original unflattened form. Default is False.

        Raises
        ------
        KeyError
            If the specified name is not found in the data.
        """

    @overload
    def get_prediction(
        self,
        x: np.ndarray,
        raw: Literal[True],
        nested: bool = False,
    ) -> R: ...
    @overload
    def get_prediction(
        self,
        x: np.ndarray,
        raw: Literal[False] = False,
        nested: Literal[False] = False,
    ) -> Array2D: ...
    @overload
    def get_prediction(
        self,
        x: np.ndarray,
        raw: Literal[False] = False,
        nested: bool = False,
    ) -> np.ndarray: ...
    @abstractmethod
    def get_prediction(self, x: np.ndarray, raw: bool = False, nested: bool = False):
        """
        Get the prediction for the given input x.

        Parameters
        ----------
        x: np.ndarray
            The input data for which to get the prediction.
        raw: bool, optional
            If True, return the raw unfiltered and unflattened prediction. Default is False.
        nested: bool, optional
            If True, return the prediction in its original unflattened form. Default is False.
        """

    @overload
    def get_model_error(
        self,
        method: str,
        raw: Literal[True],
        nested: bool = False,
        **kwargs,
    ) -> R: ...
    @overload
    def get_model_error(
        self,
        method: str,
        raw: Literal[False] = False,
        nested: Literal[False] = False,
        **kwargs,
    ) -> np.ndarray[tuple[int]]: ...
    @overload
    def get_model_error(
        self,
        method: str,
        raw: Literal[False] = False,
        nested: bool = False,
        **kwargs,
    ) -> np.ndarray: ...
    @abstractmethod
    def get_model_error(self, method, raw=False, nested=False, **kwargs):
        """
        Wrap around :meth:`ObservableModel.get_error`.

        Parameters
        ----------
        method: str
            The method to use for the model error calculation.
        raw: bool, optional
            If True, return the raw unfiltered and unflattened error. Default is False.
        nested: bool, optional
            If True, return the error in its original unflattened form. Default is False.
        **kwargs
            Additional keyword arguments for the model error calculation.

        Returns
        -------
        R | np.ndarray
            The model error with filters and selection applied as a NumPy array, unless raw=True.
            Defaults to a 1D array of shape (n_features, ) unless nested=True,
            in which case the shape matches the structure of the nested data, and
            selection is skipped (see :meth:`set_selection`).
        """


# %% Product components
class BaseObservable[R](Formatter[R], ABC):
    """Base class defining the interface for all Observable classes."""

    def __init__(self, model: ObservableModel | None = None) -> None:
        super().__init__()
        self.model = model  # Public attribute

    def __copy__(self) -> Self:
        """Create a shallow copy of the class instance."""
        return self._copy(deep=False)

    def __deepcopy__(self, memo=None, **kwargs) -> Self:  # noqa: ANN001
        """Create a deep copy of the class instance."""
        return self._copy(deep=True, memo=memo, **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the observable instance."""
        shapes = {}
        for name in ("x", "y", "covariance_y"):
            try:
                shapes[name] = self.get_data(name).shape
            except KeyError:
                continue
        has_model = self.model is not None
        shape_str = ", ".join(f"{k}={v}" for k, v in shapes.items())
        sel = f", select={list(self._select)}" if self._select else ""
        return f"{type(self).__name__}({shape_str}, filters={self.filters}{sel}, {has_model=})"

    @abstractmethod
    def _copy(self, deep: bool = False, **kwargs) -> Self:
        """Create a copy of the current instance."""

    @classmethod
    @abstractmethod
    def load(cls, filename: str | Path, **kwargs) -> Self:
        """Load an observable instance from a file."""

    @classmethod
    @abstractmethod
    def can_load(cls, filename: str | Path) -> bool:
        """Determine if the class can load the given file."""

    @property
    @abstractmethod
    def x_names(self) -> list[str]:
        """List of the parameter names."""

    def get_covariance_matrix(
        self,
        volume_factor: float = 64,
        prefactor: float = 1.0,
    ) -> Array2D:
        """
        Get the data covariance matrix, infered from the covariance_y data object.

        Parameters
        ----------
        volume_factor : float, optional
            The volume factor to scale the covariance matrix. Defaults to 64.
        prefactor : float
            Prefactor to apply to the covariance matrix (e.g. Hartlap or Percival).
            Defaults to 1.0.

        Returns
        -------
        np.ndarray[tuple[int, int]]
            The covariance matrix, matching the selected filtering.

        Notes
        -----
        The covariance matrix is computed from the covariance_y data object with filters
        and selections applied before flattening the result on 2D (sample, features).
        """
        cov_y = self.get_data("covariance_y")
        cov = prefactor / volume_factor * np.cov(cov_y, rowvar=False)
        return cov

    @abstractmethod
    def get_model_covariance(self, prefactor: float = 1, **kwargs) -> Array2D:
        """Wrap around :meth:`ObservableModel.make_covariance` with formatting."""
