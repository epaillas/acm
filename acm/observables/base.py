"""Definition of the Observable product interface."""
import logging
from abc import ABC, abstractmethod
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self, overload

import numpy as np

from acm.utils.default import short_hash

from .model import ObservableModel

logger = logging.getLogger(__name__)

def _format_filter_value(value) -> str:  # noqa: ANN001
    """Format a filter value for string representation."""
    if isinstance(value, slice):
        parts = [value.start, value.stop]
        if value.step is not None:
            parts.append(value.step)
        return "-".join([str(p) for p in parts]) # (start, stop, step)
    if isinstance(value, (list, tuple)):
        vals = list(value)
        if len(vals) > 2 and all(isinstance(v, (int, float)) for v in vals):
            steps = {b-a for a, b in pairwise(vals)}
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

class Formatter[R, N](ABC): # NOTE: splitting interface for clarity
    """
    Class handling the output formatting of the Observable product interface.

    Dynamic type parameters:
    - R: Raw data type, with or without selections applied.
    - N: 2D NumPy array type for the output.
    """

    def __init__(self) -> None:
        self._filters: dict = {}
        self._select: list[int] | None = None
        self._select_names: list[str] = []

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
        self.filters = kwargs # Uses setter

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
        """
        self._select = indices
        self._select_names = names
        logger.debug(f"Selection set: {indices=}, {names=}")

    def clear_filters(self) -> None:
        """Clear all filters and selection indices."""
        self._filters.clear()
        self._select = None
        self._select_names = []
        logger.debug("All filters and selection indices cleared.")

    @abstractmethod
    def _apply_filters(self, data: R) -> R:
        """Apply the set filters to the provided data."""

    @abstractmethod
    def _apply_selection(self, data: N, name: str) -> N:
        """Select specified indices from a 2D NumPy array."""

    @staticmethod
    @abstractmethod
    def _to_numpy(data: R, nested: bool = False) -> N:
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
        N
            The data cast to a 2D NumPy array, unless nested=True.
        """

    def _format_data(self, data: R, name: str, nested: bool = False) -> N:
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
        data = self._apply_filters(data)
        data = self._to_numpy(data, nested=nested)
        if not nested:
            data = self._apply_selection(data, name)
        return data

    @overload
    def get_data(self, name: str, raw: Literal[True], nested: bool = False) -> R:
        ...
    @overload
    def get_data(self,
        name: str,
        raw: Literal[False] = False,
        nested: bool = False,
    ) -> N:
        ...
    @abstractmethod
    def get_data(self, name, raw = False, nested = False):
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
        """

    @overload
    def get_prediction(self, x: N, raw: Literal[True], nested: bool = False) -> R:
        ...
    @overload
    def get_prediction(self,
        x: N,
        raw: Literal[False] = False,
        nested: bool = False,
    ) -> N:
        ...
    @abstractmethod
    def get_prediction(self, x: N, raw = False, nested = False):
        """
        Get the prediction for the given input x.

        Parameters
        ----------
        x: N
            The input data for which to get the prediction.
        raw: bool, optional
            If True, return the raw unfiltered and unflattened prediction. Default is False.
        nested: bool, optional
            If True, return the prediction in its original unflattened form. Default is False.
        """
        # TODO: Final transform (e.g. phase correction) - Here or in model?

    @overload
    def get_model_error(
        self,
        method: str,
        raw: Literal[True],
        nested: bool = False,
        **kwargs,
    ) -> R:
        ...
    @overload
    def get_model_error(
        self,
        method: str,
        raw: Literal[False] = False,
        nested: Literal[False] = False,
        **kwargs,
    ) -> N:
        ...
    @abstractmethod
    def get_model_error(self, method, raw = False, nested = False, **kwargs):
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
        """


#%% Product components
class BaseObservable[R, N](Formatter[R, N], ABC):
    """Base class defining the interface for all Observable classes."""

    def __init__(self, model: ObservableModel | None = None) -> None:
        super().__init__()
        self.model = model # Public attribute

    def __copy__(self) -> Self:
        """Create a shallow copy of the class instance."""
        return self._copy(deep=False)

    def __deepcopy__(self, **kwargs) -> Self:
        """Create a deep copy of the class instance."""
        return self._copy(deep=True, **kwargs)

    # def __repr__(self) -> str: # TODO

    def get_handle(self, name: str | None = None, hlength: int | None = None) -> str:
        """
        Get a unique handle for the current instance based on its filters and selection.

        Parameters
        ----------
        name: str | None, optional
            An optional name to prepend to the handle. Defaults to None.
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
        if name is not None:
            handle = f"{name}_{handle}"
        return handle

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
    ) -> np.ndarray:
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
        np.ndarray
            The covariance matrix, matching the selected filtering.

        Notes
        -----
        The covariance matrix is computed from the covariance_y data object with filters
        and selections applied before flattening the result on 2D (sample, features).
        """
        cov_y = self.get_data("covariance_y", raw=True)
        cov_y = self._apply_filters(cov_y)
        cov_y = self._to_numpy(cov_y)
        cov_y = self._apply_selection(cov_y, "y")
        cov = prefactor / volume_factor * np.cov(cov_y, rowvar=False)
        return cov

    @abstractmethod
    def get_model_covariance(self, prefactor: float = 1, **kwargs) -> np.ndarray:
        """Wrap around :meth:`ObservableModel.make_covariance` with formatting."""
