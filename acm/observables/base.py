"""Definition of the Observable product interface."""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Self, overload

import numpy as np

from .model import ObservableModel

logger = logging.getLogger(__name__)

class Formatter[R, T](ABC): # NOTE: splitting interface for clarity
    """Class handling the output formatting of the Observable product interface."""

    def __init__(self, **kwargs) -> None:
        self.set_output(**kwargs)
        self._filters: dict = {}
        self._select: list[int] | None = None
        self._select_names: list[str] = []

    def set_output(
        self,
        numpy: bool = False,
        squeeze: bool = True,
        flatten: int | None = None,
    ) -> None:
        """
        Set the output format of the current instance.

        Parameters
        ----------
        numpy : bool, optional
            If True, the output will be a NumPy array.
            If False, the output will match the data type. Defaults to False.
        squeeze : bool, optional
            If True, the output will be squeezed to remove single-dimensional entries.
            Defaults to True.
        flatten : int or None, optional
            Flatten to the specified number of dimensions.
            Currently supports 1D and 2D flattening.
            Otherwise, no flattening is applied. Defaults to None.

        Raises
        ------
        ValueError
            If flatten is not None and not in [1, 2].
        """
        if flatten is not None and flatten not in [1, 2]:
            raise ValueError("Flattening is only supported for 1D and 2D outputs.")
        logger.info(f"Setting output format: {numpy=}, {squeeze=}, {flatten=}")
        self._output_numpy = numpy
        self._output_squeeze = squeeze
        self._output_flatten = flatten

    def set_filters(self, **kwargs) -> None:
        """
        Set filters for the current instance.

        Parameters
        ----------
        **kwargs
            Keyword arguments specifying the filter criteria.

        Examples
        --------
        >>> obs.set_filter(cosmo=0, ells=[0, 2], s=slice(0, 10))
        # Will select the coordinates 'cosmo' 0, 'ells' 0 or 2 and 's' in [0, 10].
        """
        self._filters = kwargs
        logger.info(f"Filter set: {kwargs}")

    def set_select(self, *names: str, indices: list[int]) -> None:
        """Set selection indices for 1D or 2D-formatted outputs."""
        self._select_names = list(names)
        self._select = indices
        logger.info(f"Selection indices set: {indices}")

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
    def _apply_selection(self, data: R) -> R:
        """Apply the set selection indices to the provided data."""

    @abstractmethod
    @staticmethod
    def _flatten(data: R, ndim: int) -> R:
        """Flatten the provided data to the specified number of dimensions."""

    @abstractmethod
    @staticmethod
    def _squeeze(data: R) -> R:
        """Squeeze the provided data."""

    @abstractmethod
    @staticmethod
    def _to_numpy(data: R) -> np.ndarray:
        """Cast the provided data to a NumPy array."""

    def _format_data(self, data: R) -> R | np.ndarray:
        """
        Format the provided data according to the set filters, selection, and output.

        Parameters
        ----------
        data: R
            The data to format.

        Returns
        -------
        R or np.ndarray
            The formatted data, either as the original type or as a NumPy array.
        """
        out = self._apply_filters(data)
        if self._output_flatten is not None:
            out = self._flatten(out, ndim=self._output_flatten)
        out = self._apply_selection(out)
        if self._output_squeeze:
            out = self._squeeze(out)
        if self._output_numpy:
            out = self._to_numpy(out)
        return out

    @overload
    def get_data(self, name: str, raw: Literal[True]) -> R: ...
    @overload
    def get_data(self, name: str, raw: Literal[False] = False) -> T: ...
    @abstractmethod
    def get_data(self, name: str, raw: bool = False) -> T:
        """Get the matching data object of the observable instance."""

    @overload
    def get_prediction(self, x: T, raw: Literal[True]) -> R: ...
    @overload
    def get_prediction(self, x: T, raw: Literal[False] = False) -> T: ...
    @abstractmethod
    def get_prediction(self, x: T, raw: bool = False) -> T:
        """Wrap around :meth:`ObservableModel.get_prediction` with formatting."""

    @overload
    def get_model_error(self, method: str, raw: Literal[True], **kwargs) -> R: ...
    @overload
    def get_model_error(self, method: str, raw: Literal[False] = False, **kwargs) -> T: ...
    @abstractmethod
    def get_model_error(self, method: str, raw: bool = False, **kwargs) -> T:
        """Wrap around :meth:`ObservableModel.get_error` with formatting."""

    @abstractmethod
    def get_model_covariance(self, prefactor: float = 1, **kwargs) -> np.ndarray:
        """Wrap around :meth:`ObservableModel.make_covariance` with formatting."""


#%% Product components
class BaseObservable[R, T](Formatter[R, T], ABC):
    """
    Base class defining the interface for all Observable classes.

    Dynamically typed with two type variables:
    - R: The raw data type (e.g., :class:`xarray.DataArray`, etc.)
    - T: The default data type, usually an union between `R` and :class:`numpy.ndarray`
    """

    def __init__(self, model: ObservableModel | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model # Public attribute

    def __copy__(self) -> Self:
        """Create a shallow copy of the class instance."""
        return self._copy(deep=False)

    def __deepcopy__(self, **kwargs) -> Self:
        """Create a deep copy of the class instance."""
        return self._copy(deep=True, **kwargs)

    # def __repr__(self) -> str:

    # def get_handle(self) -> str:

    @abstractmethod
    def _copy(self, deep: bool = False, **kwargs) -> Self:
        """Create a copy of the current instance."""

    @abstractmethod
    @classmethod
    def load(cls, filename: str | Path, **kwargs) -> Self:
        """Load an observable instance from a file."""

    @abstractmethod
    @classmethod
    def can_load(cls, filename: str | Path) -> bool:
        """Determine if the class can load the given file."""

    @abstractmethod
    @property
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
        cov_y = self._apply_selection(cov_y)
        cov_y = self._flatten(cov_y, ndim=2)
        cov_y = self._to_numpy(cov_y)
        cov = prefactor / volume_factor * np.cov(cov_y, rowvar=False)
        return cov
