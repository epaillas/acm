"""Concrete implementation for the BaseObservable product for the xarray interface."""
import logging
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Literal, Self, overload

import numpy as np
import xarray as xr

from acm.utils.logging import suppress_logging
from acm.utils.xarray import dataset_from_dict

from .base import Array2D, BaseObservable
from .model import ObservableModel

type _ArrayLike = xr.DataArray | np.ndarray

logger = logging.getLogger(__name__)

def _load_dataset(filename: str | Path) -> xr.Dataset:
    """Load a compressed XarrayObservable file and model."""
    filename = Path(filename)
    if filename.suffix == ".npy": # Legacy files
        return dataset_from_dict(np.load(filename, allow_pickle=True).item())
    return xr.load_dataset(filename) # Should be h5, keeping engine free just in case

def _is_valid_dataset(ds: xr.Dataset) -> bool:
    """Check if the dataset has the required variables and coordinates."""
    required_vars = {"x", "y"}
    required_attrs = {"sample", "features"}
    if not required_vars.issubset(ds.data_vars):
        logger.debug(f"Dataset missing required variables: {required_vars - set(ds.data_vars)}")
        return False
    for da in ds.data_vars.values():
        if not required_attrs.issubset(da.attrs):
            logger.debug(f"DataArray '{da.name}' missing required attributes: {required_attrs - set(da.attrs)}")
            return False
    return True

def _stack_on(new: str, da: xr.DataArray, *dims: str) -> xr.DataArray:
    """
    Stack a DataArray on a list of dimensions.

    Parameters
    ----------
    new: str
        The name of the stacked dimension.
    da: xr.DataArray
        The DataArray on which dimensions are stacked.
    *dims: str
        The dimension names to stack in the 'new' dimension.

    Returns
    -------
    xr.DataArray
        The modified DataArray, with all *dims stacked in a 'new' MultiIndex dimension.
        If *dims is empty, will create a 'new' 1D dimension instead.

    Notes
    -----
    The initial DataArray is returned if 'new' is already an existing dimension.
    """
    if new in da.dims:
        return da
    if len(dims) == 0:
        return da.expand_dims(new)
    return da.stack({new: dims})

def format_like(da: xr.DataArray, arr: np.ndarray, new: str = "dim0") -> xr.DataArray:
    """
    Format a NumPy array to have the same shape and coordinates as a given xarray DataArray.

    Parameters
    ----------
    da : xr.DataArray
        The reference DataArray whose shape and coordinates will be used.
        Must have an attribute "features" that specifies the feature dimensions to reshape on.
    arr : np.ndarray
        The NumPy array to be reshaped and formatted.
    new : str, optional
        The name of the new dimension to be added to the reshaped array. Defaults to "sample".

    Returns
    -------
    xr.DataArray
        A new DataArray with the reshaped array, the new dimension, and the same coordinates as the reference DataArray.
    """
    feat_dims = da.attrs["features"] # NOTE: KeyError if "features" is missing
    data = arr.reshape(-1, *[da.sizes[d] for d in feat_dims])
    return xr.DataArray(
        data = data,
        dims = [new, *feat_dims],
        coords = {d: da.coords[d] for d in feat_dims}, # new will just be indexed
        attrs = {"sample": [new], "features": feat_dims},
        name = "like_" + str(da.name) if da.name is not None else None,
    )


class XarrayObservable(BaseObservable[xr.DataArray]):
    """
    Implementation of the :class:`BaseObservable` interface for xarray datasets and data arrays.

    Requires a dataset with at least two data variables: "x" for parameters and "y" for truth values.
    Each data variable must have attributes "sample" and "features" that specify the dimensions to
    stack on for flattening to 2D arrays.
    """

    def __init__(
        self,
        data: xr.Dataset,
        model: ObservableModel | None = None,
        silent_load: bool = False,
    ) -> None:
        """
        Initialize an XarrayObservable instance.

        Parameters
        ----------
        dataset : xr.Dataset
            The xarray Dataset containing the observable data.
        model : ObservableModel, optional
            The model associated with the observable. Defaults to None.
        silent_load : bool, optional
            If True, suppress logging during initialization. Defaults to False.
        **kwargs
            Additional keyword arguments to set output format. See :meth:`set_output` for details.
        """
        self._dataset = data
        names = list(data.data_vars)
        with suppress_logging(enabled=silent_load):
            logger.info(f"Datasets loaded with the following variables: {names}")
            super().__init__(model=model)

    @classmethod
    def load(cls, filename: str | Path, **kwargs) -> Self:
        """Load a compressed XarrayObservable file and model."""
        data = _load_dataset(filename)
        if not _is_valid_dataset(data):
            raise ValueError(f"Invalid Observable structure in file: {filename}")
        return cls(data, **kwargs)

    @classmethod
    def can_load(cls, filename: str | Path) -> bool:
        """Determine if the class can load the given file."""
        try:
            data = _load_dataset(filename)
            return _is_valid_dataset(data)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to load Observable from {filename}: {e}")
        return False

    def _copy(self, deep: bool = True, **kwargs) -> Self:
        cp = deepcopy if deep else copy
        new = self.__class__(data = cp(self._dataset, **kwargs), silent_load = True)
        cv = vars(self)
        for k, v in cv.items():
            setattr(new, k, cp(v, **kwargs))
        return new

    def get_coordinate_list(self, name: str) -> list:
        """
        Get the filtered values of the dataset coordinates.

        Parameters
        ----------
        name: str
            Name of the coordinate to retrieve.

        Returns
        -------
        list
            The list of values of the specified coordinate.

        Raises
        ------
        KeyError
            If the required name is not present in the coordinates of the filtered dataset.
        """
        coords: xr.Coordinates = self.coords # Filtered coordinates
        if name not in coords:
            raise KeyError(f"{name} not found in coordinates {list(coords)}")
        return coords[name].to_numpy().tolist()

    @property
    def x_names(self) -> list[str]:
        """List of the parameter names."""
        return self.get_coordinate_list("parameters")

    @staticmethod
    def _drop_nan_dims(da: xr.DataArray) -> xr.DataArray:
        """
        Drop dimensions from the DataArray that are marked as NaN in the attributes.

        Parameters
        ----------
        da : xr.DataArray
            The DataArray from which NaN dimensions will be dropped.

        Returns
        -------
        xr.DataArray
            The DataArray with NaN dimensions dropped. If all values are dropped, a warning is logged.
        """
        nan_dims = da.attrs.get("nan_dims", [])
        for d in nan_dims: # Loop because multi-dims not supported by dropna
            if d in da.dims:
                da = da.dropna(d, how="all")
        if nan_dims and da.size == 0:
            logger.warning(f"All values of '{da.name}' dropped due to NaN dimensions.")
        return da

    @overload
    def _apply_filters(self, data: xr.Dataset) -> xr.Dataset: ...
    @overload
    def _apply_filters(self, data: xr.DataArray) -> xr.DataArray: ...

    def _apply_filters(
        self,
        data: xr.Dataset | xr.DataArray,
    ) -> xr.Dataset | xr.DataArray:
        """
        Apply the set filters to the provided data.

        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The data to which the filters will be applied.

        Returns
        -------
        xr.Dataset or xr.DataArray
            The filtered data.
        """
        dims = set(data.dims)
        fdims = set(self._filters)
        if not fdims.issubset(dims):
            logger.warning(
                f"Filter dimensions {fdims - dims} are not present in the data dimensions {dims}."
            )
        subset_filters = {k: v for k, v in self._filters.items() if k in dims}
        logger.debug(f"Applying filters: {subset_filters}")
        return data.sel(**subset_filters)

    def _apply_selection(self, data: Array2D, name: str) -> Array2D:
        """
        Select specific indices from the last dimension of the array.

        Applicable only if the array is 1D or 2D, and on the names
        specified in the selection setup (see :meth:`set_select`).

        Also accepts objects with names prefixed by "like_" and accepted names
        (e.g., "like_y" if "y" is in the selection names).

        Parameters
        ----------
        data : np.ndarray[tuple[int, int]]
            The 2D NumPy array from which to select indices.
        name : str
            The name of the DataArray, used to determine if selection should be applied.

        Returns
        -------
        np.ndarray[tuple[int, int]]
            The selected NumPy array.

        Raises
        ------
        ValueError
            If the selection indices exceed the size of the last dimension of the array.
        """
        like_names = ["like_" + name for name in self._select_names] # e.g. "like_y"
        ok_names = set(self._select_names + like_names)
        if self._select is not None and data.ndim < 3 and name in ok_names:
            ls = data.shape[-1]
            if ls <= max(self._select):
                raise ValueError(f"Indices number exceed last dimension size {ls}.")
            return data[..., self._select]
        logger.debug(f"No selection applied to DataArray '{name}'.")
        return data

    @overload
    @staticmethod
    def _to_numpy(data: xr.DataArray, nested: Literal[True]) -> np.ndarray: ...
    @overload
    @staticmethod
    def _to_numpy(data: xr.DataArray, nested: Literal[False] = False) -> Array2D: ...
    @staticmethod
    def _to_numpy(data: xr.DataArray, nested: bool = False):
        """
        Cast the provided DataArray to a NumPy array.

        Parameters
        ----------
        data: xr.DataArray
            The object to cast on a numpy array.
        nested: bool
            If True, returns an unflattened array. Defaults to False (2D array)

        Returns
        -------
        np.ndarray
            The data cast to a 2D NumPy array, unless nested=True.
        """
        if nested is False:
            data = _stack_on("sample", data, *data.attrs["sample"])
            data = _stack_on("features", data, *data.attrs["features"])
            data= data.transpose("sample", "features") # Ensure correct dim order
        return data.to_numpy()

    @overload
    def _format_data(
        self,
        data: xr.DataArray,
        name: str,
        nested: Literal[True],
    ) -> np.ndarray:
        ...
    @overload
    def _format_data(
        self,
        data: xr.DataArray,
        name: str,
        nested: Literal[False] = False,
    ) -> Array2D:
        ...
    def _format_data(self, data: xr.DataArray, name: str, nested: bool = False):
        """
        Format the provided DataArray by applying filters, casting to numpy and applying selection.

        Parameters
        ----------
        data: xr.DataArray
            The DataArray to format.
        name: str
            The name of the DataArray, used for selection.
        nested: bool, optional
            If True, the data is nested and will not be flattened. Default is False.

        Returns
        -------
        np.ndarray
            The formatted data as a 2D NumPy array, unless nested=True.
        """
        out = self._drop_nan_dims(data)
        out = super()._format_data(out, name, nested)
        return out

    @overload
    def get_data(
        self,
        name: str,
        raw: Literal[True],
        nested: bool = False,
    ) -> xr.DataArray:
        ...
    @overload
    def get_data(
        self,
        name: str,
        raw: Literal[False],
        nested: Literal[True],
    ) -> np.ndarray:
        ...
    @overload
    def get_data(
        self,
        name: str,
        raw: Literal[False] = False,
        nested: Literal[False] = False,
    ) -> Array2D:
        ...
    def get_data(self, name: str, raw: bool = False, nested: bool = False):
        """
        Get the data variable from the dataset from the given name.

        Parameters
        ----------
        name : str
            The name of the data variable to retrieve.
        raw : bool, optional
            If True, return the raw DataArray without applying filters, selection, or numpy conversion.
            Defaults to False.
        nested : bool, optional
            If True, return the data in its original unflattened form. Default is False.

        Returns
        -------
        xr.DataArray or np.ndarray
            The requested data variable, formatted according to the output settings.

        Raises
        ------
        KeyError
            If the specified data variable name is not found in the dataset.
        """
        if name not in self._dataset.data_vars:
            raise KeyError(f"Data variable '{name}' not found in the dataset.")
        da = self._dataset[name]
        if raw:
            return da
        return self._format_data(da, name, nested)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Get an attribute from the dataset, with filters applied."""
        data = self._dataset
        if name in data.data_vars:
            return self.get_data(name)
        if not hasattr(data, name): # Early check before filtering
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self._apply_filters(data), name)

    @overload
    def get_prediction(
        self,
        x: _ArrayLike,
        raw: Literal[True],
        nested: bool = False,
    ) -> xr.DataArray:
        ...
    @overload
    def get_prediction(self,
        x: _ArrayLike,
        raw: Literal[False],
        nested: Literal[True],
    ) -> np.ndarray:
        ...
    @overload
    def get_prediction(self,
        x: _ArrayLike,
        raw: Literal[False] = False,
        nested: Literal[False] = False,
    ) -> Array2D:
        ...
    def get_prediction(self, x: _ArrayLike, raw: bool = False, nested: bool = False):
        """
        Get the prediction for the given input x.

        Parameters
        ----------
        x : xr.DataArray or np.ndarray
            The input data for which to get the prediction.
        raw : bool, optional
            If True, return the raw unfiltered and unflattened prediction. Default is False.
        nested : bool, optional
            If True, return the data in its original unflattened form. Default is False.

        Returns
        -------
        xr.DataArray or np.ndarray
            The model prediction, formatted according to the output settings.

        Raises
        ------
        AttributeError
            If no model has been registered.
        """
        if self.model is None:
            raise AttributeError("No model has been registered.")
        pred = self.model.get_prediction(np.asarray(x)) # asarray = faster torch.Tensor
        y = self.get_data("y", raw=True)
        pred = format_like(da=y, arr=pred, new="n_pred")
        if raw:
            return pred
        return self._format_data(pred, str(pred.name), nested)

    def get_test_set(self) -> tuple[Array2D, Array2D]:
        """
        Get the test (x_test, y_test) set from the Dataset.

        Returns
        -------
        x: np.ndarray[tuple[int, int]]
            Parameters values, of shape (n_samples, n_params).
        truth: np.ndarray[tuple[int, int]]
            Truth values of selected values, of shape (n_samples, n_features).

        Notes
        -----
        Assumes that `x_test` and `y_test` DataArrays are present in the dataset.
        """
        arrs: list[Array2D] = [] # x, truth
        for _name in ["x_test", "y_test"]:
            arr = self.get_data(_name, raw=True)
            arr = self._drop_nan_dims(arr) # nan_dims should exist by construction
            arr = self._to_numpy(arr) # (n_samples, n_params/n_features)
            arrs.append(arr)
        x, truth = arrs
        return x, truth

    @overload
    def get_model_error(
        self,
        method: str,
        raw: Literal[True],
        nested: bool = False,
        **kwargs,
    ) -> xr.DataArray:
        ...
    @overload
    def get_model_error(
        self,
        method: str,
        raw: Literal[False],
        nested: Literal[True],
        **kwargs,
    ) -> np.ndarray:
        ...
    @overload
    def get_model_error(
        self,
        method: str,
        raw: Literal[False] = False,
        nested: Literal[False] = False,
        **kwargs,
    ) -> Array2D:
        ...
    def get_model_error(self, method, raw = False, nested = False, **kwargs):
        """
        Get the model error from the registered model, with filters, selection, and output formatting applied.

        Parameters
        ----------
        method : str
            The method to use for the model error calculation. See :meth:`ObservableModel.get_error` for allowed methods.
        raw : bool, optional
            If True, return the raw unfiltered and unflattened error. Default is False.
        nested : bool, optional
            If True, return the error in its original unflattened form. Default is False.
        **kwargs
            Extra arguments to pass to :meth:`ObservableModel.get_error`.

        Returns
        -------
        xr.DataArray or np.ndarray
            The model error, formatted according to the output settings.

        Raises
        ------
        AttributeError
            If no model has been registered.

        See Also
        --------
        get_test_set : For retrieving the test set used in the error calculation.
        """
        if self.model is None:
            raise AttributeError("No model has been registered.")
        x, truth = self.get_test_set()
        error = self.model.get_error(x, truth, method=method, **kwargs)
        error = format_like(
            da = self.get_data("y", raw=True),
            arr = error,
        )
        if raw:
            return error
        return self._format_data(error, str(error.name), nested)

    def get_model_covariance(self, prefactor: float = 1, **kwargs) -> Array2D:
        """
        Get the model covariance matrix matching the filtered dataset.

        Parameters
        ----------
        prefactor : float, optional
            Prefactor to apply to the covariance matrix (e.g. Hartlap or Percival).
            Defaults to 1.0.
        **kwargs
            Extra arguments to pass to :meth:`ObservableModel.make_covariance`.

        Returns
        -------
        np.ndarray[tuple[int, int]]
            The model covariance matrix, matching the filtered dataset.

        Raises
        ------
        AttributeError
            If no model has been registered.

        Notes
        -----
        The covariance matrix is computed from the difference between the true values and the model predictions,
        with filters and selections applied before flattening the result on 2D (sample, features).

        See Also
        --------
        get_test_set : For retrieving the test set used in the covariance calculation.
        """
        if self.model is None:
            raise AttributeError("No model has been registered.")
        x, truth = self.get_test_set()
        pred = self.model.get_prediction(x)
        diff = truth - pred
        diff = format_like(
            da = self.get_data("y", raw=True),
            arr = diff,
        )
        diff = self._apply_filters(diff)
        diff = self._to_numpy(diff)
        diff = self._apply_selection(diff, "y")
        return prefactor * self.model.make_covariance(diff, **kwargs)
