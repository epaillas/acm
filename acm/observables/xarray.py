"""Concrete implementation for the BaseObservable product for the xarray interface."""
import logging
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import xarray as xr

from acm.utils.logging import suppress_logging
from acm.utils.xarray import dataset_from_dict

from .base import BaseObservable
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


class XarrayObservable(BaseObservable[xr.DataArray, _ArrayLike]):
    """TODO: docstring + file requirements."""

    def __init__(
        self,
        dataset: xr.Dataset,
        model: ObservableModel | None = None,
        silent_load: bool = False,
        **kwargs,
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
        self._dataset = dataset
        with suppress_logging(enabled=silent_load):
            logger.info(
                f"Datasets loaded with the following variables: {list(self._dataset.data_vars)}"
            )
            super().__init__(model=model, **kwargs)

    @classmethod
    def load(cls, filename: str | Path, **kwargs) -> "XarrayObservable":
        """Load a compressed XarrayObservable file and model."""
        dataset = _load_dataset(filename)
        if not _is_valid_dataset(dataset):
            raise ValueError(f"Invalid dataset in file: {filename}")
        return cls(dataset, **kwargs)

    @classmethod
    def can_load(cls, filename: str | Path) -> bool:
        """
        Check if the given filename can be loaded by this class.

        Parameters
        ----------
        filename : str or Path
            The filename to check.

        Returns
        -------
        bool
            True if the file can be loaded, False otherwise.
        """
        try:
            dataset = _load_dataset(filename)
            return _is_valid_dataset(dataset)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to load dataset from {filename}: {e}")
        return False

    def _copy(self, deep: bool = True, **kwargs) -> "XarrayObservable":
        method = deepcopy if deep else copy
        new = self.__class__(
            dataset = method(self._dataset, **kwargs),
            silent_load = True, # Do not log during copy
        )
        cv = vars(self)
        for k, v in cv.items():
            setattr(new, k, method(v, **kwargs))
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

    def _apply_selection(self, data: xr.DataArray) -> xr.DataArray:
        """
        Select specific indices from the last dimension of the DataArray.

        Applicable only if the DataArray is 1D or 2D, and on the DataArrays
        specified in the selection names (see :meth:`set_select`).

        Also accepts DataArrays with names prefixed by "like_" and accepted names
        (e.g., "like_y" if "y" is in the selection names).

        Parameters
        ----------
        data : xr.DataArray
            The DataArray to which the selection will be applied.

        Returns
        -------
        xr.DataArray
            The selected DataArray.

        Raises
        ------
        ValueError
            If the selection indices exceed the size of the last dimension of the DataArray.
        """
        like_names = ["like_" + name for name in self._select_names] # e.g. "like_y"
        ok_names = set(self._select_names + like_names)
        if self._select is not None and data.ndims < 3 and data.name in ok_names:
            if data.shape[-1] < max(self._select):
                raise ValueError(
                    f"Selection indices exceed the last dimension size {data.shape[-1]}."
                )
            return data.isel({data.dims[-1]: self._select})
        logger.debug(f"No selection applied to DataArray '{data.name}'.")
        return data

    @staticmethod
    def _flatten(data: xr.DataArray, ndim: int) -> xr.DataArray:
        """
        Flatten a DataArray on 1 or 2 dimensions.

        Parameters
        ----------
        data: xr.DataArray
            The DataArray to flatten. Must contain a "sample" and "feature" attribute for 2D flattening.
        ndim: int
            The number of dimensions to flatten on.

        Returns
        -------
        xr.DataArray
            The flattened array, with a "dims" or ("sample", "features") stacked
            MultiIndex dimensions for respectively 1D and 2D flattening.
        """
        if ndim == 1:
            return data.stack(dims=[...])
        if ndim == 2:
            # NOTE: requires "sample" and "features" to exist in attrs
            data = _stack_on("sample", data, *data.attrs["sample"])
            data = _stack_on("features", data, *data.attrs["features"])
            return data.transpose("sample", "features") # Ensure correct dim order
        raise ValueError("Only 1D and 2D flattening are allowed.")

    @staticmethod
    def _squeeze(data: xr.DataArray) -> xr.DataArray:
        """Wrap around :func:`~xarray.DataArray.squeeze`."""
        return data.squeeze()

    @staticmethod
    def _to_numpy(data: xr.DataArray) -> np.ndarray:
        """Cast to numpy array."""
        return data.to_numpy()

    def _format_data(self, data: xr.DataArray) -> _ArrayLike:
        """
        Format a DataArray according to the set filters, selection, and output format.

        Parameters
        ----------
        da : xr.DataArray
            The DataArray to format.

        Returns
        -------
        xr.DataArray or np.ndarray
            The formatted DataArray, after applying filters, selection, and output formatting.
        """
        out = self._drop_nan_dims(data)
        out = super()._format_data(out)  # Apply filters, selection, flatten, squeeze
        # TODO: postprocess option (e.g. phase correction) provided by the user
        return out

    @overload
    def get_data(self, name: str, raw: Literal[True]) -> xr.DataArray: ...
    @overload
    def get_data(self, name: str, raw: Literal[False] = False) -> _ArrayLike: ...

    def get_data(self, name: str, raw: bool = False) -> _ArrayLike:
        """
        Get the data variable from the dataset, with filters, selection, and output formatting applied.

        Parameters
        ----------
        name : str
            The name of the data variable to retrieve.
        raw : bool, optional
            If True, return the raw DataArray without applying filters, selection, or output formatting.
            Defaults to False.

        Returns
        -------
        xr.DataArray or np.ndarray
            The requested data variable, formatted according to the output settings.

        Raises
        ------
        ValueError
            If the specified data variable name is not found in the dataset.
        """
        if name not in self._dataset.data_vars:
            raise ValueError(f"Data variable '{name}' not found in the dataset.")
        da = self._dataset[name]
        if raw:
            return da
        return self._format_data(da)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Get an attribute from the dataset, with filters applied."""
        ds = self._dataset
        if name in ds.data_vars:
            return self.get_data(name)
        return getattr(self._apply_filters(ds), name)

    @overload
    def get_prediction(self, x: _ArrayLike, raw: Literal[True]) -> xr.DataArray: ...
    @overload
    def get_prediction(self, x: _ArrayLike, raw: Literal[False] = False) -> _ArrayLike:
        ...

    def get_prediction(self, x: _ArrayLike, raw: bool = False) -> _ArrayLike:
        """
        Get the prediction from the registered model for the given input x, with filters, selection, and output formatting applied.

        Parameters
        ----------
        x : xr.DataArray or np.ndarray
            The input data for which the model prediction is to be computed.
        raw : bool, optional
            If True, return the raw model prediction without applying filters, selection, or output formatting.
            Defaults to False.

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
        pred = format_like(
            da = self.get_data("y", raw=True),
            arr = pred,
            new = "n_pred",
        )
        if not raw:
            return pred
        return self._format_data(pred)

    def get_test_set(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the test set from the Dataset.

        Returns
        -------
        x: np.ndarray
            Parameters values, of shape (n_samples, n_params).
        truth: np.ndarray
            Truth values of selected values, of shape (n_samples, n_features).
        """
        arrs: list[np.ndarray] = [] # x, truth
        for _name in ["x_test", "y_test"]:
            arr = self.get_data(_name, raw=True)
            arr = self._drop_nan_dims(arr) # nan_dims should exist by construction
            arr = self._flatten(arr, ndim=2)
            arr = self._to_numpy(arr) # (n_samples, n_params/n_features)
            arrs.append(arr)
        x, truth = arrs
        return x, truth

    @overload
    def get_model_error(self, method: str, raw: Literal[True], **kwargs) -> xr.DataArray: ...
    @overload
    def get_model_error(self, method: str, raw: Literal[False] = False, **kwargs) -> _ArrayLike: ...

    def get_model_error(
        self,
        method: str,
        raw: bool = False,
        **kwargs,
    ) -> _ArrayLike:
        """
        Get the model error from the registered model, with filters, selection, and output formatting applied.

        Parameters
        ----------
        method : str
            The method to use for computing the model error. See :meth:`ObservableModel.get_error` for allowed methods.
        raw : bool, optional
            If True, return the raw model error without applying filters, selection, or output formatting.
            Defaults to False.
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

        Notes
        -----
        Assumes that `x_test` and `y_test` DataArrays are present in the dataset for computing the model error.
        """
        if self.model is None:
            raise AttributeError("No model has been registered.")
        x, truth = self.get_test_set()
        error = self.model.get_error(x, truth, method=method, **kwargs)
        error = format_like(
            da = self.get_data("y", raw=True),
            arr = error,
        )
        if not raw:
            return error
        return self._format_data(error)

    def get_model_covariance(self, prefactor: float = 1, **kwargs) -> np.ndarray:
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
        np.ndarray
            The model covariance matrix, matching the filtered dataset.

        Raises
        ------
        AttributeError
            If no model has been registered.

        Notes
        -----
        The covariance matrix is computed from the difference between the true values and the model predictions,
        with filters and selections applied before flattening the result on 2D (sample, features).
        Assumes that `x_test` and `y_test` DataArrays are present in the dataset for computing the model covariance.
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
        diff = self._flatten(diff, ndim=2)
        diff = self._apply_selection(diff)
        diff = self._to_numpy(diff)
        return prefactor * self.model.make_covariance(diff, **kwargs)
