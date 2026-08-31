import logging
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Literal, Self, overload

import lsstypes
import numpy as np
from lsstypes import ObservableTree

from acm.utils.logging import suppress_logging

from .base import Array2D, BaseObservable
from .model import ObservableModel

logger = logging.getLogger(__name__)


def _is_valid_tree(tree: ObservableTree) -> bool:
    """Check if the provided tree is a valid ObservableTree."""
    req_vars = {"x", "y"}
    if "name" not in tree.labels("unflatten"):
        logger.debug("Tree is missing 'names' label in unflattened structure.")
        return False
    names = list(tree.labels("unflatten")["name"])
    if not req_vars.issubset(names):
        logger.debug(f"Tree is missing required variables: {req_vars - set(names)}")
        return False
    if tree.get(name="x").labels() != tree.get(name="y").labels():
        logger.debug("Labels for 'x' and 'y' do not match.")
        return False
    return True


def format_like(tree: ObservableTree, arr: np.ndarray, new: str) -> ObservableTree:
    """
    Format a NumPy array to match the structure of an ObservableTree.

    Parameters
    ----------
    tree: ObservableTree
        The reference tree whose structure will be used to build the new tree branches.
    arr: np.ndarray
        The array to format. First dimension will be cast as the new tree branches.
    new: str
        The name for the new variable in the tree.
        Its values will be indexed from the array first dimension.

    Returns
    -------
    ObservableTree
        A new ObservableTree with the same structure as the input tree, but with the new variable.
    """
    branches = [tree.clone(value=arr[i]) for i in range(arr.shape[0])]
    labels = {new: list(range(arr.shape[0]))}
    return ObservableTree(branches, **labels)


def get_filter_indexes(tree: ObservableTree, target: ObservableTree) -> np.ndarray:
    """Get the indexes selected from `tree` to match the shape of `target`."""

    def hook(obs, transform):  # noqa: ANN001, ANN202
        return obs, transform

    _, idx = tree.at.hook(hook)().match(target)  # lsstypes black magic
    return idx


class LsstypesObservable(BaseObservable[ObservableTree]):
    """
    Implementation of BaseObservable using lsstypes.ObservableTree for data storage and manipulation.

    Requires a Tree with at least two sub-trees: "x" for parameters and "y" for truth values.
    The top-level of those tree should be indexed for the samples, and each subtree should handle the features and parameters.
    """

    def __init__(
        self,
        data: ObservableTree,
        model: ObservableModel | None = None,
        silent_load: bool = False,
    ) -> None:
        self._data = data
        self._filters_idx: dict[str, np.ndarray] = {}
        names = list(data.labels("unflatten")["name"])
        with suppress_logging(enabled=silent_load):
            logger.info(f"Tree loaded with the following variables: {names}")
            super().__init__(model=model)

    @classmethod
    def load(cls, filename: str | Path, **kwargs) -> Self:
        """Load an observable instance from a file."""
        data = lsstypes.read(filename)
        if not _is_valid_tree(data):
            raise ValueError(f"Invalid Observable structure in file: {filename}")
        return cls(data, **kwargs)

    @classmethod
    def can_load(cls, filename: str | Path) -> bool:
        """Determine if the class can load the given file."""
        try:
            data = lsstypes.read(filename)
            return _is_valid_tree(data)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to load Observable from {filename}: {e}")
        return False

    def _copy(self, deep: bool = True, **kwargs) -> Self:
        cp = deepcopy if deep else copy
        new = self.__class__(data=cp(self._data, **kwargs), silent_load=True)
        cv = vars(self)
        for k, v in cv.items():
            setattr(new, k, cp(v, **kwargs))
        return new

    @property
    def x_names(self) -> list[str]:
        """Parameter names from the first observable."""
        x = self.get_data("x", raw=True)
        ordered_names = next(iter(x)).labels("unflatten")["parameters"]
        selected_names = self.filters.get("parameters", ordered_names)
        return [n for n in ordered_names if n in selected_names]

    @property
    def filters(self) -> dict:
        """List of filters applied to the data."""
        return self._filters

    @filters.setter
    def filters(self, value: dict) -> None:
        """Set the filters to be applied to the data."""
        self._filters = value
        logger.debug(f"Filters set: {value}")
        # Pecompute matching indices for array filtering
        self._filters_idx.clear()  # Reset previous indexes
        for name in list(self._data.labels("unflatten")["name"]):
            og = next(iter(self._data.get(name)))  # First measurement tree
            target = self._apply_filters(og)
            if og != target:
                idx = get_filter_indexes(og, target)
                logger.debug(f"Registering filter indexes for {name}")
                self._filters_idx[name] = idx

    def _apply_filters(self, data: ObservableTree) -> ObservableTree:
        """Apply any filters to the data."""
        filters = self.filters.copy()
        for k, v in filters.items():
            if isinstance(v, slice):
                filters[k] = (v.start, v.stop)  # Slice by values, not indices
        labels = data.labels("keys", level=None)
        label_filters = {k: v for k, v in filters.items() if k in labels}
        coordinate_filters = {k: v for k, v in filters.items() if k not in labels}
        if label_filters:
            data = data.get(**label_filters)
        if coordinate_filters:
            data = data.select(**coordinate_filters)
        return data

    @overload
    @staticmethod
    def _to_numpy(data: ObservableTree, nested: Literal[False]) -> Array2D: ...
    @overload
    @staticmethod
    def _to_numpy(data: ObservableTree, nested: bool = False) -> np.ndarray: ...
    @staticmethod
    def _to_numpy(data: ObservableTree, nested: bool = False):
        """
        Cast the provided data to a NumPy array.

        Parameters
        ----------
        data: lsstypes.ObservableTree
            The object to cast on a numpy array.
        nested: bool
            If True, returns an unflattened array. Defaults to False (2D array)

        Returns
        -------
        np.ndarray
            The data cast to a 2D NumPy array, unless nested=True.
        """
        return np.array(data.value(concatenate=False, nested=nested))

    def _filter_2d(
        self,
        data: Array2D,
        name: str,
        nested: bool = False,
    ) -> Array2D:
        """
        Apply precomputed filter indexes and selection to the second dimension of the provided 2D data array.

        Parameters
        ----------
        data: np.ndarray[tuple[int, int]]
            The 2D array to format.
        name: str
            The name of the data variable to format. Selection is applied
            only if the indexes for this name have been precomputed.
        nested: bool
            If True, returns a non-selected array, for eventual reshaping.
            Defaults to False (2D array).

        Returns
        -------
        np.ndarray[tuple[int, int]]
            The formatted 2D NumPy array.
        """
        idx = self._filters_idx.get(name)
        if idx is not None:
            logger.debug(f"Applying precomputed filter indexes for {name}")
            data = data[:, idx]  # Faster than making a tree
        if nested is False:
            data = self._apply_selection(name, data)
        return data

    @overload
    def get_data(
        self,
        name: str,
        raw: Literal[True],
        nested: bool = False,
    ) -> ObservableTree: ...
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
    def get_data(self, name: str, raw: bool = False, nested: bool = False):
        """
        Get the data variable from the top level of the tree from the given name.

        Parameters
        ----------
        name : str
            The name of the data variable to retrieve.
        raw : bool, optional
            If True, return the raw data without applying filters or selection. Defaults to False.
        nested : bool, optional
            If True, return the data in its original unflattened form. Defaults to False.

        Returns
        -------
        np.ndarray or lsstypes.ObservableTree
            The requested data variable, either as a 2D NumPy array or as an ObservableTree if raw=True.

        Raises
        ------
        KeyError
            If the specified name is not found in the observable tree.
        """
        if name not in self._data.labels("unflatten")["name"]:
            raise KeyError(f"Name '{name}' not found in the observable tree.")
        data = self._data.get(name=name)
        if raw:
            return data
        return self._format_data(data, name=name, nested=nested)

    def get_coordinate_list(self, name: str) -> list:
        """
        Get the list of unique coordinates for a given name across all observables.

        Notes
        -----
        When requesting a label name, the returned list is unique and may not preserve the order of the original data.
        """
        for branch in self._data:
            fbranch = self._apply_filters(branch)
            labels = fbranch.labels("unflatten", level=None)
            if name in labels:
                return list(set(labels[name]))  # FIXME: preserve order ?
            coords = next(iter(fbranch.flatten(level=None))).coords()
            if name in coords:
                return list(coords[name])
        raise KeyError(f"Name '{name}' not found in any observable.")

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Get an attribute from the tree, with filters applied."""
        data = self._data
        if name in data.labels("unflatten")["name"]:
            return self.get_data(name)
        if not hasattr(data, name):  # Early check before filtering
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        return getattr(self._apply_filters(data), name)

    @overload
    def get_prediction(
        self,
        x: np.ndarray,
        raw: Literal[True],
        nested: bool = False,
    ) -> ObservableTree: ...
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
    def get_prediction(self, x: np.ndarray, raw: bool = False, nested: bool = False):
        """
        Get the prediction for the given input x.

        Parameters
        ----------
        x : np.ndarray
            The input data for which to get the prediction.
        nested : bool, optional
            If True, return the prediction in its original unflattened form. Default is False.

        Returns
        -------
        ObservableTree or np.ndarray
            The predicted output as a 2D NumPy array, unless nested=True,
            or as an ObservableTree if raw=True.
        """
        if self.model is None:
            raise AttributeError("No model has been registered.")
        pred = self.model.get_prediction(np.asarray(x))  # asarray = faster torch.Tensor

        if raw:
            y = next(iter(self.get_data("y", raw=True)))  # First measurement tree
            return format_like(tree=y, arr=pred, new="n_pred")

        pred = self._filter_2d(pred, name="y", nested=nested)
        y = self.get_data("y", nested=nested)
        return pred.reshape(-1, *y.shape[1:])  # Replace first dim by prediction nb

    def get_test_set(self) -> tuple[Array2D, Array2D]:
        """
        Get the test (x_test, y_test) set from the main tree.

        Returns
        -------
        x: np.ndarray[tuple[int, int]]
            Parameters values, of shape (n_samples, n_params).
        truth: np.ndarray[tuple[int, int]]
            Truth values of selected values, of shape (n_samples, n_features).

        Notes
        -----
        Assumes that `x_test` and `y_test` ObservableTrees are present in the main tree.
        """
        arrs: list[Array2D] = []  # x, truth
        for _name in ["x_test", "y_test"]:
            arr = self.get_data(_name, raw=True)
            arr = self._to_numpy(arr)  # (n_samples, n_params/n_features)
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
    ) -> ObservableTree: ...
    @overload
    def get_model_error(
        self,
        method: str,
        raw: Literal[False] = False,
        nested: Literal[False] = False,
        **kwargs,
    ) -> np.ndarray[tuple[int]]:  # 1D array of shape (n_features,)
        ...
    @overload
    def get_model_error(
        self,
        method: str,
        raw: Literal[False] = False,
        nested: bool = False,
        **kwargs,
    ) -> np.ndarray: ...
    def get_model_error(self, method, raw=False, nested=False, **kwargs):
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
        ObservableTree or np.ndarray
            The model error, formatted according to the output settings.
            By default, returns a 1D NumPy array of shape (n_features, ) unless
            nested=True, in which case the shape matches the original unflattened structure.

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
        if raw:
            y = next(iter(self.get_data("y", raw=True)))  # First measurement tree
            return y.clone(value=error)
        error = self._filter_2d(error.reshape(1, -1), name="y", nested=nested)
        y = self.get_data("y", nested=nested)
        return error.reshape(*y.shape[1:])  # No first dim, as error is 1D (n_features,)

    def get_model_covariance(self, prefactor: float = 1, **kwargs) -> Array2D:
        """
        Get the model covariance matrix matching the filtered data.

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
            The model covariance matrix, matching the filtered data.

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
        diff = self._filter_2d(diff, name="y", nested=False)
        return prefactor * self.model.make_covariance(diff, **kwargs)
