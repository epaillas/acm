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
    required_vars = {"x", "y"}
    if "name" not in tree.labels("unflatten"):
        logger.debug("Tree is missing 'names' label in unflattened structure.")
        return False
    names = list(tree.labels("unflatten")["name"])
    if not required_vars.issubset(names):
        logger.debug(f"Tree is missing required variables: {required_vars - set(names)}")
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
        The reference tree whose structure will be used.
    arr: np.ndarray
        The array to format. Must match the shape of the tree's values.
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
        self._tree = data
        self._filters_idx: list[int] | None = None
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
        new = self.__class__(data = cp(self._tree, **kwargs), silent_load = True)
        cv = vars(self)
        for k, v in cv.items():
            setattr(new, k, cp(v, **kwargs))
        return new

    @property
    def x_names(self) -> list[str]:
        """Parameter names from the first observable."""
        x = self.get_data("x", raw=True)
        return next(iter(x)).labels("unflatten")["parameters"]

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
        og = next(iter(self._tree.get("y"))) # First measurement tree
        target = self._apply_filters(og)
        def hook(obs, transform): return obs, transform  # noqa: ANN001, ANN202
        _, self._filters_idx = og.at.hook(hook)().match(target) # lsstypes black magic

    def _apply_filters(self, data: ObservableTree) -> ObservableTree:
        """Apply any filters to the data."""
        filters = self.filters.copy()
        for k, v in filters.items():
            if isinstance(v, slice):
                filters[k] = (v.start, v.stop) # Slice by values, not indices
        labels = data.labels("keys", level=None)
        label_filters = {k: v for k, v in filters.items() if k in labels}
        coordinate_filters = {k: v for k, v in filters.items() if k not in labels}
        return data.get(**label_filters).select(**coordinate_filters)

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

    def _format_2d_data(
        self,
        data: Array2D,
        name: str,
        nested: bool = False,
    ) -> Array2D:
        """
        Format a 2D NumPy array to match the structure of the observable tree.

        Parameters
        ----------
        data: np.ndarray[tuple[int, int]]
            The 2D array to format.
        name: str
            The name of the variable in the tree.
        nested: bool
            If True, returns an unflattened array. Defaults to False (2D array)

        Returns
        -------
        np.ndarray[tuple[int, int]]
            The formatted 2D NumPy array.
        """
        if self._filters_idx is not None:
            data = data[:, self._filters_idx] # Faster than making a tree
        if nested is False:
            data = self._apply_selection(data, name)
        return data

    @overload
    def get_data(
        self,
        name: str,
        raw: Literal[True],
        nested: bool = False,
    ) -> ObservableTree:
        ...
    @overload
    def get_data(
        self,
        name: str,
        raw: Literal[False] = False,
        nested: Literal[False] = False,
    ) -> Array2D:
        ...
    @overload
    def get_data(self,
        name: str,
        raw: Literal[False] = False,
        nested: bool = False,
    ) -> np.ndarray:
        ...
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
        if name not in self._tree.labels("unflatten")["name"]:
            raise KeyError(f"Name '{name}' not found in the observable tree.")
        data = self._tree.get(name=name)
        if raw:
            return data
        return self._format_data(data, name=name, nested=nested)

    def get_coordinate_list(self, name: str) -> list:
        """Get the list of unique coordinates for a given name across all observables."""
        ftree = self._apply_filters(self._tree)
        for branch in ftree:
            labels = branch.labels("unflatten", level=None)
            if name in labels:
                return list(set(labels[name])) # FIXME: preserve order ?
            coords = next(iter(branch.flatten(level=None))).coords()
            if name in coords:
                return list(coords[name])
        raise KeyError(f"Name '{name}' not found in any observable.")

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Get an attribute from the tree, with filters applied."""
        data = self._tree
        if name in data.labels("unflatten")["name"]:
            return self.get_data(name)
        if not hasattr(data, name): # Early check before filtering
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self._apply_filters(data), name)

    @overload
    def get_prediction(
        self,
        x: np.ndarray,
        raw: Literal[True],
        nested: bool = False,
    ) -> ObservableTree:
        ...
    @overload
    def get_prediction(self,
        x: np.ndarray,
        raw: Literal[False] = False,
        nested: Literal[False] = False,
    ) -> Array2D:
        ...
    @overload
    def get_prediction(self,
        x: np.ndarray,
        raw: Literal[False] = False,
        nested: bool = False,
    ) -> np.ndarray:
        ...
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
        np.ndarray
            The predicted output as a 2D NumPy array, unless nested=True.
        """
        if self.model is None:
            raise AttributeError("No model has been registered.")
        pred = self.model.get_prediction(np.asarray(x)) # asarray = faster torch.Tensor

        if raw:
            y = self.get_data("y", raw=True)
            return format_like(tree=y, arr=pred, new="n_pred")

        pred = self._format_2d_data(pred, name="y", nested=nested)
        y = self.get_data("y", nested=nested)
        return pred.reshape(-1, *y.shape[1:]) # Replace first dim by prediction nb

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
        arrs: list[Array2D] = [] # x, truth
        for _name in ["x_test", "y_test"]:
            arr = self.get_data(_name, raw=True)
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
    ) -> ObservableTree:
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
    @overload
    def get_model_error(
        self,
        method: str,
        raw: Literal[False] = False,
        nested: bool = False,
        **kwargs,
    ) -> np.ndarray:
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
        ObservableTree or np.ndarray
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
        if raw:
            y = self.get_data("y", raw=True)
            return format_like(tree=y, arr=error, new="n_error")
        error = self._format_2d_data(error, name="y", nested=nested)
        y = self.get_data("y", nested=nested)
        return error.reshape(-1, *y.shape[1:]) # Replace first dim by prediction nb

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
        diff = self._format_2d_data(diff, name="y", nested=False)
        return prefactor * self.model.make_covariance(diff, **kwargs)
