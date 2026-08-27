from collections.abc import Iterator

import numpy as np
from scipy import linalg

from .base import Array2D, BaseObservable


class ObservableList[S]:
    """Class to handle a list of observables, with ordering and access by name or index."""

    def __init__(self, **observables: S) -> None:
        """Initialize the ObservableList with a dictionary of observables."""
        types = {type(obs) for obs in observables.values()}
        if len(types) > 1:
            raise ValueError(f"All observables must be of the same type, got {types}.")
        self._observables = observables
        self._order = list(observables)

    @property
    def order(self) -> list[str]:
        """Return the order of the observables."""
        return self._order

    @order.setter
    def order(self, new: list[str]) -> None:
        """Set a new order for the observables."""
        if set(new) != set(self._observables):
            raise ValueError("New order must match the registered observable names.")
        if len(new) != len(set(new)):
            raise ValueError("New order must not contain duplicates.")
        self._order = new

    def items(self) -> Iterator[tuple[str, S]]:
        """Return an iterator over the observables in the specified order."""
        return zip(self.order, self, strict=True)

    def __getitem__(self, key: str | int) -> S:
        """Get an observable by name or index."""
        if isinstance(key, int):
            key = self.order[key]
        return self._observables[key]

    def __len__(self) -> int:
        """Return the number of observables."""
        return len(self._observables)

    def __iter__(self) -> Iterator[S]:
        """Iterate over the observables in the order specified."""
        for name in self.order:
            yield self._observables[name]

    def __contains__(self, key: str | int) -> bool:
        """Check if an observable is in the combined observable by name or index."""
        if isinstance(key, int):
            return 0 <= key < len(self._observables)
        return key in self._observables

    def __reversed__(self) -> Iterator[S]:
        """Iterate over the observables in reverse order."""
        for name in reversed(self.order):
            yield self._observables[name]

    def __add__(self, other: "ObservableList[S]") -> "ObservableList[S]":
        """Combine two ObservableLists into a new ObservableList."""
        ol = set(self.order) & set(other.order)
        if ol:
            raise ValueError(f"Cannot add ObservableLists with overlapping names: {ol}")
        combined_observables = {**self._observables, **other._observables}
        return ObservableList(**combined_observables)

class CombinedObservable(ObservableList[BaseObservable]):
    """Combine multiple observables into a single observable, that returns a combined output."""

    def __repr__(self) -> str:
        """Return a string representation of the combined observable."""
        shapes = {}
        try:
            shapes["x"] = self.x.shape
        except KeyError: # Let ValueError raise if x's are not consistent
            pass
        for name in ("y", "covariance_y"):
            try:
                shapes[name] = self.get_data(name).shape
            except KeyError:
                continue
        shape_str = ", ".join(f"{k}={v}" for k, v in shapes.items())
        names = ", ".join(self.order)
        return f"{type(self).__name__}([{names}], {shape_str})"

    def get_handle(self, hlength: int | None = None) -> str:
        """Get a unique handle for the combined observable based on its components."""
        handles = [obs.get_handle(key, hlength) for key, obs in self.items()]
        return "+".join(handles)

    @property
    def x_names(self) -> list[str]:
        """Parameter names from the first observable."""
        n = [obs.x_names for obs in self]
        if any(not np.array_equal(n[0], _n) for _n in n):
            raise ValueError("All observables must have the same x_names.")
        return n[0]

    @property
    def x(self) -> Array2D:
        """Parameter values from the first observable."""
        x = [obs.get_data("x") for obs in self]
        if any(not np.array_equal(x[0], _x) for _x in x):
            raise ValueError("All observables must have the same x values.")
        return x[0]

    def _transfer_call(self, name: str, *args, **kwargs) -> np.ndarray:
        """Call a method on all observables and combine the results."""
        results = [getattr(obs, name)(*args, **kwargs) for obs in self]
        return np.concatenate(results, axis=-1)

    def get_data(self, name: str) -> Array2D:
        """Get the combined data for a given name from all observables."""
        return self._transfer_call("get_data", name)

    def get_prediction(self, x: Array2D) -> Array2D:
        """Get the combined prediction from all observables."""
        return self._transfer_call("get_prediction", x)

    def get_model_error(self, method: str, **kwargs) -> Array2D:
        """Get the combined model error from all observables."""
        call = "get_model_error"
        return self._transfer_call(call, method, raw=False, nested=False, **kwargs)

    def get_covariance_matrix(
        self,
        volume_factor: float = 64,
        prefactor: float = 1.0,
        block: bool = True,
    ) -> Array2D:
        """
        Get the combined covariance matrix from all observables.

        Parameters
        ----------
        volume_factor : float, optional
            The volume factor to scale the covariance matrix. Defaults to 64.
        prefactor : float
            Prefactor to apply to the covariance matrix (e.g. Hartlap or Percival).
            Defaults to 1.0.
        block : bool, optional
            If True, compute the covariance matrix in block diagonal form. Defaults to True.

        Returns
        -------
        np.ndarray[tuple[int, int]]
            The combined covariance matrix, matching the selected filtering.
        """
        factor = prefactor / volume_factor
        cov_y: list[Array2D] = [obs.get_data("covariance_y") for obs in self]
        if block:
            blocks = [factor * np.cov(cy, rowvar=False) for cy in cov_y]
            cov = linalg.block_diag(*blocks)
        else:
            cov_array = np.concatenate(cov_y, axis=-1)
            cov = factor * np.cov(cov_array, rowvar=False)
        return cov

    def get_model_covariance(
        self,
        block: bool = True,
        **kwargs,
    ) -> Array2D:
        """
        Get the combined model covariance from all observables.

        Parameters
        ----------
        block : bool, optional
            If True, compute the covariance matrix in block diagonal form. Defaults to True.
        **kwargs : dict
            Additional keyword arguments to pass to each observable's get_model_covariance method.

        Returns
        -------
        np.ndarray[tuple[int, int]]
            The combined model covariance matrix.
        """
        covariances = [obs.get_model_covariance(**kwargs) for obs in self]
        if block:
            return linalg.block_diag(*covariances)
        raise NotImplementedError("Non-block combined covariance is not implemented.")
