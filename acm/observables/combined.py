from collections.abc import Iterator

import numpy as np
from scipy import linalg

from .base import BaseObservable


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
        combined_observables = {**self._observables, **other._observables}
        return ObservableList(**combined_observables)

class CombinedObservable(ObservableList[BaseObservable]):
    """Combine multiple observables into a single observable, that returns a combined output."""

    def __init__(self, **observables: BaseObservable) -> None:
        super().__init__(**observables)

    def __repr__(self) -> str:
        """Return a string representation of the combined observable."""
        shapes = {"x": self.x.shape}
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
        return self[0].x_names

    @property
    def x(self) -> np.ndarray:
        """Parameter values from the first observable."""
        return self[0].get_data("x")

    def _to_numpy(self, data: list) -> list[np.ndarray]:
        """Cast a list of observable data to numpy arrays."""
        return [self[0]._to_numpy(d) for d in data]

    def _transfer_call(self, name: str, *args, **kwargs) -> np.ndarray:
        """Call a method on all observables and combine the results."""
        results = [getattr(obs, name)(*args, **kwargs) for obs in self]
        results = self._to_numpy(results)
        return np.concatenate(results, axis=-1)

    def get_data(self, name: str) -> np.ndarray:
        """Get the combined data for a given name from all observables."""
        return self._transfer_call("get_data", name)

    def get_prediction(self, x: np.ndarray) -> np.ndarray:
        """Get the combined prediction from all observables."""
        return self._transfer_call("get_prediction", x)

    def get_model_error(self, method: str, **kwargs) -> np.ndarray:
        """Get the combined model error from all observables."""
        return self._transfer_call("get_model_error", method, **kwargs)

    def get_covariance_matrix(
        self,
        volume_factor: float = 64,
        prefactor: float = 1.0,
        block: bool = True,
    ) -> np.ndarray:
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
        np.ndarray
            The combined covariance matrix, matching the selected filtering.
        """
        factor = prefactor / volume_factor
        cov_y: list[np.ndarray] = []
        for obs in self:
            cy = obs.get_data("covariance_y", raw=True)
            cy = obs._apply_filters(cy)
            cy = obs._to_numpy(cy)
            cy = obs._apply_selection(cy, "y")
            cov_y.append(cy)
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
    ) -> np.ndarray:
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
        np.ndarray
            The combined model covariance matrix.
        """
        covariances = [obs.get_model_covariance(**kwargs) for obs in self]
        if block:
            return linalg.block_diag(*covariances)
        raise NotImplementedError("Non-block combined covariance is not implemented.")
