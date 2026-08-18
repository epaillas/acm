"""File handling model predictions methods for the Observable classes."""
import logging
from pathlib import Path

import numpy as np
import scipy.stats as st
import torch
from sunbird.emulators import BaseModel, load_model_from_checkpoint

from acm.utils.covariance import orthogonal_gk_mad_covariance

logger = logging.getLogger(__name__)

class ObservableModel:
    """Wrapper around `sunbird.emulators` models for Observable classes."""

    def __init__(self, model: BaseModel) -> None:
        self._model = model

    @classmethod
    def load(
        cls,
        filename: str | Path,
        model_cls: type[BaseModel] | None = None,
        **kwargs,
    ) -> "ObservableModel":
        """Initialize the class using a model loaded trough :func:`~sunbird.emulators.load_model_from_checkpoint`."""
        model = load_model_from_checkpoint(filename, model_cls=model_cls)
        return cls(model=model, **kwargs)

    def get_prediction(self, x: np.ndarray) -> np.ndarray:
        """Get the model prediction from a single set of parameters or a 2D set of parameters (n_pred, n_params)."""
        x = np.asarray(x)  # Ensure x is an array to make torch.Tensor faster
        with torch.no_grad():
            pred = self._model.get_prediction(torch.Tensor(x))
            pred = pred.numpy()
        logger.debug(f"Generated prediction of shape {pred.shape}.")
        return pred

    def get_error(
        self,
        x: np.ndarray,
        truth: np.ndarray,
        method: str,
        factor: float = 1,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute the model error from a set of given inputs, and their true values.

        Parameters
        ----------
        x: np.ndarray
            Input of shape (N, n_parameters).
        truth: np.ndarray
            Truth value of the input measurement, of shape (N, n_features).
        method: str
            Method to use to compute the error.
        factor: float, optional
            Prefactor to apply to the covariance matrix. Defaults to 1.
        **kwargs
            Extra arguments to pass to :meth:`make_covariance`

        Returns
        -------
        np.ndarray
            Model error, of shape (n_features,).

        Notes
        -----
        If `method` is not handled by `get_error`, the error is computed as
        the diagonal of the covariance matrix of the `truth - prediction` differences.
        See :meth:`make_covariance`.

        Allowed method values in `get_error`:
            - `median`: Computes the error as the median of the absolute value of the `truth - prediction` differences.
        """
        pred = self.get_prediction(x)
        diff = truth - pred

        logger.info(f"Computing model error using '{method}' method.")
        if method == "median":
            return np.median(np.abs(diff), axis=0)
        return np.diag(factor * self.make_covariance(diff, method, **kwargs))

    @staticmethod
    def make_covariance(y: np.ndarray, method: str, diag: bool) -> np.ndarray:
        """
        Make a covariance matrix from a 2D array.

        Parameters
        ----------
        y: np.ndarray
            Covariance array, of shape (N, n_features)
        method: str
            Method to use to compute the covariance matrix.
        diag: bool
            Whether to compute only a diagonal covariance matrix.

        Returns
        -------
        np.ndarray
            Covariance matrix, of shape (n_features, n_features)

        Raises
        ------
        NotImplementedError
            For method='mean' and diag=True.
        ValueError
            When an unknown method is required.

        Notes
        -----
        Allowed `method` values in `make_covariance`:
            - `mad`: Median absolute deviation. See `~scipy.stats.median_abs_deviation`
        """
        logger.info(f"Computing covariance matrix using '{method}' method ({diag=}).")
        if method == "mad" and diag:
            #  norm to make summary consistent with stdev for a normal distribution
            mad = st.median_abs_deviation(y, axis=0) / st.norm.ppf(3/4)
            return np.diag(mad**2)
        if method == "mad" and not diag:
            return orthogonal_gk_mad_covariance(y)
        if method == "mean" and diag:
            mad = np.mean(np.abs(y - np.mean(y, axis=0)), axis=0) * np.sqrt(np.pi/2)
            return np.diag(mad**2)
        if method == "mean" and not diag:
            raise NotImplementedError(
                f"Mean absolute deviation covariance is not implemented for full matrix ({diag=})."
            )
        if method == "stdev" and diag:
            return np.diag(np.std(y, axis=0)**2)
        if method == "stdev" and not diag:
            return np.cov(y, rowvar=False)
        raise ValueError(f"Unknown method: {method}.")
