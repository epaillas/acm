import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import linalg

from acm.utils.covariance import check_covariance_matrix

from .base import Observable


class CombinedModel:
    """
    Class for the combination of theory models.
    """

    def __init__(self, observables: list[Observable]):
        """
        Parameters
        ----------
        observables : list[Observable]
            List of observables to be combined, initialized with their respective filters.
        """
        self.observables = observables
        self.models = [obs.model for obs in self.observables]

    def get_prediction(self, x):
        """
        Get the prediction from the model.

        Parameters
        ----------
        x : array_like
            Input features.

        Returns
        -------
        array_like
            Model prediction, with respective filters applied to each observable.
        """
        return np.concatenate(
            [obs.get_model_prediction(x) for obs in self.observables], axis=-1
        )


class CombinedObservable:
    """
    Class for the combination of observables.
    It has list properties, that allow to access easily self.observables for readibility.
    """

    def __init__(self, observables: list[Observable]):
        """
        Parameters
        ----------
        observables : list[Observable]
            List of observables to be combined, initialized with their respective filters.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.observables = observables
        self.slice_filters = [obs.slice_filters for obs in self.observables]
        self.select_filters = [obs.select_filters for obs in self.observables]

        is_reshaped = [obs.flat_output_dims == 2 for obs in self.observables]
        if not all(is_reshaped):
            self.logger.warning(
                "Not all observables have flat_output_dims=2. Some outputs might not be properly reshaped, which might cause concatenation issues."
            )

    def __str__(self):
        """
        Returns a string representation of the object (statistic names and slice filters).
        """
        return self.get_save_handle()

    def __getitem__(self, item):
        """
        Allows to access the observables by their statistic name or their index.
        """
        if isinstance(item, int):
            return self.observables[item]
        elif isinstance(item, str):
            try:
                idx = self.stat_name.index(item)
                return self.observables[idx]
            except ValueError:  # If the item is not found in the list, raise an error
                KeyError(f"Observable with name {item} not found.")
        else:
            raise TypeError(f"Item must be an int or str, not {type(item)}.")

    def __setitem__(self, item, value):
        """
        Allows to set the observable at the given index.
        """
        if not isinstance(value, Observable):
            raise TypeError(f"Value must be a Observable, not {type(value)}.")
        if not isinstance(item, int):
            raise TypeError(f"Item must be an int, not {type(item)}.")
        self.observables[item] = value

    def __len__(self):
        """
        Returns the number of observables in the combination.
        """
        return len(self.observables)

    def __iter__(self):
        """
        Returns an iterator over the observables in the combination.
        """
        return iter(self.observables)

    def __contains__(self, item):
        """
        Checks if the observable with the given statistic name is in the combination.
        """
        return item in self.stat_name

    def __reversed__(self):
        """
        Returns a reversed iterator over the observables in the combination.
        """
        return reversed(self.observables)

    def __add__(self, other):
        """
        Allows to add two CombinedObservable objects together or to add a new observable to the existing Observable.
        """
        if isinstance(other, CombinedObservable):
            return CombinedObservable(self.observables + other.observables)
        elif isinstance(other, Observable):
            return CombinedObservable(self.observables + [other])
        else:
            raise TypeError(f"Cannot add {type(other)} to CombinedObservable.")

    def __getattr__(self, name):
        """
        Allows to access the observables by their statistic name as an attribute,
        or the concatenated output of their attributes.
        """
        if name in self.stat_name:
            idx = self.stat_name.index(name)
            return self.observables[idx]
        else:
            try:
                return np.concatenate(
                    [getattr(obs, name) for obs in self.observables], axis=-1
                )
            except AttributeError:
                raise AttributeError(
                    f"'CombinedObservable' object has no attribute '{name}'"
                )

    @property
    def stat_name(self) -> list:
        """
        Name of the statistic.
        """
        return [obs.stat_name for obs in self.observables]

    @property
    def x(self) -> np.ndarray:
        """
        Input features (samples).

        Note: We assume all observable have the same input features, so we just
        return the first from the list.
        """
        return [obs.x for obs in self.observables][0]

    @property
    def x_names(self) -> list:
        """
        Names of the input features.

        Note: We assume all observable have the same input features, so we just
        return the first from the list.
        """
        return [obs.x_names for obs in self.observables][0]

    @property
    def model(self):
        """
        Theory model of the combination of observables.
        `model.get_prediction(x)` returns the prediction of the combination of observables,
        with the respective filters applied to each observable.
        """
        return CombinedModel(self.observables)

    def get_model_prediction(self, x) -> np.ndarray:
        """
        Get the prediction from the model.

        Parameters
        ----------
        x : array_like
            Input features.

        Returns
        -------
        array_like
            Model prediction.
        """
        return np.concatenate(
            [obs.get_model_prediction(x) for obs in self.observables], axis=-1
        )

    def get_covariance_matrix(
        self, volume_factor: float = 64, prefactor: float = 1, **kwargs
    ) -> np.ndarray:
        """
        Covariance matrix for the statistic.
        The prefactor is here for corrections if needed, and the volume factor is the volume correction of the boxes.

        Parameters
        ----------
        volume_factor : float
            Volume correction factor for the boxes. Default is 64.
        prefactor : float
            Prefactor to apply to the covariance matrix (e.g. Hartlap or Percival).
        **kwargs : dict
            Additional arguments for the covariance matrix checker.

        Returns
        -------
        np.ndarray
            The combined data covariance matrix.
        """
        cov_y = self.covariance_y
        prefactor = prefactor / volume_factor

        cov = prefactor * np.cov(
            cov_y, rowvar=False
        )  # rowvar=False : each column is a variable and each row is an observation

        check_covariance_matrix(cov, name="combined data covariance", **kwargs)

        return cov

    def get_emulator_covariance_matrix(
        self, prefactor: float = 1, method: str = "median", diag: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Covariance matrix of the emulator residuals for a combination of multiple summary statistics.
        The matrix is block-diagonal, with each block corresponding to the covariance matrix of each
        individual observable.

        Parameters
        ----------
        prefactor : float
            Prefactor to apply to the covariance matrix (e.g. Hartlap or Percival). Defaults to 1.
        method : str
            Method to compute the covariance matrix from the emulator residuals.
            Options include the mean absolute deviation ('mean'), median absolute deviation ('median'),
            or standard deviation ('stdev'). Defaults to 'median'.
        diag : bool
            If True, only the diagonal of the covariance matrix is computed. Defaults to False.
        **kwargs : dict
            Additional arguments for the covariance matrix checker.

        Returns
        -------
        np.ndarray
            The emulator covariance matrix.
        """
        covs = []
        for observable in self.observables:
            cov_y = observable.get_emulator_covariance_matrix(
                prefactor=prefactor, method=method, diag=diag
            )
            covs.append(cov_y)

        cov = linalg.block_diag(*covs)

        # Perform sanity checks on the covariance matrix
        check_covariance_matrix(cov, name="combined emulator covariance", **kwargs)

        return cov

    def get_save_handle(self, save_dir: str | Path = None) -> str | Path:
        """
        Creates a handle that combines the handles of the observables,
        separated by a '+'. They contain the statistic name and the filters used.
        This can be used to save anything related to this observable.

        Parameters
        ----------
        save_dir : str
            Directory where the results will be saved.
            If provided, the directory is created if it does not exist.
            If None, the handle is returned as a string.
            Default is None.

        Returns
        -------
        str|Path
            The handle for saving the results, to be completed with the file extension.
            Returned as a Path instance if save_dir is provided as a Path.
        """
        statistic_handles = [
            observable.get_save_handle() for observable in self.observables
        ]
        statistic_handle = "+".join(statistic_handles)

        if save_dir is None:
            return statistic_handle

        # If save_path is provided, make sure it exists
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        cout = Path(save_dir) / f"{statistic_handle}"

        if isinstance(save_dir, str):
            return cout.as_posix()  # Return as string if save_dir is a string
        return Path(save_dir) / f"{statistic_handle}"

    def plot_observable(
        self,
        model_params: dict,
        save_fn: str | Path = None,
    ):
        """
        Plot a compilation of all summary statistics included at class instantiation.

        Parameters
        ----------
        model_params : dict
            Dictionary of model parameters to use for the prediction.
        save_fn : str|Path
            File name to save the plot. If None, the plot is not saved.
            Default is None.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The last figure plotted.
        """
        # check that save_fn has a .pdf extension
        if save_fn is not None:
            save_fn = Path(save_fn)
            if save_fn.suffix != ".pdf":
                raise ValueError(
                    f"save_fn must have a .pdf extension, got {save_fn.suffix}"
                )
        with PdfPages(save_fn) if save_fn is not None else nullcontext() as pdf:
            for i, observable in enumerate(self.observables):
                fig, ax = observable.plot_observable(model_params=model_params)
                if pdf is not None:
                    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2)
                else:
                    fig.show()
        self.logger.info(f"Saving {save_fn}")
        return fig
