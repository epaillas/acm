import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray

from acm.observables import Observable
from acm.utils import lookup_registry_path
from acm.utils.decorators import temporary_class_state
from acm.utils.default import cosmo_list  # List of cosmologies in AbacusSummit
from acm.utils.xarray import dataset_to_dict

logger = logging.getLogger(__name__)


class BaseObservableBGS(Observable):
    """
    Base class for the application of the ACM pipeline to the BGS dataset.
    """

    def __init__(
        self, flat_output_dims: int = 2, squeeze_output: bool = True, **kwargs
    ):
        dataset = kwargs.get("dataset", None)
        paths = kwargs.pop("paths", None)
        if dataset is None and paths is None:
            paths = lookup_registry_path("projects.yaml", "bgs", "Mr-20")

        self.n_test = kwargs.pop(
            "n_test", 6 * 100
        )  # FIXME: Remove this on next file compression !
        super().__init__(
            paths=paths,
            flat_output_dims=flat_output_dims,
            squeeze_output=squeeze_output,
            **kwargs,
        )

    def get_emulator_covariance_y(
        self, nofilters: bool = False
    ) -> xarray.DataArray | np.ndarray:
        """
        Returns the unfiltered covariance array of the emulator error of the statistic, with shape (n_test, n_statistics).

        Parameters
        ----------
        nofilters : bool, optional
            If True, no filters are applied to the output and the full DataArray is returned. Defaults to False.

        Returns
        -------
        np.ndarray
            Array of the emulator covariance array, with shape (n_test, n_features).
        """
        # Get unfiltered values
        x_test = getattr(self._dataset, "x_test", None)
        y_test = getattr(self._dataset, "y_test", None)

        if x_test is None or y_test is None:
            # For backward compatibility
            if hasattr(self, "n_test"):
                n_test = self.n_test
                idx_test = range(n_test) if isinstance(n_test, int) else n_test
                x_test = self.flatten_output(self._dataset.x, flat_output_dims=2)[
                    idx_test
                ]
                y_test = self.flatten_output(self._dataset.y, flat_output_dims=2)[
                    idx_test
                ]
                logger.warning(
                    "DEPRECATED: n_test is deprecated. Please provide x_test and y_test in the dataset in the future."
                )
            else:
                raise ValueError(
                    "x_test and y_test are not available in the dataset. Please provide them or set n_test in the class."
                )
        else:
            x_test = self.drop_nan_dimensions(x_test)
            y_test = self.drop_nan_dimensions(y_test)

        # Flatten on 2D for indexing
        # unstack=False because it's either already unstacked or 2D - avoids NaN issues
        x_test = self.flatten_output(x_test, flat_output_dims=2, unstack=False)
        y_test = self.flatten_output(y_test, flat_output_dims=2, unstack=False)

        prediction = self.get_model_prediction(
            x_test, nofilters=True
        )  # Unfiltered prediction !

        # Flatten on 2D for indexing
        prediction = self.flatten_output(prediction, flat_output_dims=2)

        if isinstance(y_test, xarray.DataArray):
            y_test = y_test.values
        if isinstance(prediction, xarray.DataArray):
            prediction = prediction.values

        diff = (
            y_test - prediction
        )  # NOTE: 2D flattening is done to ensure correct broadcasting here !

        n_test = y_test.shape[
            0
        ]  # Indexing on n_test to prevent filtering issues later on
        y = self._dataset.y.unstack()
        shape = (n_test,) + y.shape[len(y.attrs["sample"]) :]
        emulator_covariance_y = xarray.DataArray(
            diff.reshape(shape),
            coords={
                "n_test": range(n_test),
                **{k: y.coords[k] for k in y.dims if k in y.attrs["features"]},
            },
            attrs={
                "sample": ["n_test"],
                "features": y.attrs["features"],
            },
            name="emulator_covariance_y",
        )

        if nofilters:
            return emulator_covariance_y

        emulator_covariance_y = self.apply_filters(emulator_covariance_y)
        emulator_covariance_y = self.flatten_output(
            emulator_covariance_y, self.flat_output_dims
        )
        if "emulator_covariance_y" in self.select_indices_on:
            emulator_covariance_y = self.apply_indices_selection(emulator_covariance_y)
        if self.squeeze_output:
            emulator_covariance_y = emulator_covariance_y.squeeze()
        if self.numpy_output:
            emulator_covariance_y = emulator_covariance_y.values
        return emulator_covariance_y

    def get_emulator_error(self) -> xarray.DataArray | np.ndarray:
        """
        Returns the unfiltered emulator error of the statistic, with shape (n_statistics, ).

        Returns
        -------
        np.ndarray
           Emulator error, with shape (n_features, ).
        """
        emulator_covariance_y = self.get_emulator_covariance_y(
            nofilters=True
        )  # Unfiltered covariance array !

        # Flatten on 2D for indexing
        emulator_covariance_y = self.flatten_output(
            emulator_covariance_y,  # ty:ignore[invalid-argument-type]
            flat_output_dims=2
        )

        emulator_error = np.median(np.abs(emulator_covariance_y), axis=0)

        y = self._dataset.y.unstack()
        shape = y.shape[len(y.attrs["sample"]) :]
        emulator_error = xarray.DataArray(
            emulator_error.reshape(shape),
            coords={**{k: y.coords[k] for k in y.dims if k in y.attrs["features"]}},
            attrs={
                "sample": [],
                "features": y.attrs["features"],
            },
            name="emulator_error",
        )
        emulator_error = self.apply_filters(emulator_error)
        emulator_error = self.flatten_output(emulator_error, self.flat_output_dims)
        if "emulator_error" in self.select_indices_on:
            emulator_error = self.apply_indices_selection(emulator_error)
        if self.squeeze_output:
            emulator_error = emulator_error.squeeze()
        if self.numpy_output:
            emulator_error = emulator_error.values
        return emulator_error

    @classmethod
    def get_hod_from_files(
        cls,
        paths: dict,
        cosmo_idx: int,
        phase: int = 0,
        seed: int = 0,
        density_threshold: float | None = None,
        return_fn: bool = False,
    ) -> np.ndarray:
        """
        Get the existing HOD indexes from the mock folders for a given phase and seed.

        Parameters
        ----------
        paths : dict
            Dictionary containing the paths to the data directories.
        cosmo_idx : int
            Cosmology index to read the HOD indexes from.
        phase : int, optional
            Phase index to read the HOD indexes from. Defaults to 0.
        seed : int, optional
            Seed index to read the HOD indexes from. Defaults to 0.
        density_threshold : float, optional
            Tries to read the `density.npy` file in each HOD folder and only keep HODs with density above this threshold. Defaults to None.
        return_fn : bool, optional
            If True, returns the list of file paths instead of the HOD indexes. Defaults to False.

        Returns
        -------
        np.ndarray
            Array of HOD indexes or file paths.

        Notes
        -----
        The HOD indexes are read from the mock folders for each cosmology. It assumes an architecture like:
        `measurements_dir/base/c{cosmo_idx:03d}_ph{phase:03d}/seed{seed}/hod{hod:03}/`
        and only non-empty HOD directories are considered.
        """
        measurements_dir = paths["measurements_dir"]
        stat_dir = (
            Path(measurements_dir)
            / "base"
            / f"c{cosmo_idx:03d}_ph{phase:03d}"
            / f"seed{seed}"
        )
        hod_idx = [
            int(fn.stem.lstrip("hod"))
            for fn in sorted(stat_dir.glob("hod*"))
            if any(fn.iterdir())
        ]  # Only keep non-empty directories numbers, sorted

        if density_threshold is not None:
            filtered_hod_idx = []
            for hod in hod_idx:
                density_fn = stat_dir / f"hod{hod:03d}" / "density.npy"
                if density_fn.exists():
                    density = np.load(density_fn).item()
                    if density >= density_threshold:
                        filtered_hod_idx.append(hod)
            hod_idx = filtered_hod_idx

        if return_fn:
            fn_list = [stat_dir / f"hod{hod:03d}" for hod in hod_idx]
            return np.array(fn_list)
        return np.array(hod_idx)

    @classmethod
    def compress_x(
        cls, paths: dict, cosmos: list = cosmo_list, n_hod: int | None = None, **kwargs
    ) -> xarray.DataArray:
        """
        Compress the x values from the parameters files.

        Parameters
        ----------
        paths : dict
            Dictionary containing the paths to the data directories.
        cosmos : list, optional
            List of cosmologies to get from the files. Defaults to cosmo_list.
        n_hod : int, optional
            Number of HODs to consider per cosmology.
            If None, it is determined from the first cosmology and restricted to that number for all cosmologies.
            Defaults to None.
        **kwargs : dict
            Additional arguments to pass to `get_hod_from_files`.

        Returns
        -------
        xarray.DataArray
            Compressed x values.

        Raises
        ------
        ValueError
            If the number of HODs for a cosmology is lower than the expected number,
            as the compression requires all cosmologies to have the same number of HODs.

        Notes
        -----
        The parameters are read from the `param_dir/AbacusSummit_c{cosmo_idx:03}.csv` files.
        """
        logger = cls.get_logger()

        param_dir = paths["param_dir"]
        
        # Determine the number of HODs from the first cosmology
        if n_hod is None:
            hod_idx = cls.get_hod_from_files(paths=paths, cosmo_idx=cosmos[0], **kwargs)
            n_hod = len(hod_idx)  
            logger.info(f"Number of HODs determined for c{cosmos[0]:03d}: {n_hod}")

        x = []
        for cosmo_idx in cosmos:
            data_fn = Path(param_dir) / f"AbacusSummit_c{cosmo_idx:03}.csv"
            x_i = pd.read_csv(data_fn)
            x_names = list(x_i.columns)
            x_names = [name.replace(" ", "").replace("#", "") for name in x_names]

            # Get the HOD indexes from folder names (density filtering is optional)
            hod_idx = cls.get_hod_from_files(paths=paths, cosmo_idx=cosmo_idx, **kwargs)

            # Ensure the number of HODs is as expected
            if len(hod_idx) > n_hod:
                hod_idx = hod_idx[:n_hod]  # Restrict to the expected number of HODs
                logger.info(
                    f"Number of HODs for c{cosmo_idx:03d} is larger than expected ({len(hod_idx)} > {n_hod}). Restricting to the first {n_hod} HODs."
                )
            elif len(hod_idx) < n_hod:
                raise ValueError(
                    f"Number of HODs for c{cosmo_idx:03d} is lower than expected ({len(hod_idx)} < {n_hod}). Cannot proceed with compression."
                )

            x.append(x_i.values[hod_idx, :])

        x = np.concatenate(x)
        x = xarray.DataArray(
            x.reshape(len(cosmos), n_hod, -1),
            coords={
                "cosmo_idx": cosmos,
                "hod_idx": list(range(n_hod)),  # re-index HODs to be continuous
                "parameters": x_names,
            },
            attrs={
                "sample": ["cosmo_idx", "hod_idx"],
                "features": ["parameters"],
            },
            name="x",
        )
        return x

    @temporary_class_state(
        flat_output_dims=0,
        numpy_output=False,
        squeeze_output=False,
        select_filters=None,
        slice_filters=None,
        select_indices=None,
    )
    def compress_emulator_error(self, save_to: str | None = None) -> xarray.Dataset:
        """
        From the statistics files for the simulations, the associated parameters, and the covariance array, create the emulator error file.

        Parameters
        ----------
        save_to : str, optional
            Path of the directory where to save the emulator error file. If None, the emulator error file is not saved.
            Default is None.

        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'emulator_error' and 'emulator_covariance_y' DataArrays.
        """
        emulator_cov_y = self.get_emulator_covariance_y()
        emulator_error = self.get_emulator_error()

        emulator_error_dataset = xarray.Dataset(
            data_vars={
                "emulator_covariance_y": emulator_cov_y,
                "emulator_error": emulator_error,
            }
        )

        if save_to:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f"{self.stat_name}.npy"
            payload = np.array(dataset_to_dict(emulator_error_dataset), dtype=object)
            np.save(save_fn, payload)
        return emulator_error_dataset
