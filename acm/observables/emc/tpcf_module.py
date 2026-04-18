from pathlib import Path

import numpy as np
import xarray
from pycorr import TwoPointCorrelationFunction

from acm.utils.default import cosmo_list  # List of cosmologies in AbacusSummit
from acm.utils.xarray import dataset_to_dict, split_vars

from .base import BaseObservableEMC


class GalaxyCorrelationFunctionMultipoles(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge galaxy correlation
    function multipoles.
    """

    def __init__(self, stat_name="tpcf", **kwargs):
        super().__init__(stat_name=stat_name, **kwargs)

    @classmethod
    def compress_covariance(
        cls,
        paths: dict,
        stat_name: str = "tpcf",
        save_to: str | None = None,
        rebin: int = 4,
        ells: list = [0, 2, 4],
        overwrite_s: np.ndarray | None = None,
    ) -> xarray.DataArray:
        """
        Compress the covariance array from the raw measurement files.

        Parameters
        ----------
        paths : dict
            Dictionary containing the paths to the data directories.
        stat_name : str, optional
            Name of the statistic to compress.
            Defines the name of the subfolder in the measurements directory, and the
            saved filename if save_to is provided.
            Defaults to the class's stat_name.
        save_to : str, optional
            Path of the directory where to save the compressed covariance and bin_values. If None, it is not saved.
            Default is None.
        rebin : int, optional
            Rebinning factor for the statistics. Default is 4.
        ells : list, optional
            List of multipoles to compute the statistics for. Default is [0, 2, 4].
        overwrite_s : np.ndarray, optional
            If not None, overwrite the final separation values with this array.
            This is primarily useful to ensure consistency between the covariance and the data dims.
            Default is None.

        Returns
        -------
        xarray.DataArray
            Covariance array.
        """
        logger = cls.get_logger()

        # Directories
        base_dir = Path(paths["measurements_dir"]) / "small" / stat_name
        data_fns = list(
            base_dir.glob("tpcf_ph*_hod466.npy")
        )  # NOTE: File name format hardcoded !

        y = []
        for data_fn in data_fns:
            data = TwoPointCorrelationFunction.load(data_fn)[::rebin]
            s, multipoles = data(ells=ells, return_sep=True)
            y.append(np.concatenate(multipoles))
        y = np.array(y)
        s = overwrite_s if overwrite_s is not None else s

        y = xarray.DataArray(
            data=y.reshape(y.shape[0], len(ells), -1),
            coords={
                "phase_idx": list(range(y.shape[0])),
                "ells": ells,
                "s": s,
            },
            attrs={
                "sample": ["phase_idx"],
                "features": ["ells", "s"],
            },
            name="covariance_y",
        )

        logger.info(f"Loaded covariance with shape: {y.shape}")

        cout = xarray.Dataset(data_vars={"covariance_y": y})
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f"{stat_name}.npy"
            payload = np.array(dataset_to_dict(cout), dtype=object)
            np.save(save_fn, payload)
            logger.info(f"Saving compressed covariance file to {save_fn}")
        return cout

    @classmethod
    def compress_data(
        cls,
        paths: dict,
        stat_name: str = "tpcf",
        add_covariance: bool = False,
        save_to: str | None = None,
        rebin: int = 4,
        ells: list = [0, 2, 4],
        cosmos: list = cosmo_list,
        n_hod: int = 100,
        phase: int = 0,
        seed: int = 0,
        test_filters: dict | None = None,
    ) -> dict:
        """
        Compress the data from the tpcf raw measurement files.

        Parameters
        ----------
        paths : dict
            Dictionary containing the paths to the data directories.
        stat_name : str, optional
            Name of the statistic to compress.
            Defines the name of the subfolder in the measurements directory, and the
            saved filename if save_to is provided.
            Defaults to the class's stat_name.
        add_covariance : bool, optional
            If True, add the covariance to the compressed data. Default is False.
        save_to : str, optional
            Path of the directory where to save the compressed file. If None, it is not saved.
            Default is None.
        rebin : int, optional
            Rebinning factor for the statistics. Default is 4.
        ells : list, optional
            List of multipoles to compute the statistics for. Default is [0, 2, 4].
        cosmos : list, optional
            List of cosmological parameters to use. If None, use all cosmological parameters.
            Default is None.
        n_hod : int, optional
            Number of HOD parameters to use. Default is 100.
        phase : int, optional
            Phase index to read the data from. Default is 0.
        seed : int, optional
            Seed index to read the data from. Default is 0.
        test_filters : dict, optional
            Dictionary of selection criteria to split data into train and test sets,
            passed to `split_vars`. Each key-value pair specifies a dimension and its
            selection values (e.g., {'cosmo_idx': [0, 1, 2]}). If None, no train/test
            split is performed. Default is None.

        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'x' and 'y' DataArrays.
            If add_covariance is True, also contains 'covariance_y' DataArray.
        """
        logger = cls.get_logger()

        base_dir = paths["measurements_dir"] + f"base/{stat_name}/"

        y = []
        for cosmo_idx in cosmos:
            data_dir = base_dir + f"c{cosmo_idx:03d}_ph{phase:03d}/seed{seed}/"
            for hod_idx in range(n_hod):
                data_fn = f"{data_dir}/tpcf_hod{hod_idx:03}.npy"  # NOTE: File name format hardcoded !
                data = TwoPointCorrelationFunction.load(data_fn)[::rebin]
                s, multipoles = data(ells=ells, return_sep=True)
                y.append(np.concatenate(multipoles))
        y = np.array(y)
        y = xarray.DataArray(
            data=y.reshape(len(cosmos), n_hod, len(ells), -1),
            coords={
                "cosmo_idx": cosmos,
                "hod_idx": list(range(n_hod)),
                "ells": ells,
                "s": s,
            },
            attrs={
                "sample": ["cosmo_idx", "hod_idx"],
                "features": ["ells", "s"],
            },
            name="y",
        )
        x = cls.compress_x(
            paths=paths, cosmos=cosmos, n_hod=n_hod, phase=phase, seed=seed
        )

        logger.info(f"Loaded data with shape: {x.shape}, {y.shape}")

        cout = xarray.Dataset(
            data_vars={
                "x": x,
                "y": y,
            },
        )
        if add_covariance:
            cov_y = cls.compress_covariance(
                paths=paths, stat_name=stat_name, rebin=rebin, ells=ells, overwrite_s=s
            )
            cout = xarray.merge([cout, cov_y], join="outer")

        if test_filters is not None:
            for v_in, v_out in split_vars(cout.x, cout.y, **test_filters):
                v_in.name = v_in.name + "_test"
                v_out.name = v_out.name + "_train"
                v_in.attrs["nan_dims"] = list(
                    test_filters.keys()
                )  # Mark filtered dimensions that will be filled with NaNs
                v_out.attrs["nan_dims"] = list(test_filters.keys())
                cout = xarray.merge([cout, v_in, v_out], join="outer")

        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f"{stat_name}.npy"
            payload = np.array(dataset_to_dict(cout), dtype=object)
            np.save(save_fn, payload)
            logger.info(f"Saving compressed data to {save_fn}")
        return cout

    def compute_phase_correction(self, rebin: int = 4, ells: list = [0, 2, 4]):
        """
        Correction factor to bring the fixed phase precictions (p000) to the ensemble average.

        Parameters
        ----------
        rebin : int
            Rebinning factor for the statistics. Default is 4.
        ells : list
            List of multipoles to compute the correction for. Default is [0, 2, 4].

        Returns
        -------
        np.ndarray
            Correction factor for the fixed phase predictions.
        """

        base_dir = self.paths["measurements_dir"] + f"base/{self.stat_name}/"

        multipoles_mean = []
        for phase in range(25):  # NOTE: Hardcoded !
            data_dir = f"{base_dir}/c000_ph{phase:03}/seed0"  # NOTE: Hardcoded !
            multipoles_hods = []
            for hod in range(50):  # NOTE: Hardcoded !
                data_fn = (
                    Path(data_dir) / f"tpcf_hod{hod:03}.npy"
                )  # NOTE: File name format hardcoded !
                data = TwoPointCorrelationFunction.load(data_fn)[::rebin]
                s, multipoles = data(ells=ells, return_sep=True)
                multipoles_hods.append(multipoles)
            multipoles_hods = np.array(multipoles_hods).mean(axis=0)
            multipoles_mean.append(multipoles_hods)
        multipoles_mean = np.array(multipoles_mean).mean(axis=0)

        data_dir = f"{base_dir}/c000_ph000/seed0"  # NOTE: Hardcoded !
        multipoles_ph0 = []
        for hod in range(50):  # NOTE: Hardcoded !
            data_fn = (
                Path(data_dir) / f"tpcf_hod{hod:03}.npy"
            )  # NOTE: File name format hardcoded !
            data = TwoPointCorrelationFunction.load(data_fn)[::4]
            s, multipoles = data(ells=ells, return_sep=True)
            multipoles_ph0.append(multipoles)
        multipoles_ph0 = np.array(multipoles_ph0).mean(axis=0)
        delta = ((multipoles_mean + 1) - (multipoles_ph0 + 1)) / (multipoles_ph0 + 1)
        return delta.reshape(-1)

    def apply_phase_correction(self, prediction):
        """
        Apply the phase correction to the predictions.
        We apply this to (1 + prediction) to avoid zero-crossings.

        Parameters
        ----------
        prediction : np.ndarray
            Array of predictions.

        Returns
        -------
        np.ndarray
            Corrected predictions.
        """
        return (1 + prediction) * (1 + self.phase_correction) - 1


# Alias
tpcf = GalaxyCorrelationFunctionMultipoles
