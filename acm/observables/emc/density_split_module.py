from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray

from acm.utils.decorators import temporary_class_state
from acm.utils.default import cosmo_list  # List of cosmologies in AbacusSummit
from acm.utils.plotting import set_plot_style
from acm.utils.xarray import dataset_to_dict, split_vars

from .base import BaseObservableEMC


class DensitySplitBaseClass(BaseObservableEMC):
    """
    Base class for density-split correlation observables in the EMC pipeline.

    Subclasses must set the `self.measurement_root` attribute in their `__init__` method.
    This attribute is used by methods in this class to locate measurement files.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def compress_covariance(
        cls,
        stat_name: str,
        paths: dict,
        measurement_root: str,
        save_to: str = None,
        smin: float = 0.0,
        smax: float = 150,
        rebin: int = 4,
        ells: list = [0, 2],
        quantiles: list = [0, 1, 3, 4],
        overwrite_s: np.ndarray = None,
    ):
        """
        Compress the covariance array from the raw measurement files.

        Parameters
        ----------
        paths : dict
            Dictionary containing the paths to the data directories.
        measurement_root : str
            Root name of the measurement files to load.
        stat_name : str, optional
            Name of the statistic to compress.
            Defines the name of the subfolder in the measurements directory, and the
            saved filename if save_to is provided.
            Defaults to the class's stat_name.
        save_to : str, optional
            Path of the directory where to save the compressed covariance and bin_values. If None, it is not saved.
            Default is None.
        smin : float, optional
            Minimum separation value to consider, in Mpc/h. Default is 0.0.
        smax : float, optional
            Maximum separation value to consider, in Mpc/h. Default is 150.
        rebin : int, optional
            Rebinning factor for the statistics. Default is 4.
        ells : list, optional
            List of multipoles to compute the statistics for. Default is [0, 2, 4].
        quantiles : list, optional
            List of quantiles to compute the statistics for. Default is [0, 1, 3, 4].
        statistics : list, optional
            List of statistics to compute the statistics for. Used in the filenames.
            Default is ['quantile_data_correlation', 'quantile_correlation'].
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
        base_dir = Path(paths["measurements_dir"]) / "small" / "density_split"
        data_fns = list(
            base_dir.glob(f"{measurement_root}_poles_ph*.npy")
        )  # NOTE: File name format hardcoded !
        n_sims = len(data_fns)

        y = []
        for data_fn in data_fns:
            data = np.load(data_fn, allow_pickle=True)
            for q in quantiles:
                result = data[q][::rebin].select((smin, smax))
                s, multipoles = result(ells=ells, return_sep=True)
                y.append(np.concatenate(multipoles))
        y = np.array(y)
        y = y.reshape(n_sims, len(quantiles), len(ells), -1)
        s = overwrite_s if overwrite_s is not None else s

        y = xarray.DataArray(
            data=y,
            coords={
                "phase_idx": list(range(y.shape[0])),
                "quantiles": quantiles,
                "ells": ells,
                "s": s,
            },
            attrs={
                "sample": ["phase_idx"],
                "features": ["quantiles", "ells", "s"],
            },
            name="covariance_y",
        )

        logger.info(f"Loaded covariance with shape: {y.shape}")

        cout = xarray.Dataset(data_vars={"covariance_y": y})
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f"{stat_name}.npy"
            np.save(save_fn, dataset_to_dict(cout))
            logger.info(f"Saving compressed covariance file to {save_fn}")
        return cout

    @classmethod
    def compress_data(
        cls,
        paths: dict,
        measurement_root: str,
        stat_name: str,
        add_covariance: bool = False,
        save_to: str = None,
        rebin: int = 4,
        smin: float = 0.0,
        smax: float = 150,
        ells: list = [0, 2],
        quantiles: list = [0, 1, 3, 4],
        cosmos: list = cosmo_list,
        n_hod: int = 100,
        phase: int = 0,
        seed: int = 0,
        test_filters: dict = None,
    ):
        """
        Compress the data from the densitysplit raw measurement files.

        Parameters
        ----------
        paths : dict
            Dictionary containing the paths to the data directories.
        measurement_root : str
            Root name of the measurement files to load.
        stat_name : str
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
            Dictionary of filters to split the dataset into training and test sets.
            Keys are the dimension names and values are the values to filter on for the test set.
            If None, no splitting is done. Default is None.

        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'x' and 'y' DataArrays.
            If add_covariance is True, also contains 'covariance_y' DataArray.
        """
        logger = cls.get_logger()

        base_dir = Path(paths["measurements_dir"]) / "base" / "density_split"

        y = []
        hods = {}
        for cosmo_idx in cosmos:
            logger.info(f"Compressing c{cosmo_idx:03d}")
            handle = f"c{cosmo_idx:03d}_ph{phase:03d}/seed{seed}/{measurement_root}_poles_c{cosmo_idx:03d}_hod*.npy"
            filenames = sorted(base_dir.glob(handle))[:n_hod]
            hods[cosmo_idx] = [int(f.stem.split("hod")[-1]) for f in filenames]
            logger.info(f"Number of HODs: {len(hods[cosmo_idx])}")
            for filename in filenames:
                data = np.load(filename, allow_pickle=True)
                for q in quantiles:
                    result = data[q][::rebin]
                    result.select((smin, smax))
                    s, multipoles = result(ells=ells, return_sep=True)
                    y.append(np.concatenate(multipoles))
        y = np.array(y)
        y = xarray.DataArray(
            data=y.reshape(len(cosmos), n_hod, len(quantiles), len(ells), -1),
            coords={
                "cosmo_idx": cosmos,
                "hod_idx": list(range(n_hod)),
                "quantiles": quantiles,
                "ells": ells,
                "s": s,
            },
            attrs={
                "sample": ["cosmo_idx", "hod_idx"],
                "features": ["quantiles", "ells", "s"],
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
                paths=paths,
                measurement_root=measurement_root,
                stat_name=stat_name,
                rebin=rebin,
                ells=ells,
                quantiles=quantiles,
                overwrite_s=s,
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
            np.save(save_fn, dataset_to_dict(cout))
            logger.info(f"Saving compressed data to {save_fn}")
        return cout

    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_training_set(self, save_fn: str = None):
        ells = self._dataset.y.coords["ells"].values.tolist()
        quantiles = self._dataset.y.coords["quantiles"].values.tolist()

        fig, lax = plt.subplots(len(ells), 1, figsize=(4, 5), sharex=True)

        for ell in ells:
            self.select_filters.update({"ells": ell})
            s = self.s.values

            for i, quantile in enumerate(quantiles):
                self.select_filters.update({"quantiles": quantile})

                for data in self.y:
                    lax[ell // 2].plot(
                        s, s**2 * data, ls="-", color=f"C{i}", lw=0.1, alpha=0.5
                    )

            lax[ell // 2].set_ylabel(r"$s^2\xi_{\ell}(s)\,[h^{-2}{\rm Mpc}^2]$")
        lax[-1].set_xlabel(r"$s\,[h^{-1}{\rm Mpc}]$")

        plt.tight_layout()
        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saving plot to {save_fn}")
        return fig, lax


class DensitySplitQuantileGalaxyCorrelationFunctionMultipoles(DensitySplitBaseClass):
    """
    Class for the Emulator's Mock Challenge density-split cross-correlation function multipoles.
    """

    def __init__(self, stat_name="ds_xiqg", n_test=6 * 200, **kwargs):
        super().__init__(stat_name=stat_name, n_test=n_test, **kwargs)

    @classmethod
    def compress_covariance(cls, **kwargs) -> xarray.DataArray:
        kwargs.setdefault("measurement_root", "dsc_xiqg")
        kwargs.setdefault("stat_name", "ds_xiqg")
        return super().compress_covariance(**kwargs)

    @classmethod
    def compress_data(cls, **kwargs) -> xarray.Dataset:
        kwargs.setdefault("measurement_root", "dsc_xiqg")
        kwargs.setdefault("stat_name", "ds_xiqg")
        return super().compress_data(**kwargs)


class DensitySplitQuantileCorrelationFunctionMultipoles(DensitySplitBaseClass):
    """
    Class for the Emulator's Mock Challenge density-split auto-correlation function multipoles.
    """

    def __init__(self, stat_name="ds_xiqq", **kwargs):
        super().__init__(stat_name=stat_name, **kwargs)

    @classmethod
    def compress_covariance(cls, **kwargs) -> xarray.DataArray:
        kwargs.setdefault("measurement_root", "dsc_xiqq")
        kwargs.setdefault("stat_name", "ds_xiqq")
        return super().compress_covariance(**kwargs)

    @classmethod
    def compress_data(cls, **kwargs) -> xarray.Dataset:
        kwargs.setdefault("measurement_root", "dsc_xiqq")
        kwargs.setdefault("stat_name", "ds_xiqq")
        return super().compress_data(**kwargs)


# Aliases
ds_xiqg = DensitySplitQuantileGalaxyCorrelationFunctionMultipoles
ds_xiqq = DensitySplitQuantileCorrelationFunctionMultipoles
