import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray
from pycorr import TwoPointCorrelationFunction

from acm.utils.decorators import temporary_class_state
from acm.utils.default import cosmo_list  # List of cosmologies in AbacusSummit
from acm.utils.plotting import set_plot_style
from acm.utils.xarray import dataset_to_dict, split_vars

from .base import BaseObservableEMC

logger = logging.getLogger(__name__)


class ProjectedGalaxyCorrelationFunction(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge galaxy correlation
    function multipoles.
    """

    def __init__(self, stat_name="projected_tpcf", n_test=6 * 500, **kwargs):
        super().__init__(stat_name=stat_name, n_test=n_test, **kwargs)

    @classmethod
    def compress_covariance(
        cls,
        paths: dict,
        stat_name: str = "projected_tpcf",
        save_to: str | None = None,
    ) -> xarray.Dataset:
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

        Returns
        -------
        xarray.DataArray
            Covariance array.
        """
        # Directories
        base_dir = Path(paths["measurements_dir"]) / "small" / stat_name
        data_fns = list(
            base_dir.glob("tpcf_rppi_ph*.npy")
        )  # NOTE: File name format hardcoded !

        y = []
        for data_fn in data_fns:
            data = TwoPointCorrelationFunction.load(data_fn)
            r_p, w_p = data(pimax=None, return_sep=True)
            y.append(w_p)
        y = np.array(y)

        y = xarray.DataArray(
            data=y.reshape(y.shape[0], -1),
            coords={
                "phase_idx": list(range(y.shape[0])),
                "r_p": r_p,
            },
            attrs={
                "sample": ["phase_idx"],
                "features": ["r_p"],
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
        stat_name: str = "projected_tpcf",
        add_covariance: bool = False,
        save_to: str | None = None,
        cosmos: list = cosmo_list,
        n_hod: int = 500,
        phase: int = 0,
        seed: int = 0,
        test_filters: dict | None = None,
    ) -> xarray.Dataset:
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
        base_dir = Path(paths["measurements_dir"], f"base/{stat_name}/")

        y = []
        hods = {}
        for cosmo_idx in cosmos:
            hods[cosmo_idx] = []
            logger.info(f"Compressing c{cosmo_idx:03d}")
            handle = f"c{cosmo_idx:03d}_ph{phase:03d}/seed{seed}/tpcf_rppi_c{cosmo_idx:03d}_hod*.npy"
            filenames = sorted(base_dir.glob(handle))[:n_hod]
            hods[cosmo_idx] = [int(f.stem.split("hod")[-1]) for f in filenames]
            logger.info(f"Number of HODs: {len(hods[cosmo_idx])}")
            for filename in filenames:
                data = TwoPointCorrelationFunction.load(filename)
                r_p, w_p = data(pimax=None, return_sep=True)
                y.append(w_p)

        y = np.array(y)
        y = xarray.DataArray(
            data=y.reshape(len(cosmos), n_hod, -1),
            coords={
                "cosmo_idx": cosmos,
                "hod_idx": list(range(n_hod)),
                "r_p": r_p,
            },
            attrs={
                "sample": ["cosmo_idx", "hod_idx"],
                "features": ["r_p"],
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
            cov_y = cls.compress_covariance(paths=paths, stat_name=stat_name)
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

    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_observable(self, model_params: dict, save_fn: str | None = None):
        """
        Plot the projected galaxy correlation function data, model, and residuals.

        Parameters
        ----------
        model_params : dict
            Dictionary of model parameters to use for the prediction.
        save_fn : str
            Filename to save the plot. If None, the plot is not saved.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, numpy.ndarray
            Figure and axes of the plot.
        """

        height_ratios = [3, 1]
        figsize = (6, 1.5 * sum(height_ratios))
        fig, lax = plt.subplots(
            len(height_ratios),
            sharex=True,
            sharey=False,
            gridspec_kw={"height_ratios": height_ratios},
            figsize=figsize,
            squeeze=True,
        )
        fig.subplots_adjust(hspace=0.1)
        show_legend = False

        lax[-1].set_xlabel(r"$r_p$ [$h^{-1}\,\mathrm{Mpc}$]", fontsize=15)
        lax[0].set_ylabel(r"$r_p w_p(r_p)$ [$h^{-1}\,\mathrm{Mpc}$]", fontsize=15)

        rp = self.r_p.values
        data = self.y
        model = self.get_model_prediction(model_params)

        cov = self.get_covariance_matrix(volume_factor=64)
        error = np.sqrt(np.diag(cov))

        lax[0].errorbar(
            rp,
            rp * data,
            rp * error,
            marker="o",
            ms=4,
            ls="",
            color="C0",
            elinewidth=1.0,
            capsize=None,
        )
        lax[0].plot(rp, rp * model, ls="-", color="C0")
        lax[1].plot(rp, (data - model) / error, ls="-", color="C0")

        for offset in [-2, 2]:
            lax[1].axhline(offset, color="k", ls="--")
        lax[1].set_ylabel(r"$\Delta w_p / \sigma_{w_p}$", fontsize=15)
        lax[1].set_ylim(-4, 4)

        for ax in lax:
            ax.grid(True)
            ax.tick_params(axis="both", labelsize=14)
        if show_legend:
            lax[0].legend(fontsize=15)

        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches="tight")
            logger.info(f"Saving plot to {save_fn}")
        return fig, lax


# Alias
projected_tpcf = ProjectedGalaxyCorrelationFunction
