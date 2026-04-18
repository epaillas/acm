import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray

from acm.utils.decorators import temporary_class_state
from acm.utils.default import cosmo_list  # List of cosmologies in AbacusSummit
from acm.utils.plotting import set_plot_style
from acm.utils.xarray import dataset_to_dict, split_vars

from .base import BaseObservableEMC

logger = logging.getLogger(__name__)


class WaveletScatteringTransform(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge galaxy correlation
    function multipoles.
    """

    def __init__(self, stat_name="wst", n_test=6 * 250, **kwargs):
        super().__init__(stat_name=stat_name, n_test=n_test, **kwargs)

    @staticmethod
    def renorm_wst(inpt, config="J5_L3_q0.8_sigma0.4"):
        if config == "J5_L3_q0.8_sigma0.4":
            s0 = inpt[0]
            s12 = inpt[1:].reshape(21, 4)
            outp = np.zeros_like(s12)
            outp[:6, :] = s12[:6, :] / s0
            outp[6:11, :] = s12[6:11, :] / s12[0, :]
            outp[11:15, :] = s12[11:15, :] / s12[1, :]
            outp[15:18, :] = s12[15:18, :] / s12[2, :]
            outp[18:20, :] = s12[18:20, :] / s12[3, :]
            outp[20:, :] = s12[20:, :] / s12[4, :]
            sfin = np.hstack((1.0, outp.flatten()))
            return sfin
        else:
            s0 = inpt[0]
            s12 = inpt[1:].reshape(15, 5)
            outp = np.zeros_like(s12)
            outp[:5, :] = s12[:5, :] / s0
            outp[5:9, :] = s12[5:9, :] / s12[0, :]
            outp[9:12, :] = s12[9:12, :] / s12[1, :]
            outp[12:14, :] = s12[12:14, :] / s12[2, :]
            outp[14:, :] = s12[14:, :] / s12[3, :]
            sfin = np.hstack((1.0, outp.flatten()))
        return sfin

    @classmethod
    def compress_covariance(
        cls,
        paths: dict,
        stat_name: str = "wst",
        save_to: str | None = None,
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

        Returns
        -------
        xarray.DataArray
            Covariance array.
        """
        logger = cls.get_logger()

        # Directories
        base_dir = Path(paths["measurements_dir"]) / "small" / stat_name

        # Define WST configurations to concatenate
        configs = [
            "J4_L4_q1_sigma0.8",
            "J4_L4_q1_sigma1.0",
            "J5_L3_q0.8_sigma0.4",
        ]

        # WST coefficient indices to mask due to instabilities
        mask = [
            95,
            96,
            97,
            98,
            99,
            116,
            117,
            118,
            119,
            131,
            132,
            133,
            134,
            141,
            142,
            143,
            144,
            146,
            147,
            148,
            149,
        ]

        # Get phase files from first configuration
        first_config_dir = base_dir / configs[0]
        data_fns = list(first_config_dir.glob("wst_ph*.npy"))

        y = []
        for data_fn in data_fns:
            phase_filename = data_fn.name
            concatenated_coeffs = []
            for config_folder in configs:
                config_dir = base_dir / config_folder
                filename = config_dir / phase_filename
                data = np.load(filename, allow_pickle=True)
                normalized = cls.renorm_wst(data, config=config_folder)[
                    1:
                ]  # Exclude first element
                concatenated_coeffs.append(normalized)
            # Concatenate coefficients from all three configurations
            concatenated_coeffs = np.concatenate(concatenated_coeffs)
            # concatenated_coeffs = np.delete(concatenated_coeffs, mask)  # Apply mask to remove unstable coefficients
            y.append(concatenated_coeffs)
        y = np.array(y)

        y = xarray.DataArray(
            data=y.reshape(y.shape[0], -1),
            coords={
                "phase_idx": list(range(y.shape[0])),
                "bin_idx": np.arange(y.shape[1]),
            },
            attrs={
                "sample": ["phase_idx"],
                "features": ["bin_idx"],
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
        stat_name: str = "wst",
        add_covariance: bool = False,
        save_to: str | None = None,
        cosmos: list = cosmo_list,
        n_hod: int = 500,
        phase: int = 0,
        seed: int = 0,
        test_filters: dict | None = None,
    ) -> dict:
        """
        Compress the data from raw measurement files.

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
        logger = cls.get_logger()

        base_dir = Path(paths["measurements_dir"], f"base/{stat_name}/")

        # Define WST configurations to concatenate
        configs = [
            "J4_L4_q1_sigma0.8",
            "J4_L4_q1_sigma1.0",
            "J5_L3_q0.8_sigma0.4",
        ]

        # WST coefficient indices to mask due to instabilities
        mask = [
            95,
            96,
            97,
            98,
            99,
            116,
            117,
            118,
            119,
            131,
            132,
            133,
            134,
            141,
            142,
            143,
            144,
            146,
            147,
            148,
            149,
        ]

        y = []
        hods = {}
        for cosmo_idx in cosmos:
            logger.info(f"Compressing c{cosmo_idx:03}")

            # Get HOD indices from first configuration
            first_config_dir = (
                base_dir / configs[0] / f"c{cosmo_idx:03}_ph{phase:03}" / f"seed{seed}"
            )
            filenames = sorted(
                first_config_dir.glob(f"wst_c{cosmo_idx:03}_hod???.npy")
            )[:n_hod]
            hod_indices = [int(f.stem.split("hod")[-1]) for f in filenames]
            hods[cosmo_idx] = hod_indices
            logger.info(f"Number of HODs: {len(hod_indices)}")

            # Load and concatenate data from all configurations for each HOD
            for hod_idx in hod_indices:
                concatenated_coeffs = []
                for config_folder in configs:
                    config_dir = (
                        base_dir
                        / config_folder
                        / f"c{cosmo_idx:03}_ph{phase:03}"
                        / f"seed{seed}"
                    )
                    filename = config_dir / f"wst_c{cosmo_idx:03}_hod{hod_idx:03}.npy"
                    data = np.load(filename, allow_pickle=True)
                    normalized = cls.renorm_wst(data, config=config_folder)[
                        1:
                    ]  # Exclude first element
                    concatenated_coeffs.append(normalized)
                # Concatenate coefficients from all three configurations
                concatenated_coeffs = np.concatenate(concatenated_coeffs)
                # concatenated_coeffs = np.delete(concatenated_coeffs, mask)  # Apply mask to remove unstable coefficients
                y.append(concatenated_coeffs)
        y = np.array(y)
        y = xarray.DataArray(
            data=y.reshape(len(cosmos), n_hod, -1),
            coords={
                "cosmo_idx": cosmos,
                "hod_idx": list(range(n_hod)),
                "bin_idx": np.arange(y.shape[-1]),
            },
            attrs={
                "sample": ["cosmo_idx", "hod_idx"],
                "features": ["bin_idx"],
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
    def plot_training_set(self, save_fn: str | None = None):
        """
        Plot the training set for the observable.

        Parameters
        ----------
        save_fn : str, optional
            Path to save the figure. If None, the figure is not saved.
            Default is None.
        """

        fig, ax = plt.subplots(figsize=(5, 4))

        for data in self.y:
            ax.plot(data, color="gray", alpha=0.5, lw=0.1)

        ax.set_xlabel("bin index")
        ax.set_ylabel("WST coefficient")

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches="tight")
            logger.info(f"Saving training set figure to {save_fn}")

        return fig, ax

    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_observable(self, model_params: dict, save_fn: str | None = None):
        """
        Plot multi-scale Minkowski functionals predictions against data.

        Parameters
        ----------
        model_params : dict
            Dictionary of model parameters to use for the prediction.
        save_fn : str, optional
            Filename to save the plot. If None, the plot is not saved.

        Returns
        -------
        fig, lax : matplotlib.figure.Figure, np.ndarray
            Figure and axes array of the plot.
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

        lax[-1].set_xlabel(r"$\textrm{bin index}$]", fontsize=15)
        lax[0].set_ylabel(r"$\textrm{WST coefficient}$", fontsize=15)

        bin_idx = self.bin_idx.values
        data = self.y
        model = self.get_model_prediction(model_params)

        cov = self.get_covariance_matrix(volume_factor=64)
        error = np.sqrt(np.diag(cov))

        lax[0].errorbar(
            bin_idx,
            data,
            error,
            marker="o",
            ms=3,
            ls="",
            color="C0",
            elinewidth=1.0,
            capsize=None,
        )
        lax[0].plot(bin_idx, model, ls="-", color="C1")
        lax[1].plot(bin_idx, (data - model) / error, ls="-", color="C0")

        for offset in [-2, 2]:
            lax[1].axhline(offset, color="k", ls="--")
        lax[1].set_ylabel(r"$\Delta \textrm{WST} / \sigma_\textrm{WST}$", fontsize=15)
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

    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_covariance_set(self, save_fn: str | None = None):
        """
        Plot the covariance matrix for the observable.

        Parameters
        ----------
        save_fn : str, optional
            Filename to save the plot. If None, the plot is not saved.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
            Figure and axes of the plot.
        """
        fig, ax = plt.subplots(figsize=(5, 4))

        for data in self.covariance_y:
            ax.plot(data, color="gray", alpha=0.5, lw=0.1)

        mean = np.mean(self.covariance_y, axis=0)
        ax.plot(mean, color="k", lw=1.0)

        ax.set_xlabel("bin index")
        ax.set_ylabel("WST coefficient")

        # cov = np.cov(self.covariance_y, rowvar=False)
        # prec = np.linalg.inv(cov)

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches="tight")
            logger.info(f"Saving training set figure to {save_fn}")

        return fig, ax


# Alias
wst = WaveletScatteringTransform
