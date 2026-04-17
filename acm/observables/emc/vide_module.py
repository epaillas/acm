from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray

from acm.utils.decorators import temporary_class_state
from acm.utils.default import cosmo_list  # List of cosmologies in AbacusSummit
from acm.utils.plotting import set_plot_style
from acm.utils.xarray import dataset_to_dict, split_vars

from .base import BaseObservableEMC


class VIDEVoidGalaxyCorrelationFunctionMultipoles(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge VIDE void-galaxy correlation
    function multipoles observable.
    """

    def __init__(self, stat_name="vide_ccf", n_test=6 * 100, **kwargs):
        super().__init__(stat_name=stat_name, n_test=n_test, **kwargs)

    @classmethod
    def compress_covariance(
        cls,
        paths: dict,
        stat_name: str = "vide_ccf",
        save_to: str = None,
        ells: list = [0, 2, 4],
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
        ells : list, optional
            List of multipoles to compute the statistics for. Default is [0, 2, 4].

        Returns
        -------
        xarray.DataArray
            Covariance array.
        """
        logger = cls.get_logger()

        measurements_dir = "/global/cfs/cdirs/desicollab/users/nschuster/ACM_VIDE_data/"
        base_dir = Path(measurements_dir)
        # base_dir = Path(self.paths['measurements_dir'],  f'base/vide/')

        filename = (
            base_dir
            / "multipoles_[0, 2, 4]_85cosmologies_100HODs_4bins_0.0-1.0_rv0.3-2.5.npz"
        )
        data = np.load(filename, allow_pickle=True)
        y = data["cov_y"]
        rv = data["rv"]
        n_stacked_bins = 4

        y = xarray.DataArray(
            data=y.reshape(y.shape[0], n_stacked_bins, len(ells), -1),
            coords={
                "phase_idx": list(range(y.shape[0])),
                "stacked_bins": list(range(n_stacked_bins)),
                "ells": ells,
                "rv": rv,
            },
            attrs={
                "sample": ["phase_idx"],
                "features": ["stacked_bins", "ells", "rv"],
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
        stat_name: str = "vide_ccf",
        add_covariance: bool = False,
        save_to: str = None,
        cosmos: list = cosmo_list,
        n_hod: int = 500,
        ells: list = [0, 2, 4],
        test_filters: dict = None,
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
        cosmos : list, optional
            List of cosmological parameters to use. If None, use all cosmological parameters.
            Default is None.
        n_hod : int, optional
            Number of HOD parameters to use. Default is 100.
        ells : list, optional
            List of multipoles to compute the statistics for. Default is [0, 2, 4].
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

        measurements_dir = "/global/cfs/cdirs/desicollab/users/nschuster/ACM_VIDE_data/"
        base_dir = Path(measurements_dir)
        # base_dir = Path(self.paths['measurements_dir'],  f'base/vide/')

        filename = (
            base_dir
            / "multipoles_[0, 2, 4]_85cosmologies_100HODs_4bins_0.0-1.0_rv0.3-2.5.npz"
        )
        data = np.load(filename, allow_pickle=True)
        y = data["y"]
        rv = data["rv"]
        n_stacked_bins = 4

        y = xarray.DataArray(
            data=y.reshape(len(cosmos), n_hod, n_stacked_bins, len(ells), -1),
            coords={
                "cosmo_idx": cosmos,
                "hod_idx": list(range(n_hod)),
                "stacked_bins": list(range(n_stacked_bins)),
                "ells": ells,
                "rv": rv,
            },
            attrs={
                "sample": ["cosmo_idx", "hod_idx"],
                "features": ["stacked_bins", "ells", "rv"],
            },
            name="y",
        )
        x = cls.compress_x(paths=paths, cosmos=cosmos, n_hod=n_hod)

        logger.info(f"Loaded data with shape: {x.shape}, {y.shape}")

        cout = xarray.Dataset(
            data_vars={
                "x": x,
                "y": y,
            },
        )
        if add_covariance:
            cov_y = cls.compress_covariance(paths=paths, stat_name=stat_name, ells=ells)
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
    def plot_training_set(self, save_fn: str = None):
        """
        Plot the training set for the observable.

        Parameters
        ----------
        save_fn : str
            Path to save the figure. If None, the figure is not saved.
            Default is None.
        """
        ells = self._dataset.y.coords["ells"].values.tolist()
        stacked_bins = self._dataset.y.coords["stacked_bins"].values.tolist()
        rv = self.rv.values

        fig, lax = plt.subplots(len(ells), 1, figsize=(4, 5), sharex=True)

        for ell in ells:
            for sb in stacked_bins:
                self.select_filters.update({"ells": ell, "stacked_bins": sb})

                for data in self.y:
                    lax[ell // 2].plot(rv, data, color=f"C{sb}", alpha=0.5, lw=0.1)

            lax[ell // 2].set_ylabel(rf"$\xi_{ell}(r / R_{{\rm void}})$")
        lax[-1].set_xlabel(r"$r / R_{\rm void}$")

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saving training set figure to {save_fn}")

        return fig, lax

    @set_plot_style
    def plot_covariance_set(self, save_fn: str = None):
        """
        Plot the covariance set for the observable.

        Parameters
        ----------
        save_fn : str
            Path to save the figure. If None, the figure is not saved.
            Default is None.
        """
        ells = self._dataset.y.coords["ells"].values.tolist()
        stacked_bins = self._dataset.y.coords["stacked_bins"].values.tolist()
        rv = self.rv.values

        fig, lax = plt.subplots(len(ells), 1, figsize=(4, 5), sharex=True)

        for ell in ells:
            for sb in stacked_bins:
                self.select_filters.update({"ells": ell, "stacked_bins": sb})

                for data in self.covariance_y:
                    lax[ell // 2].plot(rv, data, color=f"C{sb}", alpha=0.5, lw=0.1)

            lax[ell // 2].set_ylabel(rf"$\xi_{ell}(r / R_{{\rm void}})$")
        lax[-1].set_xlabel(r"$r / R_{\rm void}$")

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saving training set figure to {save_fn}")

        return fig, lax

    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_observable(self, model_params: dict, save_fn: str = None):
        """
        Plot the data, model, and residuals.

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

        ells = self._dataset.y.coords["ells"].values.tolist()
        print(ells)

        height_ratios = [max(len(ells), 3)] + [1] * len(ells)
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
        show_legend = True

        for i, ell in enumerate(ells):
            lax[0].set_ylabel(rf"$\xi_{ell}(r / R_{{\rm void}})$", fontsize=15)
            lax[-1].set_xlabel(r"$r / R_{\rm void}$", fontsize=15)

            self.select_filters.update({"ells": ell, "stacked_bins": 0})

            rv = self.rv.values
            data = self.y
            model = self.get_model_prediction(model_params)

            cov = self.get_covariance_matrix(volume_factor=64)
            error = np.sqrt(np.diag(cov))

            lax[0].errorbar(
                rv,
                data,
                error,
                marker="o",
                ms=4,
                ls="",
                color=f"C{i}",
                elinewidth=1.0,
                capsize=None,
                label=f"$\ell={ell}$",
            )
            lax[0].plot(rv, model, ls="-", color=f"C{i}")
            lax[i + 1].plot(rv, (data - model) / error, ls="-", color=f"C{i}")

            for offset in [-2, 2]:
                lax[i + 1].axhline(offset, color="k", ls="--")
            lax[i + 1].set_ylabel(
                rf"$\Delta \xi_{{{ell:d}}} / \sigma_{{ \xi_{{{ell:d}}} }}$", fontsize=15
            )
            lax[i + 1].set_ylim(-4, 4)

        for ax in lax:
            ax.grid(True)
            ax.tick_params(axis="both", labelsize=14)
        if show_legend:
            lax[0].legend(fontsize=15)

        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saving plot to {save_fn}")
        return fig, lax


class VIDEVoidSizeFunction(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge VIDE void size function observable.
    """

    def __init__(self, stat_name="vide_vsf", n_test=6 * 100, **kwargs):
        super().__init__(stat_name=stat_name, n_test=n_test, **kwargs)

    @classmethod
    def compress_covariance(
        cls,
        paths: dict,
        stat_name: str = "vide_vsf",
        save_to: str = None,
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

        measurements_dir = "/global/cfs/cdirs/desicollab/users/nschuster/ACM_VIDE_data/"
        base_dir = Path(measurements_dir)
        # base_dir = Path(self.paths['measurements_dir'],  f'base/vide/')

        filename = base_dir / "vsf_85cosmologies_100HODs_10.0-80.0_5Mpc_steps.npz"
        data = np.load(filename, allow_pickle=True)
        y = data["cov_y"]
        rv = data["s"]

        y = xarray.DataArray(
            data=y.reshape(y.shape[0], -1),
            coords={
                "phase_idx": list(range(y.shape[0])),
                "rv": rv,
            },
            attrs={
                "sample": ["phase_idx"],
                "features": ["rv"],
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
        stat_name: str = "vide_vsf",
        add_covariance: bool = False,
        save_to: str = None,
        cosmos: list = cosmo_list,
        n_hod: int = 500,
        test_filters: dict = None,
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
        add_covariance : bool
            If True, add the covariance to the compressed data. Default is False.
        save_to : str, optional
            Path of the directory where to save the compressed file. If None, it is not saved.
            Default is None.
        cosmos : list, optional
            List of cosmological parameters to use. If None, use all cosmological parameters.
            Default is None.
        n_hod : int, optional
            Number of HOD parameters to use. Default is 100.
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

        measurements_dir = "/global/cfs/cdirs/desicollab/users/nschuster/ACM_VIDE_data/"
        base_dir = Path(measurements_dir)
        # base_dir = Path(self.paths['measurements_dir'],  f'base/vide/')

        filename = base_dir / "vsf_85cosmologies_100HODs_10.0-80.0_5Mpc_steps.npz"
        data = np.load(filename, allow_pickle=True)
        y = data["y"]
        rv = data["s"]

        y = xarray.DataArray(
            data=y.reshape(len(cosmos), n_hod, -1),
            coords={
                "cosmo_idx": cosmos,
                "hod_idx": list(range(n_hod)),
                "rv": rv,
            },
            attrs={
                "sample": ["cosmo_idx", "hod_idx"],
                "features": ["rv"],
            },
            name="y",
        )
        x = cls.compress_x(paths=paths, cosmos=cosmos, n_hod=n_hod)

        logger.info(f"Loaded data with shape: {x.shape}, {y.shape}")

        cout = xarray.Dataset(
            data_vars={
                "x": x,
                "y": y,
            },
        )
        if add_covariance:
            cov_y = cls.compress_covariance(
                paths=paths, stat_name=stat_name, cosmos=cosmos, n_hod=n_hod
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
    def plot_training_set(self, save_fn: str = None):
        """
        Plot the training set for the observable.

        Parameters
        ----------
        save_fn : str
            Path to save the figure. If None, the figure is not saved.
            Default is None.
        """
        rv = self.rv.values

        fig, ax = plt.subplots(figsize=(5, 4))

        for data in self.y:
            ax.plot(rv, data, alpha=0.5, lw=0.3)

        ax.set_ylabel(r"$\textrm{VSF}$")
        ax.set_xlabel(r"$R_{\rm void}\, [h^{-1}{\rm Mpc}]$")

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saving training set figure to {save_fn}")

        return fig, ax

    @set_plot_style
    def plot_covariance_set(self, save_fn: str = None):
        """
        Plot the covariance set for the observable.

        Parameters
        ----------
        save_fn : str
            Path to save the figure. If None, the figure is not saved.
            Default is None.
        """
        rv = self.rv.values

        fig, ax = plt.subplots(figsize=(5, 4))

        for data in self.covariance_y:
            ax.plot(rv, data, color="grey", alpha=0.5, lw=0.1)

        ax.set_ylabel(r"$\textrm{VSF}$")
        ax.set_xlabel(r"$R_{\rm void}\, [h^{-1}{\rm Mpc}$")

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saving training set figure to {save_fn}")

        return fig, ax

    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_observable(self, model_params: dict, save_fn: str = None):
        """
        Plot the data, model, and residuals.

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

        lax[-1].set_xlabel(r"$R_{\rm void}\,[h^{-1}{\rm Mpc}]$]", fontsize=15)
        lax[0].set_ylabel(r"$\textrm{VSF}$", fontsize=15)

        rv = self.rv.values
        data = self.y
        model = self.get_model_prediction(model_params)

        cov = self.get_covariance_matrix(volume_factor=64)
        error = np.sqrt(np.diag(cov))

        lax[0].errorbar(
            rv,
            data,
            error,
            marker="o",
            ms=3,
            ls="",
            color=f"C0",
            elinewidth=1.0,
            capsize=None,
        )
        lax[0].plot(rv, model, ls="-", color=f"C1")
        lax[1].plot(rv, (data - model) / error, ls="-", color=f"C0")

        for offset in [-2, 2]:
            lax[1].axhline(offset, color="k", ls="--")
        lax[1].set_ylabel(r"$\Delta \textrm{VSF} / \sigma_\textrm{VSF}$", fontsize=15)
        lax[1].set_ylim(-4, 4)

        for ax in lax:
            ax.grid(True)
            ax.tick_params(axis="both", labelsize=14)
        if show_legend:
            lax[0].legend(fontsize=15)

        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saving plot to {save_fn}")
        return fig, lax


# Aliases
vide_ccf = VIDEVoidGalaxyCorrelationFunctionMultipoles
vide_vsf = VIDEVoidSizeFunction
