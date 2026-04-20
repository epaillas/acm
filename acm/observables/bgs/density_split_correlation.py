import logging
from pathlib import Path
from typing import override

import matplotlib.pyplot as plt
import numpy as np
import xarray

from acm.utils.decorators import temporary_class_state
from acm.utils.default import cosmo_list  # List of cosmologies in AbacusSummit
from acm.utils.plotting import set_plot_style
from acm.utils.xarray import dataset_to_dict, split_vars

from .base import BaseObservableBGS

logger = logging.getLogger(__name__)


class DensitySplitBaseClass(BaseObservableBGS):
    """Base class for densitysplit observables in the ACM pipeline for the BGS dataset."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    # %% Compressed files creation
    @classmethod
    def compress_covariance(
        cls,
        paths: dict,
        stat_name: str,
        measurement_root: str,
        cosmo_idx: int = 0,
        hod_idx: int = 157,
        seed: int = 0,
        los: list[str] = ["x", "y", "z"],
        save_to: str | None = None,
        rebin: int = 1,
        ells: list = [0, 2],
        quantiles: list = [0, 1, 3, 4],
        overwrite_s: np.ndarray | None = None,
    ) -> xarray.Dataset:
        """
        Compress the covariance array from the raw measurement files.

        Parameters
        ----------
        paths : dict
            Dictionary containing the paths to the data directories.
        stat_name : str, optional
            Name of the statistic to use.
            Defines the name of the subfolder in the measurements directory, and the
            saved filename if save_to is provided.
            Defaults to the class's stat_name.
        measurement_root : str
            Root name of the measurement files.
        cosmo_idx : int
            Index of the cosmology to use. Default is 0.
        hod_idx : int
            Index of the HOD to use. Default is 157.
        seed : int
            Seed index to use. Default is 0.
        los : list[str], optional
            List of line-of-sight directions to use. Default is ['x', 'y', 'z'].
        save_to : str, optional
            Path of the directory where to save the compressed covariance and bin_values. If None, it is not saved.
            Default is None.
        rebin : int, optional
            Rebinning factor for the statistics. Default is 1.
        ells : list, optional
            List of multipoles to compute the statistics for. Default is [0, 2].
        quantiles : list, optional
            List of quantiles to compute the statistics for. Default is [0, 1, 3, 4].
        overwrite_s : np.ndarray, optional
            If not None, overwrite the final separation values with this array.
            This is primarily useful to ensure consistency between the covariance and the data dims.
            Default is None.

        Returns
        -------
        xarray.Dataset
            Compressed dataset containing the covariance and bin values.
        """
        small_dir = Path(paths["measurements_dir"]) / "small"

        y = []
        phases = [
            int(fn.stem.split("_ph")[-1])
            for fn in sorted(small_dir.glob(f"c{cosmo_idx:03d}_ph*"))
        ]  # List of available phases from files

        for phase in phases:
            y_quantiles = []
            for q in quantiles:
                fn_dir = (
                    small_dir
                    / f"c{cosmo_idx:03d}_ph{phase:03d}"
                    / f"seed{seed}"
                    / f"hod{hod_idx:03d}"
                )
                fns = [
                    fn_dir / f"{measurement_root}_los_{_l}.npy" for _l in los
                ]  # NOTE: Hardcoded !
                existing_fns = [fn for fn in fns if fn.exists()]
                if len(existing_fns) == 0:
                    raise FileNotFoundError(
                        f"No measurement files found in {fn_dir} for quantile {q}, cannot compute covariance."
                    )
                data = sum(
                    [
                        np.load(fn, allow_pickle=True)[q].normalize()
                        for fn in existing_fns
                    ]
                )
                s, multipoles = data[::rebin](ells=ells, return_sep=True)
                y_quantiles.append(multipoles)
            y.append(y_quantiles)
        y = np.array(y)
        s = overwrite_s if overwrite_s is not None else s

        y = xarray.DataArray(
            data=y.reshape(len(phases), len(quantiles), len(ells), -1),
            coords={
                "phase_idx": phases,  # TODO: continuous phase indexing ?
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
            payload = np.array(dataset_to_dict(cout), dtype=object)
            np.save(save_fn, payload)
            logger.info(f"Saving compressed covariance file to {save_fn}")
        return cout

    @classmethod
    def compress_data(
        cls,
        paths: dict,
        stat_name: str,
        measurement_root: str,
        phase: int = 0,
        seed: int = 0,
        add_covariance: bool = False,
        save_to: str | None = None,
        los: list[str] = ["x", "y", "z"],
        rebin: int = 1,
        ells: list = [0, 2],
        quantiles: list = [0, 1, 3, 4],
        cosmos: list = cosmo_list,
        n_hod: int | None = None,
        density_threshold: float | None = None,
        test_filters: dict | None = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Compress the data from the tpcf raw measurement files.

        Parameters
        ----------
        paths : dict
            Dictionary containing the paths to the data directories.
        stat_name : str, optional
            Name of the statistic to use.
            Defines the name of the subfolder in the measurements directory, and the
            saved filename if save_to is provided.
            Defaults to the class's stat_name.
        measurement_root : str
            Root name of the measurement files.
        phase : int, optional
            Phase index to read the data from. Default is 0.
        seed : int, optional
            Seed index to read the data from. Default is 0.
        add_covariance : bool
            If True, add the covariance to the compressed data. Default is False.
        save_to : str
            Path of the directory where to save the compressed file. If None, it is not saved.
            Default is None.
        los : list[str]
            List of line-of-sight directions to use. Default is ['x', 'y', 'z'].
        rebin : int
            Rebinning factor for the statistics. Default is 1.
        ells : list
            List of multipoles to compute the statistics for. Default is [0, 2].
        quantiles : list
            List of quantiles to compute the statistics for. Default is [0, 1, 3, 4].
        cosmos : list
            List of cosmological parameters to use. If None, use all cosmological parameters.
            Default is cosmo_list.
        n_hod : int, optional
            Number of HODs to consider per cosmology.
            If None, it is determined from the first cosmology and restricted to that number for all cosmologies.
            Defaults to None.
        density_threshold : float, optional
            Density threshold to use for the HOD mocks selection. If None, use all available HOD mocks.
            Default is None.
        test_filters : dict, optional
            Dictionary of filters to split the dataset into training and test sets.
            Keys are the dimension names and values are the values to filter on for the test set.
            If None, no splitting is done. Default is None.
        **kwargs
            Extra arguments to pass to `compress_covariance` (`cosmo_idx` or `hod_idx`) if needed.
            See their documentation for details and default values.

        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'x' and 'y' DataArrays.
            If add_covariance is True, also contains 'covariance_y' DataArray.
        """
        x = cls.compress_x(
            paths=paths, cosmos=cosmos, phase=phase, seed=seed, n_hod=n_hod
        )
        n_hod = len(x.hod_idx)  # Edge case if n_hod was None

        y = []
        for cosmo_idx in cosmos:
            # Get the HODs folders available for this cosmology
            hod_fns = cls.get_hod_from_files(
                paths=paths,
                cosmo_idx=cosmo_idx,
                phase=phase,
                seed=seed,
                density_threshold=density_threshold,
                return_fn=True,
            )[:n_hod]  # Restrict to n_hod if needed

            for fn_dir in hod_fns:
                y_quantiles = []
                fns = [
                    fn_dir / f"{measurement_root}_los_{_l}.npy" for _l in los
                ]  # NOTE: Hardcoded !
                for q in quantiles:
                    data = sum(
                        [
                            np.load(fn, allow_pickle=True)[q].normalize()
                            for fn in fns
                            if fn.exists()
                        ]
                    )
                    if data == 0:
                        raise FileNotFoundError(
                            f"No measurement files found in {fn_dir} for quantile {q}, cannot load data."
                        )
                    s, multipoles = data[::rebin](ells=ells, return_sep=True)
                    y_quantiles.append(multipoles)
                y.append(y_quantiles)
        y = np.array(y)

        y = xarray.DataArray(
            # NOTE: Should crash if n_hod is not consistent with the hod number from the statistics, this is intended
            data=y.reshape(len(cosmos), n_hod, len(quantiles), len(ells), -1),
            coords={
                "cosmo_idx": cosmos,
                "hod_idx": list(range(n_hod)),  # re-index HODs to be continuous
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
                stat_name=stat_name,
                rebin=rebin,
                ells=ells,
                quantiles=quantiles,
                overwrite_s=s,
                seed=seed,
                los=los,
                **kwargs,
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

    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_observable(
        self,
        model_params: dict,
        save_fn: str | None = None,
        quantiles: list = [0, 1, 3, 4],
        ell: int = 0,
        **kwargs,
    ) -> tuple:
        """
        Plot the observable with error bars and the model prediction, along with the residuals.

        Parameters
        ----------
        model_params : dict
            Dictionary of model parameters for the prediction.
        save_fn : str, optional
            Filename to save the plot. If None, the plot is not saved.
        quantiles : list, optional
            List of quantiles to plot. Defaults to [0, 1, 3, 4].
        ell : int, optional
            Multipole moment to plot. Defaults to 0.
        **kwargs : dict
            Additional arguments for the plot, such as height_ratios and show_legend, and volume_factor and prefactor for covariance calculation.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, numpy.ndarray
            Figure and axes of the plot.
        """
        height_ratios = kwargs.pop("height_ratios", [3, 1])
        show_legend = kwargs.pop("show_legend", False)
        figsize = (6, 1.5 * sum(height_ratios))
        fig, ax = plt.subplots(
            len(height_ratios),
            sharex=True,
            sharey=False,
            gridspec_kw={"height_ratios": height_ratios},
            figsize=figsize,
            squeeze=True,
        )
        fig.subplots_adjust(hspace=0.1)

        ax[-1].set_xlabel(r"$s [(\mathrm{Mpc}/h)]$", fontsize=15)
        ax[0].set_ylabel(r"$s^2 \xi_{0}(s) [(\mathrm{Mpc}/h)^2]$", fontsize=15)

        volume_factor = kwargs.pop("volume_factor", 64)
        prefactor = kwargs.pop("prefactor", 1)

        # Save current select_filters and update with ells
        if self.select_filters is None:
            default_select_filters = None
            self.select_filters = {}
        else:
            default_select_filters = self.select_filters.copy()

        s = self.s.values
        for _, q in enumerate(quantiles):
            self.select_filters.update({"ells": ell, "quantiles": q})
            data = self.y
            model = self.get_model_prediction(model_params)
            cov = self.get_covariance_matrix(
                volume_factor=volume_factor, prefactor=prefactor
            )
            error = np.sqrt(np.diag(cov))

            if len(data.shape) > 1:
                logger.warning(
                    "Multiple samples found in the data. This might lead to unexpected plotting behavior."
                )

            ax[0].errorbar(
                s,
                data * s**2,
                error * s**2,
                marker="o",
                ms=4,
                ls="",
                color=f"C{q}",
                elinewidth=1.0,
                capsize=None,
                label=rf"$Q{q}$",
            )
            ax[0].plot(s, model * s**2, ls="-", color=f"C{q}")
            ax[1].plot(s, (data - model) / error, ls="-", color=f"C{q}")

        for offset in [-2, 2]:
            ax[1].axhline(offset, color="k", ls="--")

        ax[1].set_ylabel(r"$\Delta{\rm X} / \sigma_{\rm data}$", fontsize=15)
        ax[1].set_ylim(-4, 4)

        # Restore select_filters
        self.select_filters = default_select_filters

        for a in ax:
            a.grid(True)
            a.tick_params(axis="both", labelsize=14)

        if show_legend:
            ax[0].legend(fontsize=15)

        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches="tight")
            logger.info(f"Saving plot to {save_fn}")
        return fig, ax


class DensitySplitQuantileGalaxyCorrelationFunctionMultipoles(DensitySplitBaseClass):
    """Class for the application of the densitysplit cross-correlation statistic of the ACM pipeline to the BGS dataset."""

    def __init__(self, stat_name: str = "ds_xiqg", **kwargs) -> None:
        super().__init__(stat_name=stat_name, **kwargs)

    @override
    @classmethod
    def compress_covariance(cls, **kwargs) -> xarray.Dataset:
        kwargs["measurement_root"] = kwargs.pop(
            "measurement_root", "quantile_data_correlation"
        )
        kwargs["stat_name"] = kwargs.get("stat_name", "ds_xiqg")
        return super().compress_covariance(**kwargs)

    @override
    @classmethod
    def compress_data(cls, **kwargs) -> xarray.Dataset:
        kwargs["measurement_root"] = kwargs.pop(
            "measurement_root", "quantile_data_correlation"
        )
        kwargs["stat_name"] = kwargs.get("stat_name", "ds_xiqg")
        return super().compress_data(**kwargs)


class DensitySplitQuantileCorrelationFunctionMultipoles(DensitySplitBaseClass):
    """Class for the application of the densitysplit auto-correlation statistic of the ACM pipeline to the BGS dataset."""

    def __init__(self, stat_name: str = "ds_xiqq", **kwargs) -> None:
        super().__init__(stat_name=stat_name, **kwargs)

    @override
    @classmethod
    def compress_covariance(cls, **kwargs) -> xarray.Dataset:
        kwargs["measurement_root"] = kwargs.pop(
            "measurement_root", "quantile_correlation"
        )
        kwargs["stat_name"] = kwargs.get("stat_name", "ds_xiqq")
        return super().compress_covariance(**kwargs)

    @override
    @classmethod
    def compress_data(cls, **kwargs) -> xarray.Dataset:
        kwargs["measurement_root"] = kwargs.pop(
            "measurement_root", "quantile_correlation"
        )
        kwargs["stat_name"] = kwargs.get("stat_name", "ds_xiqq")
        return super().compress_data(**kwargs)


# Aliases
ds_xiqq = DensitySplitQuantileCorrelationFunctionMultipoles
ds_xiqg = DensitySplitQuantileGalaxyCorrelationFunctionMultipoles
