from pathlib import Path

import lsstypes
import matplotlib.pyplot as plt
import numpy as np
import xarray

from acm.utils.decorators import temporary_class_state
from acm.utils.default import cosmo_list
from acm.utils.plotting import set_plot_style
from acm.utils.xarray import dataset_to_dict, split_vars

from .base import BaseObservableBGS

K_MIN = 2 * np.pi / 500  # lower limit fixed by small boxsize
K_MAX = (
    np.pi * 512 / 2200
)  # Higher limit fixed by Nyquist frequency of the base boxsize (to be safe, we take the larger boxsize created, which has a lower Nyquist frequency) with 512^3 grid


class PowerSpectrumMultipoles(BaseObservableBGS):
    def __init__(self, stat_name="spectrum", **kwargs):
        super().__init__(stat_name=stat_name, **kwargs)

    # %% Compressed files creation
    @classmethod
    def compress_covariance(
        cls,
        paths: dict,
        stat_name: str = "spectrum",
        cosmo_idx: int = 0,
        hod_idx: int = 157,
        seed: int = 0,
        los: list[str] = ["x", "y", "z"],
        save_to: str = None,
        kmin: float = K_MIN,
        kmax: float = K_MAX,
        rebin: int = 3,
        ells: list = [0, 2],
        overwrite_k: np.ndarray = None,
    ) -> xarray.DataArray:
        logger = cls.get_logger()

        small_dir = Path(paths["measurements_dir"]) / "small"

        y = []
        phases = [
            int(fn.stem.split("_ph")[-1])
            for fn in sorted(small_dir.glob(f"c{cosmo_idx:03d}_ph*"))
        ]
        data_size = None  # To check consistency of data size across mocks
        for phase in phases:
            fn_dir = (
                small_dir
                / f"c{cosmo_idx:03d}_ph{phase:03d}"
                / f"seed{seed}"
                / f"hod{hod_idx:03d}"
            )
            fns = [
                fn_dir / f"power_spectrum_los_{l}.h5" for l in los
            ]  # NOTE: Hardcoded !
            existing_fns = [fn for fn in fns if fn.exists()]

            if len(existing_fns) == 0:
                raise FileNotFoundError(
                    f"No measurement files found in {fn_dir}, cannot compute covariance."
                )

            data = lsstypes.mean([lsstypes.read(fn) for fn in existing_fns])
            data = data.select(k=slice(0, None, rebin)).select(k=(kmin, kmax))

            if data_size is None:
                data_size = data.size
            if data.size != data_size:
                raise ValueError(
                    f"Inconsistent data size across mocks: expected {data_size}, got {data.size} in {fn_dir}. Cannot compute covariance."
                )

            poles = [data.get(ell) for ell in ells]
            k = poles[0].coords("k")
            y.append(np.concatenate(poles))
        y = np.array(y)
        k = overwrite_k if overwrite_k is not None else k

        y = xarray.DataArray(
            data=y.reshape(len(phases), len(ells), -1),
            coords={
                "phase_idx": phases,  # TODO: continuous phase indexing ?
                "ells": ells,
                "k": k,
            },
            attrs={
                "sample": ["phase_idx"],
                "features": ["ells", "k"],
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
        stat_name: str = "spectrum",
        phase: int = 0,
        seed: int = 0,
        add_covariance: bool = False,
        save_to: str = None,
        los: list[str] = ["x", "y", "z"],
        kmin: float = K_MIN,
        kmax: float = K_MAX,
        rebin: int = 3,
        ells: list = [0, 2],
        cosmos: list = cosmo_list,
        n_hod: int = None,
        density_threshold: float = None,
        test_filters: dict = None,
        **kwargs,
    ) -> xarray.Dataset:
        logger = cls.get_logger()

        x = cls.compress_x(
            paths=paths, cosmos=cosmos, phase=phase, seed=seed, n_hod=n_hod
        )
        n_hod = len(x.hod_idx)  # Edge case if n_hod was None

        y = []
        data_size = None  # To check consistency of data size across samples
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
                logger.debug(
                    f"Loading data for c{cosmo_idx:03d}_hod{fn_dir.stem.split('hod')[-1]}"
                )
                fns = [
                    fn_dir / f"power_spectrum_los_{l}.h5" for l in los
                ]  # NOTE: Hardcoded !
                existing_fns = [fn for fn in fns if fn.exists()]

                if len(existing_fns) == 0:
                    raise FileNotFoundError(
                        f"No measurement files found in {fn_dir}, cannot load data."
                    )

                data = lsstypes.mean([lsstypes.read(fn) for fn in existing_fns])
                data = data.select(k=slice(0, None, rebin)).select(k=(kmin, kmax))

                if data_size is None:
                    data_size = data.size
                if data.size != data_size:
                    raise ValueError(
                        f"Inconsistent data size across samples: expected {data_size}, got {data.size} in {fn_dir}. Cannot load data."
                    )

                poles = [data.get(ell) for ell in ells]
                k = poles[0].coords("k")
                y.append(np.concatenate(poles))
        y = np.array(y)

        y = xarray.DataArray(
            # NOTE: Should crash if n_hod is not consistent with the hod number from the statistics, this is intended
            data=y.reshape(len(cosmos), n_hod, len(ells), -1),
            coords={
                "cosmo_idx": cosmos,
                "hod_idx": list(range(n_hod)),  # re-index HODs to be continuous
                "ells": ells,
                "k": k,
            },
            attrs={
                "sample": ["cosmo_idx", "hod_idx"],
                "features": ["ells", "k"],
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
                kmin=kmin,
                kmax=kmax,
                ells=ells,
                overwrite_k=k,
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
            np.save(save_fn, dataset_to_dict(cout))
            logger.info(f"Saving compressed data to {save_fn}")
        return cout

    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_observable(
        self,
        model_params: dict,
        save_fn: str = None,
        ells: list = [0, 2],
        xscale: str = "linear",
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

        ax[-1].set_xlabel(r"$k [(\mathrm{Mpc}/h)]$", fontsize=15)
        ax[0].set_ylabel(r"$k^2 \P_{\ell}(k) [(\mathrm{Mpc}/h)^2]$", fontsize=15)

        volume_factor = kwargs.pop("volume_factor", 64)
        prefactor = kwargs.pop("prefactor", 1)

        # Save current select_filters and update with ells
        if self.select_filters is None:
            default_select_filters = None
            self.select_filters = {}
        else:
            default_select_filters = self.select_filters.copy()

        k = self.k.values
        for i, ell in enumerate(ells):
            self.select_filters.update({"ells": ell})
            data = self.y
            model = self.get_model_prediction(model_params)
            cov = self.get_covariance_matrix(
                volume_factor=volume_factor, prefactor=prefactor
            )
            error = np.sqrt(np.diag(cov))

            if len(data.shape) > 1:
                self.logger.warning(
                    "Multiple samples found in the data. This might lead to unexpected plotting behavior."
                )

            ax[0].errorbar(
                k,
                data * k**2,
                error * k**2,
                marker="o",
                ms=4,
                ls="",
                color=f"C{i}",
                elinewidth=1.0,
                capsize=None,
                label=rf"$\ell={ell}$",
            )
            ax[0].plot(k, model * k**2, ls="-", color=f"C{i}")
            ax[1].plot(k, (data - model) / error, ls="-", color=f"C{i}")

        for offset in [-2, 2]:
            ax[1].axhline(offset, color="k", ls="--")

        ax[1].set_ylabel(r"$\Delta{\rm X} / \sigma_{\rm data}$", fontsize=15)
        ax[1].set_ylim(-4, 4)

        # Restore select_filters
        self.select_filters = default_select_filters

        for a in ax:
            a.grid(True)
            a.set_xscale(xscale)
            a.tick_params(axis="both", labelsize=14)

        if show_legend:
            ax[0].legend(fontsize=15)

        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saving plot to {save_fn}")
        return fig, ax


# Aliases
spectrum = PowerSpectrumMultipoles
