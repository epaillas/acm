import logging
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from jaxpower import (
    BinMesh2SpectrumPoles,
    ParticleField,
    compute_box2_normalization,
    compute_fkp2_shotnoise,
    compute_mesh2_spectrum,
)
from lsstypes import ObservableTree
from lsstypes.external import from_pycorr
from matplotlib import cm
from matplotlib.patches import Patch
from pandas import qcut
from pycorr import TwoPointCorrelationFunction

from acm.utils.plotting import set_plot_style

from .base import BaseEstimator

logger = logging.getLogger(__name__)


class DensitySplit(BaseEstimator):
    """
    Class to compute density-split clustering, as in http://arxiv.org/abs/2309.16541.

    Expects all positions passed in cartesian coordinates of shape (N, 3).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def set_quantiles(
        self,
        query_positions: np.ndarray | None = None,
        query_method: str = "randoms",
        nquery_factor: int = 5,
        nquantiles: int = 5,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """
        Get the quantiles of the overdensity density field.

        Parameters
        ----------
        query_positions : array_like, optional
            Query positions.
        query_method : str, optional
            Method to generate query points. Options are 'lattice' or 'randoms'.
        nquery_factor : int, optional
            Factor to multiply the number of data points to get the number of query points.
        nquantiles : int
            Number of quantiles.

        Returns
        -------
        quantiles : array_like
            Quantiles of the density field.
        quantiles_idx : array_like, optional
            Index of the quantile of each query point.
        delta_query : array_like, optional
            Density contrast at the query points.
        """
        t0 = time.time()
        if query_positions is None:
            if self.has_randoms:
                raise ValueError(
                    "query_positions must be provided when working with a non-uniform geometry."
                )
            query_positions = self.get_query_positions(
                method=query_method, nquery=nquery_factor * self.size_data
            )
        self.query_method = query_method
        self.query_positions = query_positions
        self.delta_query = self.read_density_contrast(query_positions)
        self.quantiles_idx = qcut(self.delta_query, nquantiles, labels=False)
        quantiles = [
            self.query_positions[self.quantiles_idx == i] for i in range(nquantiles)
        ]
        self.quantiles = quantiles
        self.nquantiles = nquantiles
        logger.info(f"Quantiles calculated in {time.time() - t0:.2f} seconds.")
        return self.quantiles, self.quantiles_idx, self.delta_query

    def save(
        self, data: list, filename: str | Path, data_type: str = "correlation"
    ) -> None:
        """
        Save the per-quantile correlations or power spectra to disk.

        Parameters
        ----------
        data : list
            List of per-quantile correlation or power-spectrum measurements.
        filename : str or path-like
            Output filename where the data will be written.
        data_type : str, optional
            Type of data being saved. Options are 'correlation' or 'power'.
        """
        attrs = {
            "nquantiles": self.nquantiles,
            "query_method": self.query_method,
            "boxsize": self.boxsize,
            "meshsize": self.meshsize,
        }
        if data_type == "correlation":
            self.save_correlations(data, filename, attrs=attrs)
        elif data_type == "power":
            self.save_powers(data, filename, attrs=attrs)
        else:
            raise ValueError(
                f"Unknown type '{data_type}'. Available types: 'correlation', 'power'"
            )

    def save_correlations(
        self,
        correlations: list,
        filename: str | Path,
        attrs: dict | None = None,
    ) -> None:
        """
        Save a list of pycorr correlation objects to an lsstypes ObservableTree.

        Parameters
        ----------
        correlations : list
            List of per-quantile correlation measurements.
        filename : str or path-like
            Output filename where the ObservableTree will be written.
        """
        if jax.process_index() != 0:  # Only process 0 saves to disk
            return  # Exit early for non-zero processes

        path = Path(filename)
        logger.info(f"Saving to {filename}")

        if path.suffix in [".hdf5", ".h5"]:
            leaves = []
            for quantile in range(self.nquantiles):
                corr = from_pycorr(correlations[quantile])
                leaves.append(corr)
            tree = ObservableTree(
                leaves, quantiles=list(range(self.nquantiles)), attrs=attrs
            )
            tmp_filename = path.with_name(path.stem + ".tmp" + path.suffix)
            tree.write(tmp_filename)
            tmp_filename.replace(path)  # Atomic move to avoid partial writes
        elif path.suffix == ".npy":
            np.save(filename, correlations)
        else:
            raise ValueError(
                f"Unrecognized file extension '{path.suffix}' for file: {filename}. "
                "Supported extensions are: .hdf5, .h5, .npy"
            )

    def save_powers(
        self,
        powers: list,
        filename: str | Path,
        attrs: dict | None = None,
    ) -> None:
        """
        Save a list of per-quantile power-spectrum objects to an lsstypes ObservableTree.

        Parameters
        ----------
        powers : list
            List of per-quantile power-spectrum measurements.
        filename : str or path-like
            Output filename where the ObservableTree will be written.
        """
        if jax.process_index() != 0:  # Only process 0 saves to disk
            return  # Exit early for non-zero processes

        path = Path(filename)
        logger.info(f"Saving to {filename}")

        if path.suffix in [".hdf5", ".h5"]:
            leaves = [powers[quantile] for quantile in range(self.nquantiles)]
            tree = ObservableTree(
                leaves, quantiles=list(range(self.nquantiles)), attrs=attrs
            )
            tmp_filename = path.with_name(path.stem + ".tmp" + path.suffix)
            tree.write(tmp_filename)
            tmp_filename.replace(path)  # Atomic move to avoid partial writes
        elif path.suffix == ".npy":
            np.save(filename, powers)
        else:
            raise ValueError(
                f"Unrecognized file extension '{path.suffix}' for file: {filename}. "
                "Supported extensions are: .hdf5, .h5, .npy"
            )

    def quantile_data_correlation(
        self,
        data_positions: np.ndarray,
        save_fn: str | Path | None = None,
        **kwargs,
    ) -> list:
        """
        Compute the cross-correlation function between the density field quantiles and the data.

        Parameters
        ----------
        data_positions : array_like
            Positions of the data.
        save_fn : str or path-like, optional
            If provided, save the per-quantile correlations to disk using
            :meth:`save_correlations`.
        kwargs : dict
            Additional arguments for pycorr.TwoPointCorrelationFunction.

        Returns
        -------
        quantile_data_ccf : list
            Cross-correlation function between quantiles and data.
        """
        if self.has_randoms:
            if "randoms_positions" not in kwargs:
                raise ValueError(
                    "Randoms positions must be provided when working with a non-uniform geometry."
                )
            kwargs["randoms_positions1"] = kwargs["randoms_positions"]
            kwargs["randoms_positions2"] = kwargs["randoms_positions"]
            kwargs.pop("randoms_positions")
            if "data_weights" in kwargs:
                kwargs["data_weights1"] = None  # setting default weights for quantiles
                kwargs["data_weights2"] = kwargs.pop("data_weights")
            if "randoms_weights" in kwargs:
                kwargs["randoms_weights1"] = None
                kwargs["randoms_weights2"] = kwargs.pop("randoms_weights")
        elif "boxsize" not in kwargs:
            kwargs["boxsize"] = self.boxsize
        self._quantile_data_correlation = []
        R1R2 = None
        for quantile in self.quantiles:
            result = TwoPointCorrelationFunction(
                data_positions1=quantile,
                data_positions2=data_positions,
                mode="smu",
                position_type="pos",
                R1R2=R1R2,
                **kwargs,
            )
            self._quantile_data_correlation.append(result)
            if "estimator" in kwargs and kwargs["estimator"] != "davispeebles":
                R1R2 = result.R1R2

            # R1R2 = result.R1R2
        if save_fn is not None:
            self.save(self._quantile_data_correlation, save_fn, data_type="correlation")
        return self._quantile_data_correlation

    def quantile_correlation(self, save_fn: str | Path | None = None, **kwargs) -> list:
        """
        Compute the auto-correlation function of the density field quantiles.

        Parameters
        ----------
        save_fn : str or path-like, optional
            If provided, save the per-quantile correlations to disk using
            :meth:`save_correlations`.
        kwargs : dict
            Additional arguments for pycorr.TwoPointCorrelationFunction.

        Returns
        -------
        quantile_acf : list
            Auto-correlation function of quantiles.
        """
        if self.has_randoms:
            if "randoms_positions" not in kwargs:
                raise ValueError(
                    "Randoms positions must be provided when working with a non-uniform geometry."
                )
            kwargs["randoms_positions1"] = kwargs.pop("randoms_positions")
            kwargs["data_weights1"] = None  # setting default weights for quantiles
            kwargs["randoms_weights1"] = None
        elif "boxsize" not in kwargs:
            kwargs["boxsize"] = self.boxsize
        self._quantile_correlation = []
        R1R2 = None
        for quantile in self.quantiles:
            result = TwoPointCorrelationFunction(
                data_positions1=quantile,
                mode="smu",
                position_type="pos",
                R1R2=R1R2,
                **kwargs,
            )
            self._quantile_correlation.append(result)
            if "estimator" in kwargs and kwargs["estimator"] != "davispeebles":
                R1R2 = result.R1R2
        if save_fn is not None:
            self.save(self._quantile_correlation, save_fn, data_type="correlation")
        return self._quantile_correlation

    def quantile_data_power(
        self,
        data_positions: np.ndarray,
        edges: dict = {"step": 0.001},
        ells: tuple | list = (0, 2, 4),
        los: str = "z",
        resampler: str = "tsc",
        interlacing: int = 0,
        compensate: bool = True,
        save_fn: str | Path | None = None,
        **kwargs,
    ) -> list:
        """
        Compute the cross-power spectrum between the data and the density field quantiles.

        Parameters
        ----------
        data_positions : array_like
            Positions of the data.
        edges : dict, optional
            Bin edges for the power spectrum.
        ells : tuple, optional
            Multipole moments to compute.
        los : str, optional
            Line-of-sight direction.
        resampler : str, optional
            Resampling scheme for the mesh painting.
        interlacing : int, optional
            Interlacing factor for the mesh painting.
        compensate : bool, optional
            Whether to apply compensation for the mass assignment scheme.
        save_fn : str or path-like, optional
            If provided, save the per-quantile spectra to disk using
            :meth:`save_powers`.
        kwargs : dict
            Additional arguments for pypower.CatalogFFTPower.

        Returns
        -------
        quantile_data_power : list
            Cross-power spectrum between quantiles and data.
        """
        if self.has_randoms:
            if "randoms_positions" not in kwargs:
                raise ValueError(
                    "Randoms positions must be provided when working with a non-uniform geometry."
                )
            kwargs["randoms_positions1"] = kwargs["randoms_positions"]
            kwargs["randoms_positions2"] = kwargs["randoms_positions"]
            kwargs.pop("randoms_positions")
            if "data_weights" in kwargs:
                kwargs["data_weights1"] = None  # setting default weights for quantiles
                kwargs["data_weights2"] = kwargs.pop("data_weights")
            if "randoms_weights" in kwargs:
                kwargs["randoms_weights1"] = None
                kwargs["randoms_weights2"] = kwargs.pop("randoms_weights")
        elif "boxsize" not in kwargs:
            kwargs["boxsize"] = self.boxsize

        # TODO handle survey-mode geometry with FKPField for data mesh

        jitted_compute_mesh2_spectrum = jax.jit(
            compute_mesh2_spectrum, static_argnames=["los"], donate_argnums=[0]
        )

        bin_mesh = BinMesh2SpectrumPoles(
            self.mattrs,
            edges=edges,
            ells=ells,
        )

        kw = dict(resampler=resampler, compensate=compensate, interlacing=interlacing)

        data = ParticleField(
            data_positions, attrs=self.mattrs, exchange=True, backend="jax"
        )
        data_mesh = data.paint(**kw, out="real")
        data_mesh = data_mesh - data_mesh.mean()

        self._quantile_data_power = []
        for i, quantile_positions in enumerate(self.quantiles):
            t0 = time.time()

            quantile = ParticleField(
                quantile_positions, attrs=self.mattrs, exchange=True, backend="jax"
            )

            norm = compute_box2_normalization(quantile, data, bin=bin_mesh)

            quantile_mesh = quantile.paint(**kw, out="real")
            quantile_mesh = quantile_mesh - quantile_mesh.mean()

            spectrum = jitted_compute_mesh2_spectrum(
                quantile_mesh, data_mesh, bin=bin_mesh, los=los
            )
            spectrum = spectrum.clone(norm=norm)

            self._quantile_data_power.append(spectrum)
            logger.info(f"Q{i}-galaxy spectrum calculated in {time.time() - t0:.2f} s.")
        if save_fn is not None:
            self.save(self._quantile_data_power, save_fn, data_type="power")
        return self._quantile_data_power

    def quantile_power(
        self,
        edges: dict = {"step": 0.001},
        ells: tuple | list = (0, 2, 4),
        los: str = "z",
        resampler: str = "tsc",
        interlacing: int = 0,
        compensate: bool = True,
        save_fn: str | Path | None = None,
        **kwargs,
    ) -> list:
        """
        Compute the auto-power spectrum of the density field quantiles.

        Parameters
        ----------
        data_positions : array_like
            Positions of the data.
        edges : dict, optional
            Bin edges for the power spectrum.
        ells : tuple, optional
            Multipole moments to compute.
        los : str, optional
            Line-of-sight direction.
        resampler : str, optional
            Resampling scheme for the mesh painting.
        interlacing : int, optional
            Interlacing factor for the mesh painting.
        compensate : bool, optional
            Whether to apply compensation for the mass assignment scheme.
        save_fn : str or path-like, optional
            If provided, save the per-quantile spectra to disk using
            :meth:`save_powers`.
        kwargs : dict
            Additional arguments for pypower.CatalogFFTPower.

        Returns
        -------
        quantile_power : list
            Auto-power spectrum of quantiles.
        """
        if self.has_randoms:
            if "randoms_positions" not in kwargs:
                raise ValueError(
                    "Randoms positions must be provided when working with a non-uniform geometry."
                )
            kwargs["randoms_positions1"] = kwargs.pop("randoms_positions")
            kwargs["data_weights1"] = None  # setting default weights for quantiles
            kwargs["randoms_weights1"] = None
        elif "boxsize" not in kwargs:
            kwargs["boxsize"] = self.boxsize

        # TODO handle survey-mode geometry with FKPField for data mesh

        jitted_compute_mesh2_spectrum = jax.jit(
            compute_mesh2_spectrum, static_argnames=["los"], donate_argnums=[0]
        )

        bin_mesh = BinMesh2SpectrumPoles(
            self.mattrs,
            edges=edges,
            ells=ells,
        )

        kw = dict(resampler=resampler, compensate=compensate, interlacing=interlacing)

        self._quantile_power = []
        for i, quantile_positions in enumerate(self.quantiles):
            t0 = time.time()
            quantile = ParticleField(
                quantile_positions, attrs=self.mattrs, exchange=True, backend="jax"
            )

            norm = compute_box2_normalization(quantile, bin=bin_mesh)
            num_shotnoise = compute_fkp2_shotnoise(quantile, bin=bin_mesh)

            quantile_mesh = quantile.paint(**kw, out="real")
            # quantile_mesh = quantile_mesh / quantile_mesh.mean() - 1.
            quantile_mesh = quantile_mesh - quantile_mesh.mean()

            spectrum = jitted_compute_mesh2_spectrum(
                quantile_mesh, bin=bin_mesh, los=los
            )
            spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

            self._quantile_power.append(spectrum)
            logger.info(f"Q{i} auto-spectrum calculated in {time.time() - t0:.2f} s.")
        if save_fn is not None:
            self.save(self._quantile_power, save_fn, data_type="power")
        return self._quantile_power

    @set_plot_style
    def plot_quantiles(self, save_fn: str | Path | None = None) -> plt.Figure:
        """Plot the quantiles of the overdensity field."""
        fig, ax = plt.subplots(figsize=(4, 4))
        cmap = cm.get_cmap("coolwarm")
        colors = cmap(np.linspace(0.01, 0.99, 5))
        _, bin_edges, bar_container = ax.hist(
            self.delta_query,
            bins=200,
            density=True,
            lw=2.0,
            color="grey",
        )
        imin = 0
        handles = []
        patches = getattr(
            bar_container, "patches", []
        )  # handle both BarContainer and list of patches
        for i in range(len(self.quantiles)):
            dmax = self.delta_query[self.quantiles_idx == i].max()
            imax = np.digitize([dmax], bin_edges)[0] - 1
            for patch in patches[imin:imax]:
                patch.set_facecolor(colors[i])
            imin = imax
            handles.append(Patch(color=colors[i], label=rf"${{\rm Q}}_{i}$"))
        ax.set_xlabel(r"$\Delta \left(R_s = 10\, h^{-1}{\rm Mpc}\right)$", fontsize=15)
        ax.set_ylabel("PDF", fontsize=15)
        ax.set_xlim(-1.3, 3.0)
        ax.legend(handlelength=1.0, handles=handles)
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn, bbox_inches="tight", dpi=300)
        return fig

    @set_plot_style
    def plot_quantile_data_correlation(
        self, ell: int = 0, save_fn: str | Path | None = None
    ) -> plt.Figure:
        """Plot the cross-correlation functions for each quantile with the data."""
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            s, multipoles = self._quantile_data_correlation[i](
                ells=(0, 2, 4),
                return_sep=True,
            )
            ax.plot(
                s,
                s**2 * multipoles[ell // 2],
                lw=2.0,
                color=colors[i],
                label=rf"${{\rm Q}}_{i}$",
            )
        ax.set_xlabel(r"$s\, [h^{-1}{\rm Mpc}]$", fontsize=15)
        ax.set_ylabel(r"$s^2 \xi_\ell\, [h^{-2}{\rm Mpc^2}](s)$", fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn, bbox_inches="tight", dpi=300)
        return fig

    @set_plot_style
    def plot_quantile_correlation(
        self, ell: int = 0, save_fn: str | Path | None = None
    ) -> plt.Figure:
        """Plot the auto-correlation functions for each quantile."""
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            s, multipoles = self._quantile_correlation[i](
                ells=(0, 2, 4), return_sep=True
            )
            ax.plot(s, s**2 * multipoles[ell // 2], lw=2.0, label=rf"${{\rm Q}}_{i}$")
        ax.set_xlabel(r"$s\, [h^{-1}{\rm Mpc}]$", fontsize=15)
        ax.set_ylabel(r"$s^2 \xi_\ell\, [h^{-2}{\rm Mpc^2}](s)$", fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn, bbox_inches="tight")
        plt.show()
        return fig

    @set_plot_style
    def plot_quantile_data_power(
        self, ell: int = 0, save_fn: str | Path | None = None
    ) -> plt.Figure:
        """Plot the power spectrum for each quantile with the data."""
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            k, poles = self._quantile_data_power[i](
                ell=(0, 2, 4), return_k=True, complex=False
            )
            ax.plot(k, k * poles[ell // 2], lw=2.0, label=rf"${{\rm Q}}_{i}$")
        ax.set_xlabel(r"$k\, [h\,{\rm Mpc}^{-1}]$", fontsize=15)
        ax.set_ylabel(r"$k P_{\ell}(k)\, [h^{2}\,{\rm Mpc}^{-2}] $", fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn, bbox_inches="tight")
        plt.show()
        return fig

    @set_plot_style
    def plot_quantile_power(
        self, ell: int = 0, save_fn: str | Path | None = None
    ) -> plt.Figure:
        """Plot the power spectrum for each quantile."""
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(len(self.quantiles)):
            k, poles = self._quantile_power[i](
                ell=(0, 2, 4), return_k=True, complex=False
            )
            ax.plot(k, k * poles[ell // 2], lw=2.0, label=rf"${{\rm Q}}_{i}$")
        ax.set_xlabel(r"$k\, [h\,{\rm Mpc}^{-1}]$", fontsize=15)
        ax.set_ylabel(r"$k P_{\ell}(k)\, [h^{2}\,{\rm Mpc}^{-2}] $", fontsize=15)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn, bbox_inches="tight")
        plt.show()
        return fig
