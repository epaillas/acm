import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from pyrecon import RealMesh

from .base import BaseEstimator
from .src import minkowski as mk

logger = logging.getLogger(__name__)


class MinkowskiFunctionals(BaseEstimator):
    """Class to compute the Minkowski functionals."""

    def __init__(self, **kwargs) -> None:
        self.mask_mesh = RealMesh(**kwargs)
        logger.info("Initializing MinkowskiFunctionals.")
        super().__init__(**kwargs)

    def set_density_contrast(
        self,
        global_mean: float | None = None,
        smoothing_radius: float | None = None,
        check: bool = False,
        ran_min: float = 0.01,
        save_wisdom: bool = False,
        plan: str = "estimate",
    ) -> np.ndarray:
        """
        Set the density contrast.

        Parameters
        ----------
        smoothing_radius : float, optional
            Smoothing radius.
        check : bool, optional
            Check if there are enough randoms.
        ran_min : float, optional
            Minimum randoms.
        nquery_factor : int, optional
            Factor to multiply the number of data points to get the number of query points.
        plan : str, optional
            Plan for FFTW, by default "estimate".

        Returns
        -------
        delta_mesh : array_like
            Density contrast.
        """
        t0 = time.time()
        if smoothing_radius:
            self.data_mesh.smooth_gaussian(
                smoothing_radius, engine="fftw", save_wisdom=save_wisdom, plan=plan
            )
        if self.has_randoms:
            if check:
                mask_nonzero = self.randoms_mesh.value > 0.0
                nnonzero = mask_nonzero.sum()
                if nnonzero < 2:
                    raise ValueError("Very few randoms.")
            if smoothing_radius:
                self.randoms_mesh.smooth_gaussian(
                    smoothing_radius, engine="fftw", save_wisdom=save_wisdom, plan=plan
                )
            sum_data, sum_randoms = (
                np.sum(self.data_mesh.value),
                np.sum(self.randoms_mesh.value),
            )
            alpha = sum_data * 1.0 / sum_randoms
            self.delta_mesh = self.data_mesh - alpha * self.randoms_mesh
            self.ran_min = ran_min * sum_randoms / self._size_randoms
            mask = self.randoms_mesh > self.ran_min
            self.delta_mesh[mask] /= alpha * self.randoms_mesh[mask]
            self.delta_mesh[~mask] = -3.0
        elif global_mean:
            self.delta_mesh = self.data_mesh / global_mean - 1.0
        else:
            self.delta_mesh = self.data_mesh / np.mean(self.data_mesh) - 1.0
        logger.info(f"Set density contrast in {time.time() - t0:.2f} seconds.")
        return self.delta_mesh

    def get_rho(
        self,
        rho_thres: float,
        smoothing_radius: float | None = None,
        save_wisdom: bool = False,
        plan: str = "estimate",
    ) -> np.ndarray:
        """
        Get rho.

        Parameters
        ----------
        rho_thres : float
            Threshold for rho.
        smoothing_radius : float, optional
            Smoothing radius.
        save_wisdom : bool, optional
            Whether to save FFTW wisdom, by default False.
        plan : str, optional
            Plan for FFTW, by default "estimate".

        Returns
        -------
        rho_mesh : array_like
            Density.
        """
        t0 = time.time()
        self.mask_mesh.value = self.data_mesh > 0
        if smoothing_radius:
            self.data_mesh.smooth_gaussian(
                smoothing_radius, engine="fftw", save_wisdom=save_wisdom, plan=plan
            )
            self.mask_mesh.smooth_gaussian(
                smoothing_radius, engine="fftw", save_wisdom=save_wisdom, plan=plan
            )
        mask = self.mask_mesh > rho_thres
        self.rho_mesh = 1.0 * self.data_mesh
        self.rho_mesh[~mask] = -3.0
        self.rho_used = self.rho_mesh[mask]
        logger.info(f"Get rho in {time.time() - t0:.2f} seconds.")
        return self.rho_mesh

    def run(
        self,
        thres_mask: float = -2,
        thresholds: np.ndarray | None = None,
        nthreads: int = 1,
    ) -> np.ndarray:
        """
        Run the Minkowski functionals.

        Parameters
        ----------
        thres_mask : float, optional
            Threshold for masking, by default -2.
        thresholds : array_like, optional
            Thresholds to evaluate Minkowski functionals at. If set to None, defaults to np.linspace(-1, 5, num=201, dtype=np.float32).
            Defaults to None.
        nthreads : int, optional
            Number of threads to use, by default 1.

        Returns
        -------
        MFs3D : array_like
            3D Minkowski Functionals V_0, V_1, V_2, V_3.
        """
        t0 = time.time()
        self.thres_mask = thres_mask
        self.thresholds = thresholds or np.linspace(-1, 5, num=201, dtype=np.float32)
        if nthreads == 1:
            mf = mk.MFs(
                self.delta_mesh,
                self.data_mesh.cellsize[0],
                self.thres_mask,
                self.thresholds,
            )
        else:
            mf = mk.MFs_parallel(
                self.delta_mesh,
                self.data_mesh.cellsize[0],
                self.thres_mask,
                self.thresholds,
                nthreads,
            )
        self.MFs = mf.MFs3D
        logger.info(f"Minkowski functionals elapsed in {time.time() - t0:.2f} seconds.")
        return self.MFs

    def measure_v123(self, thres_mask: float = -2, thres_bins: int = 200) -> np.ndarray:
        """
        Only measure the 1,2,3th Minkowski functionals.

        Parameters
        ----------
        thres_mask : float, optional
            Threshold for masking, by default -2.
        thres_bins : int, optional
            Number of bins for thresholds, by default 200.

        Returns
        -------
        MFs3D : array_like
            3D Minkowski Functionals V_1, V_2, V_3.
        """
        t0 = time.time()
        self.thres_mask = thres_mask
        self.thresholds = np.quantile(
            self.rho_used, np.linspace(0, 1, num=thres_bins + 1)
        )
        mf = mk.MFs(
            self.rho_mesh,
            self.data_mesh.cellsize[0],
            self.thres_mask,
            np.float32(self.thresholds),
        )
        self.MFs = mf.MFs3D
        logger.info(f"Minkowski functionals elapsed in {time.time() - t0:.2f} seconds.")
        return self.MFs

    def plot_mfs(
        self, x: list = [], label: str = "MFs", mf_cons: list = [1, 10**3, 10**5, 10**7]
    ) -> plt.Figure:
        """Plot the Minkowski functionals."""
        fig = plt.figure(constrained_layout=False, figsize=[10, 10])
        spec = fig.add_gridspec(ncols=2, nrows=2, hspace=0.2, wspace=0.3)
        if len(x) == 0:
            x = list(self.thresholds)

        ylabels = [
            r"$V_{0}$",
            r"$V_{1}[10^{- " + str(int(np.log10(mf_cons[1]))) + "}hMpc^{-1}]$",
            r"$V_{2}[10^{- " + str(int(np.log10(mf_cons[2]))) + "}(hMpc^{-1})^2]$",
            r"$V_{3}[10^{- " + str(int(np.log10(mf_cons[3]))) + "}(hMpc^{-1})^3]$",
        ]

        for i in range(4):
            ii = i // 2
            jj = i % 2
            ax = fig.add_subplot(spec[ii, jj])
            ax.plot(x, self.MFs[:, i] * mf_cons[i], color="blue", label=label)
            if i == 0:
                ax.legend()

            ax.set_xlabel(r"$\delta$")
            ax.set_ylabel(ylabels[i])
            ax.axhline(color="black")
            ax.set_xlim(np.min(x), np.max(x))

        return fig

    def plot_v123(
        self, x: list = [], label: str = "MFs", mf_cons: list = [10**2, 10**3, 10**4]
    ) -> plt.Figure:
        """Plot the 1,2,3th Minkowski functionals."""
        fig = plt.figure(constrained_layout=False, figsize=[15, 4])
        spec = fig.add_gridspec(ncols=3, nrows=1, hspace=0.2, wspace=0.3)
        if len(x) == 0:
            x = list(self.thresholds)

        ylabels = [
            r"$V_{1}[10^{- " + str(int(np.log10(mf_cons[0]))) + "}hMpc^{-1}]$",
            r"$V_{2}[10^{- " + str(int(np.log10(mf_cons[1]))) + "}(hMpc^{-1})^2]$",
            r"$V_{3}[10^{- " + str(int(np.log10(mf_cons[2]))) + "}(hMpc^{-1})^3]$",
        ]

        for i in range(3):
            ax = fig.add_subplot(spec[0, i])
            ax.plot(x, self.MFs[:, i + 1] * mf_cons[i], color="blue", label=label)
            if i == 0:
                ax.legend()

            ax.set_xlabel(r"$f_v$")
            ax.set_ylabel(ylabels[i])
            ax.axhline(color="black")
            ax.set_xlim(np.min(x), np.max(x))

        return fig
