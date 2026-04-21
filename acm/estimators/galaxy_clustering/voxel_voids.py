import logging
import random
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import cm
from matplotlib.figure import Figure
from pycorr import TwoPointCorrelationFunction

from acm.utils.plotting import set_plot_style

from .base import BaseEstimator
from .src import fastmodules

logger = logging.getLogger(__name__)


class VoxelVoids(BaseEstimator):
    """Class to calculate voxel voids, as in https://github.com/seshnadathur/Revolver."""

    def __init__(self, temp_dir: str | Path, **kwargs: Any) -> None:
        """
        Initialize VoxelVoids estimator.

        Parameters
        ----------
        temp_dir : str or Path
            Directory for temporary files generated during void finding.
        **kwargs : dict
            Additional keyword arguments passed to BaseEstimator.
            Common options include:
            - backend : str, default='jaxpower'
                Backend to use ('jaxpower', 'pypower', or 'pyrecon').
            - data_positions : array_like, shape (N, 3)
                Positions of data galaxies.
            - data_weights : array_like, shape (N,), optional
                Weights for data galaxies.
            - randoms_positions : array_like, shape (M, 3), optional
                Positions of random catalog.
            - randoms_weights : array_like, shape (M,), optional
                Weights for randoms.
            - boxsize : float or array_like
                Size of the simulation box.
            - boxcenter : float or array_like
                Center of the box.
            - meshsize : int or array_like
                Number of mesh cells per dimension.
        """
        super().__init__(**kwargs)
        self.handle = Path(temp_dir) / str(uuid.uuid4())

    def set_density_contrast(
        self,
        smoothing_radius: float | None = None,
        ran_min: float = 0.01,
    ) -> np.ndarray:
        """
        Set the density contrast.

        Parameters
        ----------
        smoothing_radius : float, optional
            Smoothing radius.
        ran_min : float, optional
            Minimum randoms.

        Returns
        -------
        delta_mesh : array_like
            Density contrast.
        """
        t0 = time.time()
        # Use backend's set_density_contrast method
        delta_mesh = self.backend.set_density_contrast(
            smoothing_radius=smoothing_radius,
        )

        # Extract numpy array from mesh objects if needed
        delta_array = delta_mesh.value if hasattr(delta_mesh, "value") else delta_mesh

        # For voxel voids, we also need rho_mesh
        if self.has_randoms:
            if hasattr(self, "_PyreconBackend") and isinstance(
                self.backend, self._PyreconBackend
            ):
                # Access meshes from backend
                data_mesh = self.backend.data_mesh
                randoms_mesh = self.backend.randoms_mesh

                sum_data = np.sum(data_mesh.value)
                sum_randoms = np.sum(randoms_mesh.value)
                alpha = sum_data * 1.0 / sum_randoms
                self.ran_min = ran_min * sum_randoms / self.backend._size_randoms
                mask = randoms_mesh.value > self.ran_min
                # Create rho_mesh
                self.rho_mesh = data_mesh.value.copy()
                self.rho_mesh[mask] /= alpha * randoms_mesh.value[mask]
                self.rho_mesh[~mask] = 0.9e30
            else:
                # For other backends, approximate rho_mesh from delta_mesh
                self.rho_mesh = np.asarray(delta_array) + 1.0
                mask = self.rho_mesh < ran_min
                self.rho_mesh[mask] = 0.9e30
        else:
            # For periodic boxes without randoms, use delta_mesh directly
            self.rho_mesh = np.asarray(delta_array) + 1.0

        logger.info(f"Set density contrast in {time.time() - t0:.2f} seconds.")
        return delta_mesh

    def find_voids(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Run the voxel voids algorithm to identify cosmic voids.

        This method implements the ZOBOV (ZOnes Bordering On Voidness) algorithm
        to find voids in the density field. It first calls _find_voids to run the
        external void-finding executable, then post-processes the results to remove
        edge voids and voids in masked regions.

        Returns
        -------
        voids : array_like, shape (N, 3)
            Positions of void centers in Cartesian coordinates.
        void_radii : array_like, shape (N,)
            Effective radii of voids, computed from void volumes as
            R = (3V / 4π)^(1/3).

        Notes
        -----
        The density contrast must be set using set_density_contrast before
        calling this method.
        """
        self.time = time.time()
        self._find_voids()
        self.voids, self.void_radii = self._postprocess_voids()
        nvoids = len(self.voids)
        logger.info(
            f"Found {nvoids} voxel voids in {time.time() - self.time:.2f} seconds."
        )
        return self.voids, self.void_radii

    def _find_voids(self) -> None:
        """
        Find voids in the overdensity field using the ZOBOV algorithm.

        Writes the density mesh to a temporary file and calls the external
        jozov-grid executable to identify void regions. The executable generates
        .void, .txt, and .zone files containing void information.

        Notes
        -----
        This is an internal method called by find_voids. The results are
        post-processed by _postprocess_voids before being returned to the user.
        """
        logger.info("Finding voids.")
        nmesh = self.meshsize
        rho_mesh_flat = np.array(self.rho_mesh, dtype=np.float32)
        fn = f"{self.handle}_rho_mesh_n{nmesh[0]}{nmesh[1]}{nmesh[2]}d.dat"
        with Path(fn).open("w") as F:
            rho_mesh_flat.tofile(F, format="%f")
        bin_path = Path(__file__).parent / "src" / "jozov-grid.exe"
        cmd = [
            bin_path,
            "v",
            f"{self.handle}_rho_mesh_n{nmesh[0]}{nmesh[1]}{nmesh[2]}d.dat",
            self.handle,
            str(nmesh[0]),
            str(nmesh[1]),
            str(nmesh[2]),
        ]
        subprocess.call(cmd)  # noqa: S603

    def _postprocess_voids(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Post-process voids to remove edge voids and voids in masked voxels.

        Applies quality cuts to the raw void catalog to remove:
        1. Voids that don't meet minimum density criteria
        2. Edge voids (touching the boundary)
        3. Voids lying in masked (empty) voxels

        Returns
        -------
        void_positions : array_like, shape (N, 3)
            Cartesian coordinates of void centers at minimum density locations.
        void_radii : array_like, shape (N,)
            Effective radii computed from void volumes.

        Notes
        -----
        This method reads the .txt and .zone files generated by _find_voids,
        applies cuts using the fastmodules.voxelvoid_cuts function, and cleans
        up temporary files.
        """
        logger.info("Post-processing voids.")
        nmesh = self.meshsize
        cellsize = self.cellsize[0]
        mask_cut = np.zeros(nmesh[0] * nmesh[1] * nmesh[2], dtype=np.intc)
        if self.has_randoms:
            # identify "empty" cells for later cuts on void catalogue
            mask_cut = np.zeros(nmesh[0] * nmesh[1] * nmesh[2], dtype=np.intc)
            randoms_mesh_value = (
                self.backend.randoms_mesh.value
                if hasattr(self.backend.randoms_mesh, "value")
                else self.backend.randoms_mesh
            )
            fastmodules.survey_mask(mask_cut, randoms_mesh_value, self.ran_min)
        self.mask_cut = mask_cut
        self.min_dens_cut = 1.0
        rawdata = np.loadtxt(f"{self.handle}.txt", skiprows=2)
        # remove voids that: a) don't meet minimum density cut, b) are edge voids, or c) lie in a masked voxel
        select = np.zeros(rawdata.shape[0], dtype=np.intc)
        fastmodules.voxelvoid_cuts(select, self.mask_cut, rawdata, self.min_dens_cut)
        select = np.asarray(select, dtype=bool)
        rawdata = rawdata[select]
        # void minimum density centre locations
        logger.info("Calculating void positions.")
        xpos, ypos, zpos = self.voxel_position(rawdata[:, 2])
        offset = self.boxcenter - self.boxsize / 2.0
        xpos += offset[0]
        ypos += offset[1]
        zpos += offset[2]
        self.core_dens = rawdata[:, 3]
        # void effective radii
        logger.info("Calculating void radii.")
        vols = rawdata[:, 5] * cellsize**3.0
        rads = (3.0 * vols / (4.0 * np.pi)) ** (1.0 / 3)
        self.zones = []
        with Path(f"{self.handle}.zone").open("r") as f:
            for line in f:
                self.zones.append([int(i) for i in line.split()])
        self.zones = [zone for i, zone in enumerate(self.zones) if select[i]]
        Path(f"{self.handle}.void").unlink(missing_ok=True)
        Path(f"{self.handle}.txt").unlink(missing_ok=True)
        Path(f"{self.handle}.zone").unlink(missing_ok=True)
        Path(f"{self.handle}_rho_mesh_n{nmesh[0]}{nmesh[1]}{nmesh[2]}d.dat").unlink(missing_ok=True)
        return np.c_[xpos, ypos, zpos], rads

    def voxel_position(
        self,
        voxel: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Calculate the Cartesian position of voxels in the mesh.

        Converts voxel indices (flattened 1D array index) to Cartesian
        coordinates (x, y, z) in physical units, accounting for box size,
        box center, and mesh resolution.

        Parameters
        ----------
        voxel : array_like
            Flattened voxel indices to convert to positions.

        Returns
        -------
        xpos : array_like
            x-coordinates in Mpc/h.
        ypos : array_like
            y-coordinates in Mpc/h.
        zpos : array_like
            z-coordinates in Mpc/h.

        Notes
        -----
        For surveys with randoms, positions are centered relative to boxcenter.
        For periodic boxes, positions start at the box origin.
        """
        voxel = voxel.astype("i")
        boxsize = self.boxsize
        boxcenter = self.boxcenter
        nmesh = self.meshsize
        all_vox = np.arange(0, nmesh[0] * nmesh[1] * nmesh[2], dtype=int)
        vind = np.zeros((np.copy(all_vox).shape[0]), dtype=int)
        xpos = np.zeros(vind.shape[0], dtype=float)
        ypos = np.zeros(vind.shape[0], dtype=float)
        zpos = np.zeros(vind.shape[0], dtype=float)
        all_vox = np.arange(0, nmesh[0] * nmesh[1] * nmesh[2], dtype=int)
        xi = np.zeros(nmesh[0] * nmesh[1] * nmesh[2])
        yi = np.zeros(nmesh[1] * nmesh[2])
        zi = np.arange(nmesh[2])
        if self.has_randoms:
            for i in range(nmesh[1]):
                yi[i * (nmesh[2]) : (i + 1) * (nmesh[2])] = i
            for i in range(nmesh[0]):
                xi[i * (nmesh[1] * nmesh[2]) : (i + 1) * (nmesh[1] * nmesh[2])] = i
            xpos = xi * boxsize[0] / nmesh[0]
            ypos = np.tile(yi, nmesh[0]) * boxsize[1] / nmesh[1]
            zpos = np.tile(zi, nmesh[1] * nmesh[0]) * boxsize[2] / nmesh[2]
            xpos += boxcenter[0] - boxsize[0] / 2.0
            ypos += boxcenter[1] - boxsize[1] / 2.0
            zpos += boxcenter[2] - boxsize[2] / 2.0
            return xpos[voxel], ypos[voxel], zpos[voxel]
        for i in range(nmesh[1]):
            yi[i * (nmesh[2]) : (i + 1) * (nmesh[2])] = i
        for i in range(nmesh[0]):
            xi[i * (nmesh[1] * nmesh[2]) : (i + 1) * (nmesh[1] * nmesh[2])] = i
        xpos = xi * boxsize[0] / nmesh[0]
        ypos = np.tile(yi, nmesh[0]) * boxsize[1] / nmesh[1]
        zpos = np.tile(zi, nmesh[1] * nmesh[0]) * boxsize[2] / nmesh[2]
        return xpos[voxel], ypos[voxel], zpos[voxel]

    def void_data_correlation(
        self,
        data_positions: npt.NDArray,
        **kwargs,
    ) -> TwoPointCorrelationFunction:
        """
        Compute the cross-correlation function between the voids and the data.

        Parameters
        ----------
        data_positions : array_like
            Positions of the data.
        kwargs : dict
            Additional arguments for pycorr.TwoPointCorrelationFunction.

        Returns
        -------
        s : array_like
            Pair separations.
        void_data_ccf : array_like
            Cross-correlation function between voids and data.
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
                kwargs["data_weights2"] = kwargs.pop("data_weights")
            if "randoms_weights" in kwargs:
                kwargs["randoms_weights2"] = kwargs.pop("randoms_weights")
        elif "boxsize" not in kwargs:
            kwargs["boxsize"] = self.boxsize
        self._void_data_correlation = TwoPointCorrelationFunction(
            data_positions1=self.voids,
            data_positions2=data_positions,
            mode="smu",
            position_type="pos",
            **kwargs,
        )
        return self._void_data_correlation

    @set_plot_style
    def plot_void_size_distribution(self, save_fn: str | Path | None = None) -> Figure:
        """
        Plot the void size distribution as a histogram.

        Parameters
        ----------
        save_fn : str or Path, optional
            If provided, save the figure to this file path.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.hist(self.void_radii, bins=25, lw=2.0, alpha=0.5)
        ax.set_xlabel(r"$R_{\rm void}\, [h^{-1}{\rm Mpc}]$", fontsize=15)
        ax.set_ylabel(r"$N$", fontsize=15)
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn, bbox_inches="tight")
        plt.show()
        return fig

    @set_plot_style
    def plot_void_data_correlation(
        self, ells: tuple[int, ...] = (0,), save_fn: str | Path | None = None
    ) -> Figure:
        """
        Plot the void-data cross-correlation multipoles.

        Parameters
        ----------
        ells : tuple of int, default=(0,)
            Multipole moments to plot (e.g., 0 for monopole, 2 for quadrupole).
        save_fn : str or Path, optional
            If provided, save the figure to this file path.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object.

        Notes
        -----
        The void_data_correlation method must be called before plotting.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        s, multipoles = self._void_data_correlation(ells=(0, 2, 4), return_sep=True)
        for ell in ells:
            ax.plot(s, multipoles[ell // 2], lw=2.0, label=f"$\\ell = {ell}$")
        ax.set_xlabel(r"$s\, [h^{-1}{\rm Mpc}]$", fontsize=15)
        ax.set_ylabel(r"$\xi_\ell(s)$", fontsize=15)
        ax.legend(fontsize=15, loc="best", handlelength=1.0)
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn, bbox_inches="tight")
        plt.show()
        return fig

    @set_plot_style
    def plot_slice(
        self,
        # data_positions: npt.NDArray | None = None,
        save_fn: str | Path | None = None,
    ) -> Figure:
        """
        Plot a 2D slice of the density field showing void zones.

        Creates a visualization of void zones projected onto a 2D plane by
        summing along the z-axis. Each void zone is colored randomly for
        visual distinction.

        Parameters
        ----------
        data_positions : array_like, optional
            Data positions to overplot (not currently implemented).
        save_fn : str or Path, optional
            If provided, save the figure to this file path.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object.

        Notes
        -----
        The find_voids method must be called before plotting to populate
        the zones attribute.
        """
        nmesh = self.meshsize
        boxsize = self.boxsize
        # boxcenter = self.boxcenter
        zones_mesh = np.zeros(nmesh).flatten()
        for _, zone in enumerate(self.zones):
            zones_mesh[zone] = random.random()
        zones_mesh = np.ma.masked_where(zones_mesh == 0, zones_mesh)
        delta_mesh = (
            self.backend.delta_mesh.value
            if hasattr(self.backend.delta_mesh, "value")
            else self.backend.delta_mesh
        )
        zones_mesh = zones_mesh.reshape(delta_mesh.shape)
        zones_mesh = np.sum(zones_mesh, axis=2)
        fig, ax = plt.subplots()
        cmap = cm.get_cmap("tab20")
        cmap.set_bad(color="white")
        ax.imshow(
            zones_mesh[:, :],
            origin="lower",
            cmap=cmap,
            extent=(0, boxsize[0], 0, boxsize[1]),
            interpolation="gaussian",
        )
        # ax.set_xlim(0, 1000)
        # ax.set_ylim(0, 1000)
        ax.set_xlabel(r"$x\, [h^{-1}{\rm Mpc}]$", fontsize=15)
        ax.set_ylabel(r"$y\, [h^{-1}{\rm Mpc}]$", fontsize=15)
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn, bbox_inches="tight")
        plt.show()
        return fig
