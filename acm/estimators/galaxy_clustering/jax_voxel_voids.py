import time
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Any, Union

import jax
import numpy as np
import jax.numpy as jnp
import numpy.typing as npt
import matplotlib
import matplotlib.pyplot as plt
from pycorr import TwoPointCorrelationFunction

from .base import BaseEstimator
from acm.utils.plotting import set_plot_style


jax.config.update('jax_enable_x64', True)


class JaxVoxelVoids(BaseEstimator):
    """
    GPU-capable voxel-void finder based on JAX arrays and in-memory processing.
    """
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the JAX voxel-void estimator.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded to :class:`BaseEstimator`.
            Typical options include data/random positions, mesh geometry,
            and backend selection.

        Notes
        -----
        This estimator currently supports only ``backend='jaxpower'``.
        """
        super().__init__(**kwargs)
        if self.backend.name != 'jaxpower':
            raise NotImplementedError(
                "JaxVoxelVoids currently supports only backend='jaxpower'."
            )

    def _prepare_void_inputs_from_backend(self) -> None:
        """Prepare masked overdensity inputs needed by void finding.

        This method reuses the density-contrast field produced by the backend
        (``backend.delta_mesh``) and derives the validity mask used during
        watershed descent. Invalid cells are set to ``+inf`` so they are never
        selected as downhill targets.

        Raises
        ------
        RuntimeError
            If density contrast has not been computed yet.
        """
        if not hasattr(self.backend, 'delta_mesh'):
            raise RuntimeError('Density contrast is not set. Call set_density_contrast() before find_voids().')

        delta_mesh = self.backend.delta_mesh
        delta_array = delta_mesh.value if hasattr(delta_mesh, 'value') else delta_mesh
        self.delta_mesh_array = jnp.asarray(delta_array)

        threshold_value = getattr(self.backend, 'randoms_threshold_value', 0.01)
        threshold_method = getattr(self.backend, 'randoms_threshold_method', 'noise')
        self.ran_min = threshold_value

        if self.has_randoms:
            randoms_real = self.backend.randoms_mesh.paint(
                resampler='cic', compensate=False, interlacing=False, halo_add=0, out='real'
            )
            randoms_value = jnp.asarray(randoms_real.value)
            threshold_randoms = self.backend._get_threshold_randoms(
                self.backend.randoms_mesh,
                threshold_value=threshold_value,
                threshold_method=threshold_method,
            )
            self.valid_mask = randoms_value > threshold_randoms
        else:
            self.valid_mask = jnp.ones_like(self.delta_mesh_array, dtype=bool)

        self.delta_mesh_array = jnp.where(self.valid_mask, self.delta_mesh_array, jnp.inf)

    def find_voids(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """Find voxel void centers and effective radii.

        Returns
        -------
        voids : ndarray of shape (N, 3)
            Cartesian coordinates of selected void centers.
        void_radii : ndarray of shape (N,)
            Effective spherical radii from basin volumes.

        Notes
        -----
        The watershed procedure itself is executed inside
        :meth:`_compute_roots` and post-processed in
        :meth:`_find_voids_in_memory`.
        """
        self.time = time.time()
        self._prepare_void_inputs_from_backend()
        self.voids, self.void_radii = self._find_voids_in_memory()
        self.logger.info(f"Found {len(self.voids)} voxel voids in {time.time() - self.time:.2f} seconds.")
        return self.voids, self.void_radii

    @staticmethod
    @partial(jax.jit, static_argnames=('nsteps',))
    def _compute_roots(
        rho: jnp.ndarray,
        valid_mask: jnp.ndarray,
        nsteps: int,
    ) -> jnp.ndarray:
        """Run the watershed descent and pointer-jumping root solve.

        Parameters
        ----------
        rho : jax.Array
            3D scalar field used for descent ordering (here, overdensity).
        valid_mask : jax.Array
            Boolean mask of voxels allowed to participate.
        nsteps : int
            Number of pointer-jumping iterations.

        Returns
        -------
        roots : jax.Array
            Flattened array mapping each voxel to its basin root voxel index
            (or ``-1`` for invalid voxels).

        Notes
        -----
        This is the core watershed stage:
        1) choose the lowest-density neighbor among 6-connectivity,
        2) repeatedly pointer-jump until each voxel reaches a local minimum.
        """
        nmesh = rho.shape
        nvox = int(np.prod(nmesh))
        idx = jnp.arange(nvox, dtype=jnp.int64).reshape(nmesh)

        rho_xp = jnp.roll(rho, -1, axis=0)
        rho_xm = jnp.roll(rho, 1, axis=0)
        rho_yp = jnp.roll(rho, -1, axis=1)
        rho_ym = jnp.roll(rho, 1, axis=1)
        rho_zp = jnp.roll(rho, -1, axis=2)
        rho_zm = jnp.roll(rho, 1, axis=2)

        idx_xp = jnp.roll(idx, -1, axis=0)
        idx_xm = jnp.roll(idx, 1, axis=0)
        idx_yp = jnp.roll(idx, -1, axis=1)
        idx_ym = jnp.roll(idx, 1, axis=1)
        idx_zp = jnp.roll(idx, -1, axis=2)
        idx_zm = jnp.roll(idx, 1, axis=2)

        candidates_rho = jnp.stack([rho, rho_xp, rho_xm, rho_yp, rho_ym, rho_zp, rho_zm], axis=0)
        candidates_idx = jnp.stack([idx, idx_xp, idx_xm, idx_yp, idx_ym, idx_zp, idx_zm], axis=0)

        min_choice = jnp.argmin(candidates_rho, axis=0)
        next_idx = jnp.take_along_axis(candidates_idx, min_choice[None, ...], axis=0)[0]
        next_idx = jnp.where(valid_mask, next_idx, -jnp.ones_like(next_idx)).reshape(-1)

        def jump_step(_: int, parents: jnp.ndarray) -> jnp.ndarray:
            safe = jnp.where(parents >= 0, parents, 0)
            return jnp.where(parents >= 0, parents[safe], -1)

        return jax.lax.fori_loop(0, nsteps, jump_step, next_idx)

    def _find_voids_in_memory(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """Convert watershed roots into a filtered void catalog.

        Returns
        -------
        void_positions : ndarray of shape (N, 3)
            Cartesian coordinates of selected basin minima.
        void_radii : ndarray of shape (N,)
            Effective radii derived from basin voxel counts.

        Notes
        -----
        This method performs the post-watershed steps: root counting,
        edge-root rejection, density-threshold cuts, zone membership assembly,
        and conversion to positions/radii.
        """
        self.logger.info('Finding voids with JAX in-memory watershed approximation.')
        nmesh = tuple(int(x) for x in self.meshsize)
        delta = jnp.asarray(self.delta_mesh_array).reshape(nmesh)
        valid_mask = jnp.asarray(self.valid_mask).reshape(nmesh)

        nvox = int(np.prod(nmesh))
        nsteps = int(np.ceil(np.log2(max(nvox, 2)))) + 1
        roots = self._compute_roots(delta, valid_mask, nsteps)

        flat_valid = np.asarray(valid_mask).reshape(-1)
        roots_np = np.asarray(roots)
        delta_np = np.asarray(delta).reshape(-1)

        valid_roots = roots_np[flat_valid]
        if valid_roots.size == 0:
            self.zones = []
            self.core_dens = np.array([], dtype=float)
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

        unique_roots, counts = np.unique(valid_roots, return_counts=True)
        core_delta = delta_np[unique_roots]

        mask_np = np.asarray(valid_mask, dtype=bool)
        nadj = (
            np.roll(mask_np, -1, axis=0).astype(np.int8)
            + np.roll(mask_np, 1, axis=0).astype(np.int8)
            + np.roll(mask_np, -1, axis=1).astype(np.int8)
            + np.roll(mask_np, 1, axis=1).astype(np.int8)
            + np.roll(mask_np, -1, axis=2).astype(np.int8)
            + np.roll(mask_np, 1, axis=2).astype(np.int8)
        )
        edge_voxels = mask_np & (nadj < 6)
        edge_roots = np.unique(roots_np[edge_voxels.reshape(-1)])
        edge_roots = edge_roots[edge_roots >= 0]

        self.min_dens_cut = 1.0
        keep = (core_delta < 0.0) & (~np.isin(unique_roots, edge_roots))
        selected_roots = unique_roots[keep]
        selected_counts = counts[keep]
        self.core_dens = core_delta[keep] + 1.0

        if selected_roots.size == 0:
            self.zones = []
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

        valid_voxel_ids = np.flatnonzero(flat_valid)
        sort_order = np.argsort(valid_roots, kind='mergesort')
        sorted_roots = valid_roots[sort_order]
        sorted_voxel_ids = valid_voxel_ids[sort_order]
        left = np.searchsorted(sorted_roots, selected_roots, side='left')
        right = np.searchsorted(sorted_roots, selected_roots, side='right')
        self.zones = [
            sorted_voxel_ids[lo:hi].astype(int).tolist()
            for lo, hi in zip(left, right)
        ]
        xpos, ypos, zpos = self.voxel_position(selected_roots.astype(np.int64))

        cell_vol = float(self.cellsize[0] * self.cellsize[1] * self.cellsize[2])
        vols = selected_counts.astype(float) * cell_vol
        rads = (3.0 * vols / (4.0 * np.pi)) ** (1.0 / 3.0)
        return np.c_[xpos, ypos, zpos], rads

    def voxel_position(self, voxel: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Map flattened voxel indices to Cartesian coordinates.

        Parameters
        ----------
        voxel : ndarray
            Flattened voxel indices.

        Returns
        -------
        xpos, ypos, zpos : ndarray
            Coordinates in the estimator box frame.
        """
        voxel = np.asarray(voxel, dtype=np.int64)
        nmesh = np.asarray(self.meshsize, dtype=np.int64)
        boxsize = np.asarray(self.boxsize, dtype=float)
        boxcenter = np.asarray(self.boxcenter, dtype=float)

        nxy = nmesh[1] * nmesh[2]
        xi = voxel // nxy
        rem = voxel % nxy
        yi = rem // nmesh[2]
        zi = rem % nmesh[2]

        xpos = xi * boxsize[0] / nmesh[0]
        ypos = yi * boxsize[1] / nmesh[1]
        zpos = zi * boxsize[2] / nmesh[2]

        if self.has_randoms:
            xpos += boxcenter[0] - boxsize[0] / 2.0
            ypos += boxcenter[1] - boxsize[1] / 2.0
            zpos += boxcenter[2] - boxsize[2] / 2.0
        
        offset = boxcenter - boxsize / 2.0
        xpos += offset[0]
        ypos += offset[1]
        zpos += offset[2]

        return xpos, ypos, zpos

    def void_data_correlation(self, data_positions: npt.NDArray, **kwargs: Any) -> TwoPointCorrelationFunction:
        """Compute the void-data two-point correlation function.

        Parameters
        ----------
        data_positions : ndarray
            Galaxy/data positions for cross-correlation with void centers.
        **kwargs : dict
            Additional options passed to :class:`pycorr.TwoPointCorrelationFunction`.

        Returns
        -------
        TwoPointCorrelationFunction
            Configured pycorr object containing measured correlation statistics.
        """
        if self.has_randoms:
            if 'randoms_positions' not in kwargs:
                raise ValueError('Randoms positions must be provided when working with a non-uniform geometry.')
            kwargs['randoms_positions1'] = kwargs['randoms_positions']
            kwargs['randoms_positions2'] = kwargs['randoms_positions']
            kwargs.pop('randoms_positions')
            if 'data_weights' in kwargs:
                kwargs['data_weights2'] = kwargs.pop('data_weights')
            if 'randoms_weights' in kwargs:
                kwargs['randoms_weights2'] = kwargs.pop('randoms_weights')
        else:
            if 'boxsize' not in kwargs:
                kwargs['boxsize'] = self.boxsize
        self._void_data_correlation = TwoPointCorrelationFunction(
            data_positions1=self.voids,
            data_positions2=data_positions,
            mode='smu',
            position_type='pos',
            **kwargs,
        )
        return self._void_data_correlation

    @set_plot_style
    def plot_void_size_distribution(self, save_fn: Optional[Union[str, Path]] = None) -> matplotlib.figure.Figure:
        """Plot the histogram of void effective radii.

        Parameters
        ----------
        save_fn : str or Path, optional
            Output filename for the figure.

        Returns
        -------
        matplotlib.figure.Figure
            The generated matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.hist(self.void_radii, bins=25, lw=2.0, alpha=0.5)
        ax.set_xlabel(r'$R_{\rm void}\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$N$', fontsize=15)
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig

    @set_plot_style
    def plot_void_data_correlation(
        self,
        ells: Tuple[int, ...] = (0,),
        save_fn: Optional[Union[str, Path]] = None,
    ) -> matplotlib.figure.Figure:
        """Plot multipoles of the void-data correlation function.

        Parameters
        ----------
        ells : tuple of int, default=(0,)
            Multipoles to display.
        save_fn : str or Path, optional
            Output filename for the figure.

        Returns
        -------
        matplotlib.figure.Figure
            The generated matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        s, multipoles = self._void_data_correlation(ells=(0, 2, 4), return_sep=True)
        for ell in ells:
            ax.plot(s, multipoles[ell // 2], lw=2.0, label=f'$\\ell = {ell}$')
        ax.set_xlabel(r'$s\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$\xi_\ell(s)$', fontsize=15)
        ax.legend(fontsize=15, loc='best', handlelength=1.0)
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig

    @set_plot_style
    def plot_slice(
        self,
        data_positions: Optional[npt.NDArray] = None,
        save_fn: Optional[Union[str, Path]] = None,
    ) -> matplotlib.figure.Figure:
        """Visualize a 2D projected slice of watershed zones.

        Parameters
        ----------
        data_positions : ndarray, optional
            Unused placeholder for API compatibility.
        save_fn : str or Path, optional
            Output filename for the figure.

        Returns
        -------
        matplotlib.figure.Figure
            The generated matplotlib figure.
        """
        _ = data_positions
        nmesh = tuple(int(x) for x in self.meshsize)
        boxsize = np.asarray(self.boxsize, dtype=float)
        zones_mesh = np.zeros(np.prod(nmesh), dtype=float)
        rng = np.random.default_rng(42)
        for zone in self.zones:
            zones_mesh[np.asarray(zone, dtype=int)] = rng.random()
        zones_mesh = np.ma.masked_where(zones_mesh == 0, zones_mesh).reshape(nmesh)
        zones_mesh = np.sum(zones_mesh, axis=2)
        fig, ax = plt.subplots()
        cmap = matplotlib.cm.tab20
        cmap.set_bad(color='white')
        ax.imshow(
            zones_mesh[:, :],
            origin='lower',
            cmap=cmap,
            extent=[0, boxsize[0], 0, boxsize[1]],
            interpolation='gaussian',
        )
        ax.set_xlabel(r'$x\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$y\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig
