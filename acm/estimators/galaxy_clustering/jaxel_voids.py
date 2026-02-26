import time
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Any, Union

import jax
import numpy as np
import jax.numpy as jnp
import numpy.typing as npt
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import xarray as xr
from lsstypes import ObservableLeaf
from lsstypes.external import from_pycorr
from pycorr import TwoPointCorrelationFunction

from .base import BaseEstimator
from acm.utils.plotting import set_plot_style


jax.config.update('jax_enable_x64', True)


class JaxelVoids(BaseEstimator):
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
                "JaxelVoids currently supports only backend='jaxpower'."
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

    def find_voids(
        self,
        save_fn: Optional[Union[str, Path]] = None,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Find voxel void centers and effective radii.

        Parameters
        ----------
        save_fn : str or Path, optional
            If provided, save the resulting void catalog to disk using
            :meth:`save_catalog`.

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
        self.logger.info(f"Found {len(self.voids)} voids in {time.time() - self.time:.2f} s.")
        self.logger.info(f"Mean void radius: {np.mean(self.void_radii):.2f} Mpc/h.")
        if save_fn is not None:
            self.save(save_fn, type='catalog')
        return self.voids, self.void_radii

    def save_catalog(self, filename: Union[str, Path], attrs: Optional[dict] = None) -> None:
        """Save the current void catalog to disk.

        Format is automatically determined from file extension:
        - .hdf5, .h5 -> lsstypes (ObservableLeaf in HDF5 format)
        - .nc -> xarray (Dataset in NetCDF format)
        - .zarr -> xarray (Dataset in Zarr format)
        - .npy -> numpy (dictionary with core arrays)

        Parameters
        ----------
        filename : str or Path
            Output filename.

        Raises
        ------
        ValueError
            If no void catalog is available or extension is unsupported.
        """
        if not hasattr(self, 'voids') or not hasattr(self, 'void_radii'):
            raise ValueError('No void catalog to save. Run find_voids() first.')

        path = Path(filename)
        self.logger.info(f'Saving void catalog to {path}')

        voids = np.asarray(self.voids, dtype=float)
        void_radii = np.asarray(self.void_radii, dtype=float)
        core_dens = np.asarray(getattr(self, 'core_dens', np.full(len(void_radii), np.nan)), dtype=float)
        n_voids = int(void_radii.shape[0])
        zone_sizes = np.asarray([len(zone) for zone in getattr(self, 'zones', [])], dtype=int)

        default_attrs = dict(
            estimator='JaxelVoids',
            backend=self.backend.name,
            ran_min=float(getattr(self, 'ran_min', np.nan)),
            min_dens_cut=float(getattr(self, 'min_dens_cut', np.nan)),
            boxsize=np.asarray(self.boxsize, dtype=float).tolist(),
            boxcenter=np.asarray(self.boxcenter, dtype=float).tolist(),
            meshsize=np.asarray(self.meshsize, dtype=int).tolist(),
            cellsize=np.asarray(self.cellsize, dtype=float).tolist(),
            has_randoms=bool(self.has_randoms),
            n_voids=n_voids,
        )
        if attrs is not None:
            default_attrs.update(attrs)
        attrs = default_attrs

        if path.suffix in ['.hdf5', '.h5']:
            leaf = ObservableLeaf(
                voids=voids,
                void_radii=void_radii,
                core_dens=core_dens,
                zone_size=zone_sizes,
                void=np.arange(n_voids, dtype=int),
                xyz=np.arange(3, dtype=int),
                coords=['void', 'xyz'],
                attrs=attrs,
            )
            leaf.write(str(path))

        elif path.suffix in ['.nc', '.zarr']:
            dataset = xr.Dataset(
                data_vars={
                    'voids': (('void', 'xyz'), voids),
                    'void_radii': (('void',), void_radii),
                    'core_dens': (('void',), core_dens),
                    'zone_size': (('void',), zone_sizes),
                },
                coords={
                    'void': np.arange(n_voids, dtype=int),
                    'xyz': np.arange(3, dtype=int),
                },
                attrs=attrs,
            )
            if path.suffix == '.nc':
                dataset.to_netcdf(str(path))
            else:
                dataset.to_zarr(str(path), mode='w')

        elif path.suffix == '.npy':
            np.save(
                str(path),
                {
                    'voids': voids,
                    'void_radii': void_radii,
                    'core_dens': core_dens,
                    'zone_size': zone_sizes,
                    'attrs': attrs,
                },
                allow_pickle=True,
            )

        else:
            raise ValueError(
                f"Unrecognized file extension '{path.suffix}' for file: {path}. "
                f"Supported extensions: .hdf5, .h5 (lsstypes), .nc (xarray NetCDF), "
                f".zarr (xarray Zarr), .npy (numpy)."
            )

    def save_correlations(
        self,
        correlation: TwoPointCorrelationFunction,
        filename: Union[str, Path],
        attrs: Optional[dict] = None,
    ) -> None:
        """Save a void-data correlation measurement to disk.

        Parameters
        ----------
        correlation : TwoPointCorrelationFunction
            Correlation object to save.
        filename : str or Path
            Output filename.
        attrs : dict, optional
            Additional metadata to attach when supported by the output format.
        """
        path = Path(filename)
        self.logger.info(f'Saving void-data correlation to {path}')

        base_attrs = {
            'estimator': 'JaxelVoids',
            'backend': self.backend.name,
            'boxsize': np.asarray(self.boxsize, dtype=float).tolist(),
            'meshsize': np.asarray(self.meshsize, dtype=int).tolist(),
            'has_randoms': bool(self.has_randoms),
        }
        if attrs is not None:
            base_attrs.update(attrs)

        if path.suffix in ['.hdf5', '.h5']:
            corr_leaf = from_pycorr(correlation)
            if hasattr(corr_leaf, 'attrs') and isinstance(corr_leaf.attrs, dict):
                corr_leaf.attrs.update(base_attrs)
            corr_leaf.write(path)
        elif path.suffix == '.npy':
            np.save(path, correlation)
        else:
            raise ValueError(
                f"Unrecognized file extension '{path.suffix}' for file: {path}. "
                "Supported extensions are: .hdf5, .h5, .npy"
            )

    def save(
        self,
        filename: Union[str, Path],
        data: Optional[Any] = None,
        type: str = 'catalog',
        attrs: Optional[dict] = None,
    ) -> None:
        """Dispatch saving to catalog or correlation serializers.

        Parameters
        ----------
        filename : str or Path
            Output filename.
        data : Any, optional
            Data payload to save when ``type='correlation'``. If omitted,
            ``self._void_data_correlation`` is used.
        type : {'catalog', 'correlation'}, default='catalog'
            Data product to serialize.
        attrs : dict, optional
            Extra metadata passed to the specific save routine.
        """
        if type == 'catalog':
            self.save_catalog(filename, attrs=attrs)
        elif type == 'correlation':
            corr = data if data is not None else getattr(self, '_void_data_correlation', None)
            if corr is None:
                raise ValueError('No void-data correlation to save. Run void_data_correlation() first.')
            self.save_correlations(corr, filename, attrs=attrs)
        else:
            raise ValueError(f"Unknown type '{type}'. Available types: 'catalog', 'correlation'.")

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

    def void_data_correlation(
        self,
        data_positions: npt.NDArray,
        save_fn: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> TwoPointCorrelationFunction:
        """Compute the void-data two-point correlation function.

        Parameters
        ----------
        data_positions : ndarray
            Galaxy/data positions for cross-correlation with void centers.
        save_fn : str or Path, optional
            If provided, save the correlation result to disk using
            :meth:`save_correlations`.
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
        if save_fn is not None:
            self.save(save_fn, data=self._void_data_correlation, type='correlation')
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
        return fig

    @set_plot_style
    def plot_slice(
        self,
        save_fn: Optional[Union[str, Path]] = None,
    ) -> matplotlib.figure.Figure:
        """Visualize a 2D projected slice of watershed zones.

        Parameters
        ----------
        save_fn : str or Path, optional
            Output filename for the figure.

        Returns
        -------
        matplotlib.figure.Figure
            The generated matplotlib figure.
        """
        nmesh = tuple(int(x) for x in self.meshsize)
        boxsize = np.asarray(self.boxsize, dtype=float)
        zones_mesh = np.zeros(np.prod(nmesh), dtype=float)
        rng = np.random.default_rng(42)
        for zone in self.zones:
            zones_mesh[np.asarray(zone, dtype=int)] = rng.random()
        zones_mesh = np.ma.masked_where(zones_mesh == 0, zones_mesh).reshape(nmesh)
        # zones_mesh = np.sum(zones_mesh, axis=2)
        zones_mesh = zones_mesh[:, :, 0]  # Take a thin slice in z
        # print(zones_mesh.shape)
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
        ax.set_xlim(0, 250)
        ax.set_ylim(0, 250)
        ax.set_xlabel(r'$x\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$y\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn, bbox_inches='tight')
        return fig

    @set_plot_style
    def gif_void_slice(
        self,
        save_fn: Optional[Union[str, Path]] = None,
        interval: int = 120,
    ) -> matplotlib.figure.Figure:
        """Create a GIF scanning zone-map slices along the z-axis.

        This method uses the same zone-map construction and styling as
        :meth:`plot_slice`, but animates consecutive slices
        ``zones_mesh[:, :, z]`` for ``z = 0, 1, ..., N_z - 1``.

        Parameters
        ----------
        save_fn : str or Path, optional
            Output GIF filename. If not provided, defaults to
            ``'void_slice.gif'`` in the current working directory.
        interval : int, default=120
            Delay between frames in milliseconds.

        Returns
        -------
        matplotlib.figure.Figure
            The generated matplotlib figure used for the animation.
        """
        nmesh = tuple(int(x) for x in self.meshsize)
        boxsize = np.asarray(self.boxsize, dtype=float)

        zones_mesh = np.zeros(np.prod(nmesh), dtype=float)
        rng = np.random.default_rng(42)
        for zone in self.zones:
            zones_mesh[np.asarray(zone, dtype=int)] = rng.random()
        zones_mesh = np.ma.masked_where(zones_mesh == 0, zones_mesh).reshape(nmesh)

        fig, ax = plt.subplots()
        cmap = matplotlib.cm.tab20
        cmap.set_bad(color='white')

        image = ax.imshow(
            zones_mesh[:, :, 0],
            origin='lower',
            cmap=cmap,
            extent=[0, boxsize[0], 0, boxsize[1]],
            interpolation='gaussian',
            animated=True,
        )
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.set_xlabel(r'$x\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$y\, [h^{-1}{\rm Mpc}]$', fontsize=15)

        title = ax.set_title('z-slice 0', fontsize=13)

        def update(zindex: int):
            image.set_data(zones_mesh[:, :, zindex])
            title.set_text(f'z-slice {zindex}')
            return image, title

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=nmesh[2],
            interval=interval,
            blit=True,
        )

        out = Path(save_fn) if save_fn else Path('void_slice.gif')
        anim.save(out, writer='pillow', dpi=120)

        plt.tight_layout()
        return fig

    @set_plot_style
    def gif_voids_3d(
        self,
        zone_id: Optional[int] = None,
        save_fn: Optional[Union[str, Path]] = None,
        elev: float = 20.0,
        n_turns: int = 3,
        n_frames: int = 180,
        interval: int = 300,
        dpi: int = 120,
        padding_cells: int = 2,
        show_axes: bool = False,
    ) -> matplotlib.figure.Figure:
        """Create a rotating 3D GIF focused on a single voxel-void zone.

        Parameters
        ----------
        zone_id : int, optional
            Index of the zone in ``self.zones`` to render. If None, the
            largest non-edge zone is selected automatically. If provided but
            it touches the simulation-box edges, the method falls back to the
            largest non-edge zone.
        save_fn : str or Path, optional
            Output GIF filename. If not provided, defaults to
            ``'void_3d.gif'``.
        elev : float, default=20.0
            Elevation angle (degrees) of the camera.
        n_turns : int, default=3
            Number of full azimuth rotations in the GIF.
        n_frames : int, default=180
            Number of animation frames.
        interval : int, default=300
            Delay between frames in milliseconds.
        dpi : int, default=120
            Output GIF resolution.
        padding_cells : int, default=2
            Number of mesh cells added around the void bounding box for zoom.
        show_axes : bool, default=False
            If True, display 3D axis spines with tick labels in physical
            coordinates ($h^{-1}\,\mathrm{Mpc}$). If False, hide all axes.

        Returns
        -------
        matplotlib.figure.Figure
            Figure used to generate the animation.

        Raises
        ------
        ValueError
            If no non-edge void is available.
        """
        nmesh = tuple(int(x) for x in self.meshsize)
        nxy = nmesh[1] * nmesh[2]

        def zone_indices(voxels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            xi_ = voxels // nxy
            rem_ = voxels % nxy
            yi_ = rem_ // nmesh[2]
            zi_ = rem_ % nmesh[2]
            return xi_, yi_, zi_

        def touches_edge(xi_: np.ndarray, yi_: np.ndarray, zi_: np.ndarray) -> bool:
            return (
                np.any(xi_ == 0)
                or np.any(xi_ == nmesh[0] - 1)
                or np.any(yi_ == 0)
                or np.any(yi_ == nmesh[1] - 1)
                or np.any(zi_ == 0)
                or np.any(zi_ == nmesh[2] - 1)
            )

        non_edge_candidates = []
        for zid, zone in enumerate(self.zones):
            voxels = np.asarray(zone, dtype=np.int64)
            if voxels.size == 0:
                continue
            xi_c, yi_c, zi_c = zone_indices(voxels)
            if not touches_edge(xi_c, yi_c, zi_c):
                non_edge_candidates.append((voxels.size, zid, voxels, xi_c, yi_c, zi_c))

        if not non_edge_candidates:
            raise ValueError('No non-edge void found. All zones touch simulation-box boundaries or are empty.')

        non_edge_candidates.sort(key=lambda x: x[0], reverse=True)

        selected = None
        if zone_id is not None and 0 <= zone_id < len(self.zones):
            vox_try = np.asarray(self.zones[zone_id], dtype=np.int64)
            if vox_try.size > 0:
                xi_t, yi_t, zi_t = zone_indices(vox_try)
                if not touches_edge(xi_t, yi_t, zi_t):
                    selected = (vox_try.size, zone_id, vox_try, xi_t, yi_t, zi_t)

        if selected is None:
            selected = non_edge_candidates[0]

        _, zone_id_used, vox, xi, yi, zi = selected
        self.logger.info(f'Using zone {zone_id_used} with {len(vox)} voxels for 3D GIF.')

        filled = np.zeros(nmesh, dtype=bool)
        filled[xi, yi, zi] = True

        facecolors = np.zeros(filled.shape + (4,), dtype=float)
        color = matplotlib.colors.to_rgba('#6eaed6', alpha=0.92)
        facecolors[filled] = color

        xmin = max(int(np.min(xi)) - padding_cells, 0)
        xmax = min(int(np.max(xi)) + padding_cells + 1, nmesh[0])
        ymin = max(int(np.min(yi)) - padding_cells, 0)
        ymax = min(int(np.max(yi)) + padding_cells + 1, nmesh[1])
        zmin = max(int(np.min(zi)) - padding_cells, 0)
        zmax = min(int(np.max(zi)) + padding_cells + 1, nmesh[2])

        sx = max(xmax - xmin, 1)
        sy = max(ymax - ymin, 1)
        sz = max(zmax - zmin, 1)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((sx, sy, sz))
        ax.voxels(
            filled,
            facecolors=facecolors,
            edgecolor=(0.17, 0.27, 0.36, 0.18),
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        if show_axes:
            offset = np.asarray(self.boxcenter, dtype=float) - np.asarray(self.boxsize, dtype=float) / 2.0
            cellsize = np.asarray(self.cellsize, dtype=float)

            xticks = np.linspace(xmin, xmax, 4)
            yticks = np.linspace(ymin, ymax, 4)
            zticks = np.linspace(zmin, zmax, 4)

            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_zticks(zticks)

            ax.set_xticklabels([f'{(x * cellsize[0] + offset[0]):.0f}' for x in xticks])
            ax.set_yticklabels([f'{(y * cellsize[1] + offset[1]):.0f}' for y in yticks])
            ax.set_zticklabels([f'{(z * cellsize[2] + offset[2]):.0f}' for z in zticks])

            ax.set_xlabel(r'$x\,[h^{-1}{\rm Mpc}]$', labelpad=8)
            ax.set_ylabel(r'$y\,[h^{-1}{\rm Mpc}]$', labelpad=8)
            ax.set_zlabel(r'$z\,[h^{-1}{\rm Mpc}]$', labelpad=8)

            ax.grid(True, alpha=0.25)
            ax.xaxis.pane.set_alpha(0.02)
            ax.yaxis.pane.set_alpha(0.02)
            ax.zaxis.pane.set_alpha(0.02)
        else:
            ax.set_axis_off()

        def update(frame: int):
            azim = 360.0 * n_turns * frame / n_frames
            ax.view_init(elev=elev, azim=azim)
            return ()

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval,
            blit=False,
        )

        out = Path(save_fn) if save_fn else Path('void_3d.gif')
        anim.save(out, writer='pillow', dpi=dpi)

        plt.tight_layout()
        return fig
