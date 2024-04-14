import numpy as np
import logging
import time
import os
import subprocess
import uuid
from pathlib import Path
from .src import fastmodules
from .base import BaseEnvironmentEstimator

class VoxelVoids(BaseEnvironmentEstimator):
    """
    Class to calculate voxel voids, as in https://github.com/seshnadathur/Revolver
    """
    def __init__(self, temp_dir, **kwargs):
        self.logger = logging.getLogger('VoxelVoids')
        self.logger.info('Initializing VoxelVoids.')
        super().__init__(**kwargs)
        self.handle = Path(temp_dir) / str(uuid.uuid4())

    def find_voids(self):
        """
        Run the voxel voids algorithm.
        """
        self.time = time.time()
        self._find_voids()
        self.voids, self.void_radii = self._postprocess_voids()
        nvoids = len(self.voids)
        self.logger.info(f"Found {nvoids} voxel voids in {time.time() - self.time:.2f} seconds.")
        return self.voids, self.void_radii

    def _find_voids(self):
        """
        Find voids in the overdensity field.
        """
        self.logger.info("Finding voids.")
        nmesh = self.delta_mesh.nmesh
        delta_mesh_flat = np.array(self.delta_mesh, dtype=np.float32)
        with open(f'{self.handle}_delta_mesh_n{nmesh[0]}{nmesh[1]}{nmesh[2]}d.dat', 'w') as F:
            delta_mesh_flat.tofile(F, format='%f')
        bin_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)), './src', 'jozov-grid.exe')
        cmd = [bin_path, "v", f"{self.handle}_delta_mesh_n{nmesh[0]}{nmesh[1]}{nmesh[2]}d.dat",
               self.handle, str(nmesh[0]),str(nmesh[1]),str(nmesh[2])]
        subprocess.call(cmd)

    def _postprocess_voids(self):
        """
        Post-process voids to remove edge voids and voids in masked voxels.
        """
        self.logger.info("Post-processing voids.")
        nmesh = self.delta_mesh.nmesh
        cellsize = self.delta_mesh.cellsize[0]
        mask_cut = np.zeros(nmesh[0] * nmesh[1] * nmesh[2], dtype='int')
        if self.has_randoms:
            # identify "empty" cells for later cuts on void catalogue
            mask_cut = np.zeros(nmesh[0] * nmesh[1] * nmesh[2], dtype='int')
            fastmodules.survey_mask(mask_cut, self.randoms_mesh.value, self.ran_min)
        self.mask_cut = mask_cut
        self.min_dens_cut = 1.0
        rawdata = np.loadtxt(f"{self.handle}.txt", skiprows=2)
        # remove voids that: a) don't meet minimum density cut, b) are edge voids, or c) lie in a masked voxel
        select = np.zeros(rawdata.shape[0], dtype='int')
        fastmodules.voxelvoid_cuts(select, self.mask_cut, rawdata, self.min_dens_cut)
        select = np.asarray(select, dtype=bool)
        rawdata = rawdata[select]
        # void minimum density centre locations
        self.logger.info('Calculating void positions.')
        xpos, ypos, zpos = self.voxel_position(rawdata[:, 2])
        self.core_dens = rawdata[:, 3]
        # void effective radii
        self.logger.info('Calculating void radii.')
        vols = (rawdata[:, 5] * cellsize ** 3.)
        rads = (3. * vols / (4. * np.pi)) ** (1. / 3)
        self.zones = []
        with open(f'{self.handle}.zone', 'r') as f:
            for line in f:
                self.zones.append([int(i) for i in line.split()])
        self.zones = [zone for i, zone in enumerate(self.zones) if select[i]]
        os.remove(f'{self.handle}.void')
        os.remove(f'{self.handle}.txt')
        os.remove(f'{self.handle}.zone')
        os.remove(f'{self.handle}_delta_mesh_n{nmesh[0]}{nmesh[1]}{nmesh[2]}d.dat')
        return np.c_[xpos, ypos, zpos], rads

    def voxel_position(self, voxel):
        """
        Calculate the position of a voxel in the mesh.
        """
        voxel = voxel.astype('i')
        boxsize = self.delta_mesh.boxsize
        boxcenter = self.delta_mesh.boxcenter
        nmesh = self.delta_mesh.nmesh
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
                yi[i*(nmesh[2]):(i + 1) * (nmesh[2])] =i
            for i in range(nmesh[0]):
                xi[i*(nmesh[1] * nmesh[2]):(i + 1) * (nmesh[1] * nmesh[2])] = i
            xpos = xi * boxsize[0] / nmesh[0]
            ypos = np.tile(yi, nmesh[0]) * boxsize[1] / nmesh[1]
            zpos = np.tile(zi, nmesh[1] * nmesh[0]) * boxsize[2 ] / nmesh[2]
            xpos += boxcenter[0] - boxsize[0] / 2.
            ypos += boxcenter[1] - boxsize[1] / 2.           
            zpos += boxcenter[2] - boxsize[2] / 2.
            return xpos[voxel],ypos[voxel],zpos[voxel]
        else:
            for i in range(nmesh[1]):
                yi[i * (nmesh[2]):(i + 1) * (nmesh[2])] = i
            for i in range(nmesh[0]):
                xi[i*(nmesh[1] * nmesh[2]):(i + 1) * (nmesh[1] * nmesh[2])] = i
            xpos = xi * boxsize[0] / nmesh[0]
            ypos = np.tile(yi, nmesh[0]) * boxsize[1] / nmesh[1]
            zpos = np.tile(zi, nmesh[1] * nmesh[0]) * boxsize[2] / nmesh[2]
            return xpos[voxel], ypos[voxel], zpos[voxel]

    def void_data_correlation(self, data_positions, **kwargs):
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
        from pycorr import TwoPointCorrelationFunction
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
                kwargs['boxsize'] = self.delta_mesh.boxsize
        self._void_data_correlation = TwoPointCorrelationFunction(
            data_positions1=self.voids,
            data_positions2=data_positions,
            mode='smu',
            position_type='pos',
            **kwargs,
        )
        return self._void_data_correlation

    def plot_void_size_distribution(self, save_fn=None):
        """
        Plot the void size distribution.
        """
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.hist(self.void_radii, bins=25, lw=2.0, alpha=0.5)
        ax.set_xlabel(r'$R_{\rm void}\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$N$', fontsize=15)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig

    def plot_void_data_correlation(self, ells=(0,), save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        s, multipoles = self._void_data_correlation(ells=(0, 2, 4), return_sep=True)
        for ell in ells:
            ax.plot(s, multipoles[ell//2], lw=2.0, label=f'$\\ell = {ell}$')
        ax.set_xlabel(r'$s\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$\xi_\ell(s)$', fontsize=15)
        ax.legend(fontsize=15, loc='best', handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig

    def plot_slice(self, data_positions=None, save_fn=None):
        """
        Plot a slice of the density field.
        """
        import matplotlib.pyplot as plt
        import matplotlib
        import random
        nmesh = self.delta_mesh.nmesh
        boxsize = self.delta_mesh.boxsize
        boxcenter = self.delta_mesh.boxcenter
        zones_mesh = np.zeros(nmesh).flatten()
        for i, zone in enumerate(self.zones):
            zones_mesh[zone] = random.random()
        zones_mesh = np.ma.masked_where(zones_mesh == 0, zones_mesh)
        zones_mesh = zones_mesh.reshape(self.delta_mesh.shape)
        zones_mesh = np.sum(zones_mesh, axis=2)
        fig, ax = plt.subplots()
        cmap = matplotlib.cm.tab20
        cmap.set_bad(color='white')
        ax.imshow(zones_mesh[:, :], origin='lower', cmap=cmap,
                  extent=[0, boxsize[0], 0, boxsize[1]], interpolation='gaussian')
        # ax.set_xlim(0, 1000)
        # ax.set_ylim(0, 1000)
        ax.set_xlabel(r'$x\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(r'$y\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig