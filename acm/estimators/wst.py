from pyrecon import RealMesh
# from kymatio.torch import HarmonicScattering3D
from kymatio.jax import HarmonicScattering3D
import numpy as np
import logging
import time


class WaveletScatteringTransform:
    """
    Class to compute the wavelet scattering transform.
    """
    def __init__(self, J_3d=4, L_3d=4, integral_powers=[0.8], sigma=0.8, **kwargs):

        self.logger = logging.getLogger('WaveletScatteringTransform')
        self.logger.info('Initializing WaveletScatteringTransform.')

        self.data_mesh = RealMesh(**kwargs)
        self.randoms_mesh = RealMesh(**kwargs)
        self.logger.info(f'Box size: {self.data_mesh.boxsize}')
        self.logger.info(f'Box center: {self.data_mesh.boxcenter}')
        self.logger.info(f'Box nmesh: {self.data_mesh.nmesh}')

        self.S = HarmonicScattering3D(J=J_3d, shape=self.data_mesh.shape, L=L_3d, sigma_0=sigma,
                                 integral_powers=integral_powers, max_order=2)

    def assign_data(self, positions, weights=None, wrap=False, clear_previous=True):
        """
        Assign data to the mesh.

        Parameters
        ----------
        positions : array_like
            Positions of the data points.
        weights : array_like, optional
            Weights of the data points. If not provided, all points are 
            assumed to have the same weight.
        wrap : bool, optional
            Wrap the data points around the box, assuming periodic boundaries.
        clear_previous : bool, optional
            Clear previous data.
        """
        if clear_previous:
            self.data_mesh.value = None
        self.data_mesh.assign_cic(positions=positions, weights=weights, wrap=wrap)

    def assign_randoms(self, positions, weights=None):
        """
        Assign randoms to the mesh.

        Parameters
        ----------
        positions : array_like
            Positions of the random points.
        weights : array_like, optional
            Weights of the random points. If not provided, all points are 
            assumed to have the same weight.
        """
        if self.randoms_mesh.value is None:
            self._size_randoms = 0
        self.randoms_mesh.assign_cic(positions=positions, weights=weights)
        self._size_randoms += len(positions)

    @property
    def has_randoms(self):
        return self.randoms_mesh.value is not None

    def set_density_contrast(self, smoothing_radius=None, check=False, ran_min=0.01):
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
            
        Returns
        -------
        delta_mesh : array_like
            Density contrast.
        """
        self.logger.info('Setting density contrast.')
        if smoothing_radius:
            self.data_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True,)
        if self.has_randoms:
            if check:
                mask_nonzero = self.randoms_mesh.value > 0.
                nnonzero = mask_nonzero.sum()
                if nnonzero < 2: raise ValueError('Very few randoms.')
            if smoothing_radius:
                self.randoms_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True)
            sum_data, sum_randoms = np.sum(self.data_mesh.value), np.sum(self.randoms_mesh.value)
            alpha = sum_data * 1. / sum_randoms
            self.delta_mesh = self.data_mesh - alpha * self.randoms_mesh
            threshold = ran_min * sum_randoms / self._size_randoms
            mask = self.randoms_mesh > threshold
            self.delta_mesh[mask] /= alpha * self.randoms_mesh[mask]
            self.delta_mesh[~mask] = 0.0
        else:
            self.delta_mesh = self.data_mesh / np.mean(self.data_mesh) - 1.
        query_positions = self.get_lattice_positions(self.delta_mesh)
        self.delta_mesh = self.delta_mesh.read_cic(query_positions).reshape(
            (self.delta_mesh.nmesh[0], self.delta_mesh.nmesh[1], self.delta_mesh.nmesh[2]))
        return self.delta_mesh

    def get_lattice_positions(self, mesh):
        """
        Get positions at the center of each mesh cell.

        Returns
        -------
        lattice_positions : array_like
            Lattice positions.
        """
        boxcenter = mesh.boxcenter
        boxsize = mesh.boxsize
        cellsize = mesh.cellsize
        xedges = np.arange(boxcenter[0] - boxsize[0]/2, boxcenter[0] + boxsize[0]/2 + cellsize[0], cellsize[0])
        yedges = np.arange(boxcenter[1] - boxsize[1]/2, boxcenter[1] + boxsize[1]/2 + cellsize[1], cellsize[1])
        zedges = np.arange(boxcenter[2] - boxsize[2]/2, boxcenter[2] + boxsize[2]/2 + cellsize[2], cellsize[2])
        xcentres = 1/2 * (xedges[:-1] + xedges[1:])
        ycentres = 1/2 * (yedges[:-1] + yedges[1:])
        zcentres = 1/2 * (zedges[:-1] + zedges[1:])
        lattice_x, lattice_y, lattice_z = np.meshgrid(xcentres, ycentres, zcentres)
        lattice_x = lattice_x.flatten()
        lattice_y = lattice_y.flatten()
        lattice_z = lattice_z.flatten()
        return np.vstack((lattice_x, lattice_y, lattice_z)).T

    def run(self):
        """
        Run the wavelet scattering transform.

        Returns
        -------
        smatavg : array_like
            Wavelet scattering transform coefficients.
        """
        t0 = time.time()
        smat_orders_12 = self.S(self.delta_mesh)
        smat = np.absolute(smat_orders_12[:, :, 0])
        s0 = np.sum(np.absolute(self.delta_mesh)**0.80)
        smatavg = smat.flatten()
        smatavg = np.hstack((s0, smatavg))
        self.logger.info(f"WST coefficients elapsed in {time.time() - t0:.2f} seconds.")
