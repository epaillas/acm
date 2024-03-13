from pyrecon import RealMesh
from kymatio.torch import HarmonicScattering3D
import numpy as np
import logging
import torch


class WaveletScatteringTransform:
    """
    Class to compute the wavelet scattering transform.

    Parameters
    ----------
    data_positions : array_like
        Positions of the data points.
    boxsize : float, optional
        Size of the box. If not provided, randoms are required.
    data_weights : array_like, optional
        Weights of the data points. If not provided, all points are 
        assumed to have the same weight.
    randoms_positions : array_like, optional
        Positions of the random points. If not provided, boxsize must 
        be provided.
    randoms_weights : array_like, optional
        Weights of the random points. If not provided, all points are 
        assumed to have the same weight.

    Attributes
    ----------
    data_positions : array_like
        Positions of the data points.
    boxsize : float
        Size of the box.
    data_weights : array_like
        Weights of the data points.
    randoms_positions : array_like
        Positions of the random points.
    randoms_weights : array_like
        Weights of the random points.
    mesh : CatalogMesh
        CatalogMesh object.
    density : array_like
        Density field at the sampling positions.
    """
    def __init__(self, data_positions, boxsize=None, boxcenter=None,
        data_weights=None, randoms_positions=None, randoms_weights=None,
        cellsize=None, wrap=False, boxpad=1.5, nthreads=None):
        self.data_positions = data_positions
        self.randoms_positions = randoms_positions
        self.boxsize = boxsize
        self.boxcenter = boxcenter
        self.cellsize = cellsize
        self.boxpad = boxpad
        self.wrap = wrap
        self.nthreads = nthreads

        self.logger = logging.getLogger('WaveletScatteringTransform')

        if data_weights is not None:
            self.data_weights = data_weights
        else:
            self.data_weights = np.ones(len(data_positions))

        if boxsize is None:
            if randoms_positions is None:
                raise ValueError(
                    'boxsize is set to None, but randoms were not provided.')
            if randoms_weights is None:
                self.randoms_weights = np.ones(len(randoms_positions))
            else:
                self.randoms_weights = randoms_weights


    def get_delta_mesh(self, smoothing_radius,
        check=False, ran_min=0.01):
        """
        Get the overdensity field.

        Parameters
        ----------
        smooth_radius : float
            Radius of the smoothing filter.
        sampling_positions : array_like
            Positions where the density field should be sampled.
        Returns
        -------
        density : array_like
            Density field at the sampling positions.
        """
        self.logger.info('Computing the overdensity field.')
        self.data_mesh = RealMesh(boxsize=self.boxsize, cellsize=self.cellsize,
                                  boxcenter=self.boxcenter, nthreads=self.nthreads,
                                  positions=self.randoms_positions, boxpad=self.boxpad)
        self.data_mesh.assign_cic(positions=self.data_positions, wrap=self.wrap,
                                  weights=self.data_weights)
        self.data_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True,)
        if self.boxsize is None:
            self.randoms_mesh = RealMesh(boxsize=self.boxsize, cellsize=self.cellsize,
                                         boxcenter=self.boxcenter, nthreads=self.nthreads,
                                         positions=self.randoms_positions, boxpad=self.boxpad)
            self.randoms_mesh.assign_cic(positions=self.randoms_positions, wrap=self.wrap,
                                         weights=self.randoms_weights)
            if check:
                mask_nonzero = self.randoms_mesh.value > 0.
                nnonzero = mask_nonzero.sum()
                if nnonzero < 2: raise ValueError('Very few randoms.')
            self.randoms_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True)
            sum_data, sum_randoms = np.sum(self.data_mesh.value), np.sum(self.randoms_mesh.value)
            alpha = sum_data * 1. / sum_randoms
            self.delta_mesh = self.data_mesh - alpha * self.randoms_mesh
            threshold = ran_min * sum_randoms / len(self.randoms_positions)
            mask = self.randoms_mesh > threshold
            self.delta_mesh[mask] /= alpha * self.randoms_mesh[mask]
            self.delta_mesh[~mask] = 0.0
            del self.data_mesh
            del self.randoms_mesh
        else:
            self.delta_mesh = self.data_mesh / np.mean(self.data_mesh) - 1.
            del self.data_mesh
        self.logger.info(f'Box size: {self.delta_mesh.boxsize}')
        self.logger.info(f'Box center: {self.delta_mesh.boxcenter}')
        self.logger.info(f'Box nmesh: {self.delta_mesh.nmesh}')
        query_positions = self.get_query_positions()
        self.delta_mesh = self.delta_mesh.read_cic(query_positions).reshape(
            (self.delta_mesh.nmesh[0], self.delta_mesh.nmesh[1], self.delta_mesh.nmesh[2]))
        return self.delta_mesh

    def get_query_positions(self):
        boxcenter = self.delta_mesh.boxcenter
        boxsize = self.delta_mesh.boxsize
        xedges = np.arange(
            boxcenter[0] - boxsize[0]/2,
            boxcenter[0] + boxsize[0]/2 + self.cellsize,
            self.cellsize)
        yedges = np.arange(
            boxcenter[1] - boxsize[1]/2,
            boxcenter[1] + boxsize[1]/2 + self.cellsize,
            self.cellsize)
        zedges = np.arange(
            boxcenter[2] - boxsize[2]/2,
            boxcenter[2] + boxsize[2]/2 + self.cellsize,
            self.cellsize)
        xcentres = 1/2 * (xedges[:-1] + xedges[1:])
        ycentres = 1/2 * (yedges[:-1] + yedges[1:])
        zcentres = 1/2 * (zedges[:-1] + zedges[1:])
        lattice_x, lattice_y, lattice_z = np.meshgrid(xcentres, ycentres, zcentres)
        lattice_x = lattice_x.flatten()
        lattice_y = lattice_y.flatten()
        lattice_z = lattice_z.flatten()
        return np.vstack((lattice_x, lattice_y, lattice_z)).T

    def get_wst(self):
        self.logger.info("Calling kymatio's HarmonicScattering3D.")
        D3d = 300
        J3d = 4
        L3d = 4
        integral_powers = [0.8]
        sigma = 0.8
        S = HarmonicScattering3D(J=J3d, shape=(D3d, D3d, D3d), L=L3d, sigma_0=sigma,
                                 integral_powers=integral_powers, max_order=2)
        delta_mesh = torch.from_numpy(self.delta_mesh).float()
        smat_orders_1_and_2 = S(self.delta_mesh)
