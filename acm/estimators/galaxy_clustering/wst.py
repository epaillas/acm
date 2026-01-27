import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from kymatio.jax import HarmonicScattering3D

from acm.utils.plotting import set_plot_style
from .base import BaseEstimator

import warnings
warnings.filterwarnings('ignore')


class WaveletScatteringTransform(BaseEstimator):
    """
    Class to compute the wavelet scattering transform.
    """
    def __init__(self, J=4, L=4, q=0.8, sigma=0.8, init_kymatio=None, **kwargs):

        self.logger = logging.getLogger('WaveletScatteringTransform')
        super().__init__(**kwargs)
        self.logger.info(f'Using {self.backend.__class__.__name__} backend.')

        self.J = J
        self.L = L
        self.sigma_0 = sigma
        self.q = q
        self.max_order = 2

        self.query_positions = self.backend.get_query_positions(method='lattice')

        if init_kymatio is not None:
            self.logger.info(f'Pre-loading Kymatio initialization.')
            self.S = init_kymatio
        else:
            self.init_kymatio()
        
    def init_kymatio(self):
        """
        Initialize the kymatio scattering transform.
        """
        t0 = time.time()
        self.logger.info('Initializing WaveletScatteringTransform.')
        self.logger.info(f'J={self.J}, L={self.L}, sigma_0={self.sigma_0}, max_order={self.max_order}')
        self.S = HarmonicScattering3D(
            J=self.J,
            L=self.L,
            shape=self.backend.meshsize,
            max_order=self.max_order,
            sigma_0=self.sigma_0,
        )
        self.logger.info(f'Initialized Kymatio in {time.time() - t0:.2f} s.')


    def run(self, delta_query=None):
        """
        Run the wavelet scattering transform.

        Returns
        -------
        smatavg : array_like
            Wavelet scattering transform coefficients.
        """
        t0 = time.time()
        if delta_query is not None:
            self.delta_query = delta_query.reshape(self.backend.meshsize)
        else:
            self.delta_query = self.read_density_contrast(self.query_positions)
        smat_orders_12 = self.S(self.delta_query)
        smat = np.absolute(smat_orders_12[:, :, 0])
        s0 = np.sum(np.absolute(self.delta_query) ** self.q)
        smatavg = smat.flatten()
        self.smatavg = np.hstack((s0, smatavg))
        self.smatavg /= np.prod(self.backend.meshsize)
        self.logger.info(f"WST coefficients done in {time.time() - t0:.2f} s.")
        return self.smatavg

    @set_plot_style
    def plot_coefficients(self, save_fn=None):
        """
        Plot the wavelet scattering transform coefficients.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(self.smatavg, ls='-', marker='o', markersize=4, label=r'{\rm AbacusSummit}')
        ax.set_xlabel('WST coefficient order')
        ax.set_ylabel('WST coefficient')
        plt.tight_layout()
        if save_fn is not None: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig
