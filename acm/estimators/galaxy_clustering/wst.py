# import torch
from kymatio.jax import HarmonicScattering3D
import numpy as np
import logging
import time
from .base import BaseDensityMeshEstimator


class WaveletScatteringTransform(BaseDensityMeshEstimator):
    """
    Class to compute the wavelet scattering transform.
    """
    def __init__(self, J_3d=4, L_3d=4, integral_powers=[1.0], sigma=0.8, init_kymatio=None, **kwargs):

        self.logger = logging.getLogger('WaveletScatteringTransform')
        self.logger.info('Initializing WaveletScatteringTransform.')
        super().__init__(**kwargs)

        self.J_3d = J_3d
        self.L_3d = L_3d
        self.sigma_0 = sigma
        self.integral_powers = integral_powers
        self.max_order = 2

        self.query_positions = self.get_query_positions(method='lattice')

        if init_kymatio is not None:
            self.logger.info('Pre-loading Kymatio initialization.')
            self.S = init_kymatio
        else:
            self.init_kymatio()

    def init_kymatio(self):
        """
        Initialize the kymatio scattering transform.
        """
        self.S = HarmonicScattering3D(
            J=self.J_3d,
            L=self.L_3d,
            shape=self.meshsize,
            max_order=self.max_order,
            sigma_0=self.sigma_0,
        )


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
            self.delta_query = delta_query.reshape(self.meshsize)
        else:
            self.delta_query = self.delta_mesh.read(self.query_positions).real.reshape(self.meshsize)
        # self.delta_query = torch.tensor(np.copy(self.delta_query), dtype=torch.float32).to(self.device)
        smat_orders_12 = self.S(self.delta_query)
        smat = np.absolute(smat_orders_12[:, :, 0])
        s0 = np.sum(np.absolute(self.delta_query)**self.integral_powers[0])
        smatavg = smat.flatten()
        # self.smatavg = np.hstack((s0, smatavg)).cpu()
        self.smatavg = np.hstack((s0, smatavg))
        self.smatavg /= np.prod(self.meshsize)
        self.logger.info(f"WST coefficients done in {time.time() - t0:.2f} s.")
        return self.smatavg

    def plot_coefficients(self, save_fn=None):
        """
        Plot the wavelet scattering transform coefficients.
        """
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(self.smatavg, ls='-', marker='o', markersize=4, label=r'{\rr AbacusSummit}')
        ax.set_xlabel('WST coefficient order')
        ax.set_ylabel('WST coefficient')
        plt.tight_layout()
        if save_fn is not None: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig