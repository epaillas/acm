import torch
from kymatio.torch import HarmonicScattering3D
import numpy as np
import logging
import time
from .base import BaseDensityMeshEstimator


class WaveletScatteringTransform(BaseDensityMeshEstimator):
    """
    Class to compute the wavelet scattering transform.
    """
    def __init__(self, J_3d=4, L_3d=4, integral_powers=[0.8], sigma=0.8, **kwargs):

        self.logger = logging.getLogger('WaveletScatteringTransform')
        self.logger.info('Initializing WaveletScatteringTransform.')
        super().__init__(**kwargs)

        self.S = HarmonicScattering3D(
            J=J_3d,
            shape=self.data_mesh.meshsize,
            L=L_3d,
            sigma_0=sigma,
            integral_powers=integral_powers,
            max_order=2
        )

        if torch.cuda.is_available():
            self.device = 'cuda'
            self.logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
        else:
            self.device = 'cpu'
            self.logger.info('Using CPU')
        self.S.to(self.device)
        self.integral_powers = integral_powers

    def run(self):
        """
        Run the wavelet scattering transform.

        Returns
        -------
        smatavg : array_like
            Wavelet scattering transform coefficients.
        """
        t0 = time.time()
        query_positions = self.get_query_positions(self.delta_mesh, method='lattice')
        self.delta_query = self.delta_mesh.read(query_positions).reshape(self.data_mesh.meshsize)
        # if self.device == 'cuda':
        self.delta_query = torch.tensor(np.copy(self.delta_query), dtype=torch.float32).to(self.device)
        smat_orders_12 = self.S(self.delta_query)
        smat = torch.absolute(smat_orders_12[:, :, 0])
        s0 = torch.sum(torch.absolute(self.delta_query)**self.integral_powers[0])
        smatavg = smat.flatten()
        self.smatavg = torch.hstack((s0, smatavg)).cpu()
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