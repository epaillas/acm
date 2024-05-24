from kymatio.jax import HarmonicScattering3D
import numpy as np
import logging
import time
from .base import BaseEnvironmentEstimator


class WaveletScatteringTransform(BaseEnvironmentEstimator):
    """
    Class to compute the wavelet scattering transform.
    """
    def __init__(self, J_3d=4, L_3d=4, integral_powers=[0.8], sigma=0.8, **kwargs):

        self.logger = logging.getLogger('WaveletScatteringTransform')
        self.logger.info('Initializing WaveletScatteringTransform.')
        super().__init__(**kwargs)

        self.S = HarmonicScattering3D(J=J_3d, shape=self.data_mesh.shape, L=L_3d, sigma_0=sigma,
                                 integral_powers=integral_powers, max_order=2)

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
        self.delta_query = self.delta_mesh.read_cic(query_positions).reshape(
            (self.delta_mesh.nmesh[0], self.delta_mesh.nmesh[1], self.delta_mesh.nmesh[2]))
        smat_orders_12 = self.S(self.delta_query)
        smat = np.absolute(smat_orders_12[:, :, 0])
        s0 = np.sum(np.absolute(self.delta_mesh)**0.80)
        smatavg = smat.flatten()
        self.smatavg = np.hstack((s0, smatavg))
        self.logger.info(f"WST coefficients elapsed in {time.time() - t0:.2f} seconds.")
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
