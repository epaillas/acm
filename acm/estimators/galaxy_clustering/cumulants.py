import numpy as np
import logging
import time
from .base import BaseEnvironmentEstimator


class DensityFieldCumulants(BaseEnvironmentEstimator):
    """
    Class to compute the cumulant generating function of the density field.
    """
    def __init__(self, **kwargs):
        self.logger = logging.getLogger('DensityFieldCumulants')
        self.logger.info('Initializing DensityFieldCumulants.')
        super().__init__(**kwargs)

    def compute_cumulants(self, lda, query_positions=None):
        """
        Compute the cumulant generating function of the density field.

        Parameters
        ----------
        lda : array
            Values of lambda at which to compute the cumulant generating function.

        Returns
        -------
        cgf : array
            Cumulant generating function of the density field at lda values.
        """
        t0 = time.time()
        self.lda = lda
        self.delta_query = self.delta_mesh.value.flatten()
        self.cgf = np.log(np.mean(np.exp(lda[:, None] * self.delta_query), axis=1))
        self.logger.info(f"Computed cumulants in {time.time() - t0:.2f} seconds.")
        return self.cgf

    def plot_cumulants(self, save_fn=None):
        """
        Plot the cumulant generating function.
        """
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(self.lda, self.cgf)
        ax.set_xlabel(r'$\lambda$', fontsize=15)
        ax.set_ylabel(r'$\log\langle e^{\lambda \delta}\rangle$', fontsize=15)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig

    def plot_density_pdf(self, save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.hist(self.delta_query, bins=200, density=True, lw=2.0)
        ax.set_xlabel(r'$\delta \left(R_s = 10\, h^{-1}{\rm Mpc}\right)$', fontsize=15)
        ax.set_ylabel('PDF', fontsize=15)
        ax.set_xlim(-1.3, 3.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig