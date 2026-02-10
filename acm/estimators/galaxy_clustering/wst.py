import time
import warnings
from typing import Optional

import numpy as np
import jax.numpy as jnp
import numpy.typing as npt
import matplotlib.pyplot as plt
import torch
from kymatio.jax import HarmonicScattering3D

from acm.utils.plotting import set_plot_style
from .base import BaseEstimator

warnings.filterwarnings('ignore', category=DeprecationWarning)


class WaveletScatteringTransform(BaseEstimator):
    """
    Class to compute the wavelet scattering transform.
    """
    def __init__(self, J: int = 4, L: int = 4, q: float = 0.8, sigma: float = 0.8, init_kymatio = None,
                 kymatio_backend: str = 'torch', **kwargs) -> None:
        
        super().__init__(**kwargs)
        
        self.J = J
        self.L = L
        self.sigma_0 = sigma
        self.q = q
        self.max_order = 2

        self.kymatio_backend = kymatio_backend
        if kymatio_backend == 'torch':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f'Using Kymatio with Torch backend on device: {self.device}')
        else:
            self.logger.info(f'Using Kymatio with JAX backend.')

        self.query_positions = self.get_query_positions(method='lattice')
    
        if init_kymatio is not None:
            self.logger.info(f'Pre-loading Kymatio initialization.')
            self.S = init_kymatio
        else:
            self.init_kymatio()
        
    def init_kymatio(self) -> None:
        """
        Initialize the kymatio scattering transform.
        """
        module = __import__(f'kymatio.{self.kymatio_backend}', fromlist=['HarmonicScattering3D'])
        HarmonicScattering3D = getattr(module, 'HarmonicScattering3D')

        t0 = time.time()
        self.logger.info('Initializing WaveletScatteringTransform.')
        self.logger.info(f'J={self.J}, L={self.L}, sigma_0={self.sigma_0}, max_order={self.max_order}')
        self.S = HarmonicScattering3D(
            J=self.J,
            L=self.L,
            shape=self.meshsize,
            max_order=self.max_order,
            sigma_0=self.sigma_0,
        )
        if self.kymatio_backend == 'torch':
            self.S.to(self.device)
        self.logger.info(f'Initialized Kymatio in {time.time() - t0:.2f} s.')


    def _run_torch(self, delta_query: npt.NDArray) -> npt.NDArray:
        """
        Run the wavelet scattering transform with Torch backend.

        Parameters
        ----------
        delta_query : array_like
            Density contrast field.

        Returns
        -------
        smatavg : array_like
            Wavelet scattering transform coefficients.
        """
        delta_query_torch = torch.from_numpy(np.asarray(delta_query, dtype='float32')).to(self.device)
        s0 = torch.sum(torch.abs(delta_query_torch) ** self.q)
        smat_orders_12 = self.S(delta_query_torch)
        smat = torch.abs(smat_orders_12[:, :, 0]).flatten()
        smatavg = torch.cat([s0.unsqueeze(0), smat])
        smatavg /= np.prod(self.meshsize)
        return smatavg.cpu().numpy()

    def _run_jax(self, delta_query: npt.NDArray) -> npt.NDArray:
        """
        Run the wavelet scattering transform with JAX backend.

        Parameters
        ----------
        delta_query : array_like
            Density contrast field.

        Returns
        -------
        smatavg : array_like
            Wavelet scattering transform coefficients.
        """
        s0 = jnp.sum(jnp.abs(delta_query) ** self.q)
        smat_orders_12 = self.S(delta_query)
        smat = jnp.abs(smat_orders_12[:, :, 0]).flatten()
        smatavg = jnp.concatenate([jnp.array([s0]), smat])
        smatavg /= np.prod(self.meshsize)
        return np.asarray(smatavg)

    def run(self, delta_query: Optional[npt.NDArray] = None) -> npt.NDArray:
        """
        Run the wavelet scattering transform.

        Returns
        -------
        smatavg : array_like
            Wavelet scattering transform coefficients.
        """
        if delta_query is not None:
            self.delta_query = delta_query.reshape(self.meshsize)
        else:
            self.delta_query = self.read_density_contrast(self.query_positions).reshape(self.meshsize)
        # Call appropriate backend
        t0 = time.time()
        if self.kymatio_backend == 'torch':
            self.smatavg = self._run_torch(self.delta_query)
        else:
            self.smatavg = self._run_jax(self.delta_query)
        self.logger.info(f"WST coefficients done in {time.time() - t0:.2f} s.")
        return self.smatavg

    @set_plot_style
    def plot_coefficients(self, save_fn: Optional[str] = None):
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
