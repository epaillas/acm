import numpy as np
import logging
import time
from pandas import qcut
from .base import BaseEnvironmentEstimator, BaseCatalogMeshEstimator


class CountsInCells(BaseEnvironmentEstimator):
    """
    Class to compute counts in cells.
    """
    def __init__(self, **kwargs):
        self.logger = logging.getLogger('CountsInCells')
        self.logger.info('Initializing CountsInCells.')
        super().__init__(**kwargs)

    def sample_pdf(self, query_positions=None, query_method='randoms',
        nquery_factor=5):
        """
        Get the quantiles of the overdensity density field.

        Parameters
        ----------
        nquantiles : int
            Number of quantiles.
        return_idx : bool, optional
            Whether to return index of the quantile of each query point.

        Returns
        -------
        quantiles : array_like
            Quantiles of the density field.
        quantiles_idx : array_like, optional
            Index of the quantile of each query point.
        """
        t0 = time.time()
        if query_positions is None:
            if self.has_randoms:
                raise ValueError('Query points must be provided when working with a non-uniform geometry.')
            else:
                query_positions = self.get_query_positions(method=query_method,
                                                           nquery=nquery_factor*self._size_data)
        self.query_method = query_method
        self.query_positions = query_positions
        self.delta_query = self.delta_mesh.read_cic(query_positions)
        return self.delta_query

    def plot_quantiles(self, save_fn=None):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(4, 4))
        hist, bin_edges, patches = ax.hist(self.delta_query, bins=200, density=True, lw=2.0)
        ax.set_xlabel(r'$\Delta \left(R_s = 10\, h^{-1}{\rm Mpc}\right)$', fontsize=15)
        ax.set_ylabel('PDF', fontsize=15)
        ax.set_xlim(-1.3, 3.0)
        ax.legend(handlelength=1.0)
        plt.tight_layout()
        if save_fn: plt.savefig(save_fn, bbox_inches='tight')
        plt.show()
        return fig