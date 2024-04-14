import numpy as np
import logging
import time
from pandas import qcut
import healpy as hp
from .base import BaseEnvironmentEstimator


class DensitySplit(BaseEnvironmentEstimator):
    """
    Class to compute density-split clustering, as in http://arxiv.org/abs/2309.16541.
    """
    def __init__(self, **kwargs):
        self.logger = logging.getLogger('DensitySplit')
        self.logger.info('Initializing DensitySplit.')
        super().__init__(**kwargs)

    def set_quantiles(self, query_positions=None, query_method='randoms',
        nquery_factor=5, nquantiles=5):
        """
        Get the quantiles of the overdensity density field.

        Parameters
        ----------
        query_positions : array_like, optional
            Query positions.
        query_method : str, optional
            Method to generate query points. Options are 'lattice' or 'randoms'.
        nquery_factor : int, optional
            Factor to multiply the number of data points to get the number of query points.
        nquantiles : int
            Number of quantiles.

        Returns
        -------
        quantiles : array_like
            Quantiles of the density field.
        quantiles_idx : array_like, optional
            Index of the quantile of each query point.
        delta_query : array_like, optional
            Density contrast at the query points.
        """
        t0 = time.time()
        if query_positions is None:
            if self.has_randoms:
                raise ValueError('Query points must be provided when working with a non-uniform geometry.')
            else:
                query_positions = self.get_query_positions(self.delta_mesh, method=query_method,
                                                           nquery=nquery_factor*self._size_data)
        self.query_method = query_method
        self.query_positions = query_positions
        self.delta_query = self.delta_mesh.read_cic(query_positions)
        self.quantiles_idx = qcut(self.delta_query, nquantiles, labels=False)
        quantiles = []
        for i in range(nquantiles):
            quantiles.append(self.query_positions[self.quantiles_idx == i])
        self.quantiles = quantiles
        self.logger.info(f"Quantiles calculated in {time.time() - t0:.2f} seconds.")
        return self.quantiles, self.quantiles_idx, self.delta_query

    def makemap(sample):
        if (sample == 5) or (sample == 6):
            print('Making lensing map')
            kappa_map_alm = hp.read_alm(read_path(sample))
            map = hp.alm2map(kappa_map_alm, nside)
            # Note that namaster autmatically multiplies by the mask compute_full_master
            # I verify this in test_namaster.py (using Eiichiro Komatsu's MASTER code for mask deconvolution)
            # Can also check this by comparing nmt.compute_coupled_cell to hp.anafast
        elif sample >= 9:
            #numcounts_map = hp.read_map(read_path(sample))
            numcounts_map = fits.open(read_path(sample))[0].data
            masked_count = numcounts_map * mask
            mean_count = np.nansum(masked_count)/np.nansum(mask)
            masked_count_dn = numcounts_map / mean_count - 1.
            map = masked_count_dn
        else:
            # Converting the masked number counts to delta_n/n. Only consider unmasked regions!
            print('Making galaxy map ' + str(sample))
            numcounts_map = hp.read_map(read_path(sample), field=[0]) #* weights
            # Correct for lower density in regions of high area lost due to stars or stellar masking
            numcounts_map = numcounts_map / mask_lost
            masked_count = numcounts_map * mask
            mean_count = np.nansum(masked_count) / np.nansum(mask)
            masked_count_dn = numcounts_map / mean_count - 1.
            
            map = masked_count_dn
            map[mask_lost == 0] = 0
            #std_map = np.sqrt( np.sum(map**2) / np.sum(mask) )
            #print std_map
            #hp.mollview(map)
            #pl.show()
        return map
