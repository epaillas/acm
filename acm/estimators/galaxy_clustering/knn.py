import logging
import numpy as np
import scipy.spatial
from numba import njit, prange
from .base import BaseEstimator

        
class KthNearestNeighbor(BaseEstimator):
    """
    Class to compute the knns.
    """
    def __init__(self, **kwargs):

        self.logger = logging.getLogger('KthNearestNeighbor')
        self.logger.info('Initializing KthNearestNeighbor.')
        super().__init__(**kwargs)

    def compute_knn_distances(self, data, query, k, periodic, nthread=1, leafsize=32):
        """
        pair finding with scipy ckdtree
        """
        # Make sure k in an array
        if isinstance(k, int): 
            k = [k] 

        # If any value is < 0, don't use periodic boxes
        if np.any(periodic <= 0):
            xtree = scipy.spatial.cKDTree(data, leafsize=leafsize)
        else:
            xtree = scipy.spatial.cKDTree(data, leafsize=leafsize, boxsize=periodic)

        # Query the tree (this is parallel)
        _, disi = xtree.query(query, k=k, workers=nthread)

        # Conversion into 2D should be done separately
        dis_trans, dis_par = convert_rppi(disi, data, query, k)
        return dis_trans, dis_par

    def calc_cdf_hist(self, rs, pis, dis_t, dis_p):
        """
        2d histogram wrapper function
        """
        cdfs = np.zeros((dis_t.shape[1], len(rs), len(pis)), dtype = np.float32)

        # are the bins lin or log
        rpbins = np.concatenate((np.zeros(1, dtype = np.float32), rs))
        pibins = np.concatenate((np.zeros(1, dtype = np.float32), pis))
        results = [self.calculate_single_cdf(ik, dis_t, dis_p, rpbins, pibins) for ik in range(dis_t.shape[1])]

        for ik, cdf in results:
            cdfs[ik] = cdf

        return cdfs

    def calculate_single_cdf(self, ik, dis_t, dis_p, rpbins, pibins):
        """
        tabulate pair distances into 2d histogram
        """
        # Do 2d histogram of obtained rp and pi
        dist_hist2d_k, _, _ = np.histogram2d(dis_t[:,ik], dis_p[:,ik], bins=(rpbins, pibins))
        dist_cdf2d_k = np.cumsum(np.cumsum(dist_hist2d_k, axis=0), axis=1)
        # Normalization
        cdf = dist_cdf2d_k / dist_cdf2d_k[-1, -1]
        return (ik, cdf)

    def run_knn(self, rs, pis, xgal, xrand, kneighbors, periodic, nthread = 32, leafsize = 32):
        """
        run the knns calculator

        Parameters
        ----------
        rs : array_like
            transverse radii to evaluate knns at.
        pis : array_like
            line of sight radii to evaluate knns at.
        xgal : array_like, (N, 3)
            positions of galaxies
        xrand : array_like, (N, 3)
            positiions of query points, feed uniform randoms if RD, feed xgal if DD
        kneighbors : int 
            the largest k to evaluate, default 1
        nthread : int
            number of threads, default 32 
        periodic : list or np.ndarray of shape (3,) with periodic boxsizes along each axis.
            If any value is less than zero, the box is assumed to be non-periodic along all axes
        leafsize : int
            leaf size for kdtree. default 32. 
            
        Returns
        -------
        knns_out : array_like
            final knn array in shape (k, len(rs), len(pis))
        """
        # Check that 3D positions are given
        assert xgal.shape[1] == 3
        assert xrand.shape[1] == 3

        # Convert to double for extra speed
        rs       = np.float32(rs)
        pis      = np.float32(pis)
        xgal     = np.float32(xgal)
        xrand    = np.float32(xrand)
        periodic = np.float32(periodic)

        xgal = np.array(xgal, order="C") # data
        xrand = np.array(xrand, order="C") # queries

        # Prep periodic array, if a number, make it an array
        periodic = np.array(periodic)
        if (len(periodic.shape) == 0) or (periodic.shape[0] < 3):
            periodic = np.ones(3) * periodic
        assert periodic.shape==(3,), 'Boxsize should have shape (3,)'

        # Do periodic wrap again in case float64->float32 conversion broke the box
        #xgal = np.mod(xgal, periodic)

        # Construct kDtree and query it
        dis_t, dis_p = self.compute_knn_distances(
                            data=xgal, 
                            query=xrand, 
                            k=kneighbors, 
                            nthread=nthread, 
                            periodic=periodic,
                            leafsize=leafsize
                        )            

        assert dis_t.shape == dis_p.shape

        # tabulate the pairs into knn histograms
        knns_out = self.calc_cdf_hist(rs, pis, dis_t, dis_p)

        return knns_out


@njit(parallel = True)
def convert_rppi(disi, xgal, xrand, k): 
    """
    Convert 3d distances to transverse and line of sight
    """
    dis_trans = -np.ones((len(xrand), len(k)), dtype = np.float32)
    dis_par = -np.ones((len(xrand), len(k)), dtype = np.float32)
    for ik in prange(len(k)):
        ineighs = disi[:, ik]
        delta_pos = xgal[ineighs] - xrand
        delta_trans = np.sqrt(delta_pos[:, 0]**2 + delta_pos[:, 1]**2)
        delta_par = np.absolute(delta_pos[:, 2])
        dis_trans[:, ik] = delta_trans          
        dis_par[:, ik] = delta_par 
    return dis_trans, dis_par
