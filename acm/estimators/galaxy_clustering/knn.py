import numpy as np
import logging
import scipy.spatial
from functools import partial
from numba import njit, types, jit, prange

import ray
from fast_histogram import histogram2d
import pyfnntw

from .base import BaseEnvironmentEstimator

class KthNearestNeighbor(BaseEnvironmentEstimator):
    """
    Class to compute the knns.
    """
    def __init__(self, **kwargs):

        self.logger = logging.getLogger('KthNearestNeighbor')
        self.logger.info('Initializing KthNearestNeighbor.')
        super().__init__(**kwargs)

    def is_linear_or_log(self, X):
        """
        check if binning is linear or log
        """
        diffs = np.diff(X)
        ratios = np.divide(diffs[:-1], diffs[1:])
        avg_diff = np.mean(diffs)
        avg_ratio = np.mean(ratios)
        std_diff = np.std(diffs)
        std_ratio = np.std(ratios)
        if abs(std_diff) < 1e-4:
            return "linear"
        elif abs(std_ratio) < 1e-4:
            return "log"
        else:
            return "neither"

    @njit(parallel = True)
    def convert_rppi(self, disi, xgal, xrand, k): 
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

    def VolumekNN_par_pimax_hist(self, xgal, xrand, pis, k = 1, nthread = 1, periodic = 0):
        """
        pair finding with scipy ckdtree
        """
        if isinstance(k, int): 
            k = [k] 

        xtree = scipy.spatial.cKDTree(xgal, boxsize=periodic, leafsize = 16)

        _, disi = xtree.query(xrand, k=k)

        dis_trans, dis_par = self.convert_rppi(disi, xgal, xrand, k)

        return dis_trans, dis_par

    def get_trans_par_fnntw(self, data, query, k, LS = 32, lbox = 1):
        """
        pair finding with pyfnntw
        """
        if lbox <= 0:
            xtree = pyfnntw.Treef32(data, leafsize=LS)
        else:
            lbox = np.float32(lbox)
            xtree = pyfnntw.Treef32(data, leafsize=LS, boxsize = np.array([lbox, lbox, lbox]))

        par, trans = xtree.query(query, k=k[-1], axis=2)

        return trans, par

    @ray.remote
    def calculate_cdfs(self, ik, dis_t, dis_p, rpbins, pibins, scaling):
        """
        tabulate pair distances into 2d histogram, accelerated with ray 
        """
        if scaling == 'linear':
            dist_hist2d_k = histogram2d(dis_t[:, ik], dis_p[:, ik], 
                            range=[[rpbins[0], rpbins[-1]], [pibins[0], pibins[-1]]], bins=[len(rpbins)-1, len(pibins)-1])
        else:
            dist_hist2d_k, _, _ = np.histogram2d(dis_t[:, ik], dis_p[:, ik], bins=(rpbins, pibins))

        dist_cdf2d_k = np.cumsum(np.cumsum(dist_hist2d_k, axis=0), axis=1)
        cdf = dist_cdf2d_k / dist_cdf2d_k[-1, -1]

    def calc_cdf_hist(self, rs, pis, dis_t, dis_p):
        """
        2d histogram wrapper function
        """
        cdfs = np.zeros((dis_t.shape[1], len(rs), len(pis)), dtype = np.float32)

        # are the bins lin or log
        rpbins = np.concatenate((np.zeros(1, dtype = np.float32), rs))
        pibins = np.concatenate((np.zeros(1, dtype = np.float32), pis))
        scaling_t = self.is_linear_or_log(rpbins)
        scaling_p = self.is_linear_or_log(pibins)

        assert scaling_t == scaling_p

        args_list = [(ik, dis_t, dis_p, rpbins, pibins, scaling_t) for ik in range(dis_t.shape[1])]
        results = ray.get([self.calculate_cdfs.remote(*args) for args in args_list])

        for ik, cdf in results:
            cdfs[ik] = cdf

        return cdfs

    def run_knn(self, rs, pis, xgal, xrand, kneighbors = 1, nthread = 32, periodic = 0, method = 'fnn', randdown = 1, LS = 32):
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
        periodic : 0 or 1
            0 for non-periodic boundaries, 1 for periodic boundaries, default 0
        method : str
            what method to use for computing kdtrees. 
            'fnn': a fast kD-tree library that aims to be one of the most, if not the most, performant parallel kNN libraries that exist. Recommended. 
            See https://pypi.org/project/pyfnntw/. 
            Otherwise, use standard ckdtree. 
        randdown : float
            downsampling factor for queries. default 1 (no downsampling).
        LS : int
            leaf size for fnn. default 32. 
            
        Returns
        -------
        knns_out : array_like
            final knn array in shape (k, len(rs), len(pis))
        """
        assert xgal.shape[1] == 3
        assert xrand.shape[1] == 3

        rs = np.float32(rs)
        pis = np.float32(pis)
        xgal = np.float32(xgal)
        xrand = np.float32(xrand)

        xgal = np.array(xgal, order="C") # data
        xrand = np.array(xrand, order="C") # queries

        # downsample the queries if requested
        if randdown > 1:
            xrand = xrand[np.random.choice(np.arange(len(xrand)), size = int(len(xrand)/randdown), replace = False)]

        # construct kdtrees and find pairs
        if method == 'fnn':
            dis_t, dis_p = self.get_trans_par_fnntw(xgal, xrand, 
                                kneighbors, lbox = periodic, LS = LS)
        else:
            dis_t, dis_p = self.VolumekNN_par_pimax_hist(xgal, xrand, pis, 
                                k=kneighbors, nthread = nthread, periodic = 0)            

        assert dis_t.shape == dis_p.shape

        # tabulate the pairs into knn histograms
        knns_out = self.calc_cdf_hist(rs, pis, dis_t, dis_p)

        return knns_out