import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from .base import BaseEstimator


class KthNearestNeighbor(BaseEstimator):
    """Class to compute the knns."""

    def compute_knn_distances(
        self,
        data: np.ndarray,
        query: np.ndarray,
        k: int | list[int],
        periodic: np.ndarray,
        nthread: int = 1,
        leafsize: int = 32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pair finding with scipy ckdtree."""
        # Make sure k in an array
        if isinstance(k, int):
            k = [k]

        # If any value is < 0, don't use periodic boxes
        if np.any(periodic <= 0):
            xtree = cKDTree(data, leafsize=leafsize)
            boxsize_for_conversion = np.array(
                [np.inf, np.inf, np.inf], dtype=np.float32
            )
        else:
            xtree = cKDTree(data, leafsize=leafsize, boxsize=periodic)
            boxsize_for_conversion = periodic.astype(np.float32)

        # Query the tree (this is parallel)
        _, disi = xtree.query(query, k=k, workers=nthread)

        # Conversion into 2D should be done separately. Careful with boundary!
        dis_trans, dis_par = convert_rppi(disi, data, query, k, boxsize_for_conversion)

        return dis_trans, dis_par

    def calc_cdf_hist(self, rs: np.ndarray, pis: np.ndarray, dis_t: np.ndarray, dis_p: np.ndarray) -> np.ndarray:
        """
        2D histogram wrapper function.

        Note: rs and pis should have shape (len(k), num_bins)
        """
        # Edges of bins are passed
        cdfs = np.zeros(
            (dis_t.shape[1], rs.shape[1] - 1, pis.shape[1] - 1), dtype=np.float32
        )

        # Compute each hist separately
        for ik in range(dis_t.shape[1]):
            h, _, _ = np.histogram2d(dis_t[:, ik], dis_p[:, ik], bins=(rs[ik], pis[ik]))
            cdf_ik = np.cumsum(np.cumsum(h, axis=0), axis=1)
            cdfs[ik] = cdf_ik / len(dis_t)

        return cdfs

    def run_knn(
        self,
        rs: np.ndarray,
        pis: np.ndarray,
        xgal: np.ndarray,
        xrand: np.ndarray,
        kneighbors: int | list[int],
        periodic: np.ndarray,
        nthread: int = 32,
        leafsize: int = 32,
    ) -> np.ndarray:
        """
        Run the knns calculator.

        Parameters
        ----------
        rs : array_like
            transverse radii to evaluate knns at. Can be a 1D array or a 2D array of shape (len(k), num_bins) to
            specify separate binning for separate ks.
        pis : array_like
            line of sight radii to evaluate knns at. Can be a 1D array or a 2D array of shape (len(k), num_bins) to
            specify separate binning for separate ks.
        xgal : array_like, (N, 3)
            positions of galaxies
        xrand : array_like, (N, 3)
            positiions of query points, feed uniform randoms if RD, feed xgal if DD
        kneighbors : int
            a list of ints, for which ks to evaluate cdfs
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
        if xgal.shape[1] != 3 or xrand.shape[1] != 3:
            raise ValueError("xgal and xrand should have shape (N, 3)")

        if isinstance(kneighbors, int):
            kneighbors = [kneighbors]

        # Convert to double for extra speed
        rs = np.array(rs, dtype=np.float32)
        pis = np.array(pis, dtype=np.float32)
        xgal = np.array(xgal, dtype=np.float32)
        xrand = np.array(xrand, dtype=np.float32)
        periodic = np.array(periodic, dtype=np.float32)

        xgal = np.array(xgal, order="C")  # data
        xrand = np.array(xrand, order="C")  # queries

        # Prep periodic array, if a number, make it an array
        periodic = np.array(periodic)
        if (len(periodic.shape) == 0) or (periodic.shape[0] < 3):
            periodic = np.ones(3) * periodic
        if periodic.shape[0] != 3:
            raise ValueError("Periodic boxsize should have shape (3,)")

        # Do periodic wrap again in case float64->float32 conversion broke the box
        # xgal = np.mod(xgal, periodic)

        # Construct kDtree and query it
        dis_t, dis_p = self.compute_knn_distances(
            data=xgal,
            query=xrand,
            k=kneighbors,
            nthread=nthread,
            periodic=periodic,
            leafsize=leafsize,
        )

        if dis_t.shape != dis_p.shape:
            raise ValueError("Transverse and line of sight distances should have the same shape")

        # A bit of care about rs and pis bins. If (len(k), N) arrays are provided,
        # use them (binning for each k individually). If not, turn them into this shape
        if len(rs.shape) != len(pis.shape):
            raise ValueError("rs and pis should have the same number of dimensions")

        if len(rs.shape) == 2 and rs.shape[0] != pis.shape[0]:
            raise ValueError("If rs and pis are 2D, they should have the same number of rows (len(k))")
        if len(rs.shape) == 1:
            # FIXME: k is not defined here !!
            rs = np.stack([rs for i in range(len(kneighbors))], axis=0)
            pis = np.stack([pis for i in range(len(kneighbors))], axis=0)
        else:
            raise ValueError("Invalid shape for rs and pis")

        # tabulate the pairs into knn histograms
        knns_out = self.calc_cdf_hist(rs, pis, dis_t, dis_p)

        return knns_out


@njit(parallel=True)
def convert_rppi(
    disi: np.ndarray,
    xgal: np.ndarray,
    xrand: np.ndarray,
    k: list,
    length: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert indices of pairs to transverse and line of sight distances.

    This function should be used if computations are performed for periodic box!
    """
    dis_trans = -np.ones((len(xrand), len(k)), dtype=np.float32)
    dis_par = -np.ones((len(xrand), len(k)), dtype=np.float32)
    half_box = length / 2

    # prange over queries so all cores are used!
    for ik in range(len(k)):
        for i in prange(len(xrand)):  # ty:ignore[not-iterable]
            # Find right index
            neighb_idx = disi[i, ik]

            # Have to check periodicity separately for all components
            dx = xgal[neighb_idx, 0] - xrand[i, 0]
            dy = xgal[neighb_idx, 1] - xrand[i, 1]
            dz = xgal[neighb_idx, 2] - xrand[i, 2]

            # Check manually. If L = inf, the check fails, no wrapping happens!
            if np.abs(dx) > half_box[0]:
                if dx > 0:
                    dx -= length[0]
                else:
                    dx += length[0]

            if np.abs(dy) > half_box[1]:
                if dy > 0:
                    dy -= length[1]
                else:
                    dy += length[1]

            if np.abs(dz) > half_box[2]:
                if dz > 0:
                    dz -= length[2]
                else:
                    dz += length[2]

            # Compute the distances
            dis_trans[i, ik] = np.sqrt(dx * dx + dy * dy)
            dis_par[i, ik] = np.abs(dz)

    return dis_trans, dis_par
