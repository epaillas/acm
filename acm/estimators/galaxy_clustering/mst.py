import logging

import matplotlib.pyplot as plt
import mistreeplus as mist  # https://github.com/knaidoo29/mistreeplus
import numpy as np

from acm.utils.plotting import set_plot_style

from .base import BaseEstimator

logger = logging.getLogger(__name__)


class MinimumSpanningTree(BaseEstimator):
    """
    Class to compute the minimum spanning tree statistics, following https://arxiv.org/abs/1907.00989
    but using a new set of powerful percolation statistics (default).
    """

    def __init__(self, **kwargs):
        logger.info("Initializing MinimumSpanningTree.")
        super().__init__(**kwargs)

    def setup(
        self,
        sigmaJ: float,
        boxsize: float | np.ndarray,
        Nthpoint: int,
        origin: float = 0.0,
        split: int = 1,
        iterations: int = 1,
        quartiles: int = 50,
    ):
        """
        Setup the minimum spanning tree percolation statistics variables.

        Parameters
        ----------
        sigmaJ : float
            The jitter dispersion scale (see https://arxiv.org/abs/2410.06202) to apply a
            point process smoothing on the positions of points.
        boxsize : float or numpy.ndarray
            Size of the simulation box.
        Nthpoint : int
            Percolation statistics N-Point, i.e. the number of linking edges from each node to
            compute the length and straight line distance.
        origin : float, optional
            The origin of the box.
        split : int, optional
            For improved speed, you may want to divide the box, to compute the MST in each sub-box
            and then aggregate the MST statistics together.
        iterations : int, optional
            The number of iterations to perform the MST calculation over, this is to marginalise
            over the stochasticity from the jitter dispersion.
        quartiles : int, optional
            The number of bins used for the MST distributions.
        """
        self.sigmaJ = sigmaJ
        if np.isscalar(boxsize):
            _boxsize = [boxsize, boxsize, boxsize]
        else:
            _boxsize = boxsize
        self.boxsize = np.asarray(_boxsize)
        self.Nthpoint = Nthpoint
        if np.isscalar(origin):
            _origin = [origin, origin, origin]
        else:
            _origin = origin
        self.origin = np.asarray(_origin)
        self.split = split
        self.iterations = iterations
        self.quartiles = quartiles

    def _periodic(self, x, y, z):
        """
        Imposes periodic boundary condition on the input data.

        Parameters
        ----------
        x, y, z : array_like
            Coordinate positions.
        """
        for i, dim in enumerate([x, y, z]):
            cond = np.where(dim < 0.0)[0]
            dim[cond] += self.boxsize[i]
            cond = np.where(dim >= self.boxsize[i])[0]
            dim[cond] -= self.boxsize[i]

        return x, y, z

    def _smooth(self, x, y, z, periodic=True):
        """
        Adds jitter dispersion to apply point-process smoothing to the input coordinates..

        Parameters
        ----------
        x, y, z : array_like
            Coordinate positions.
        """
        x += self.sigmaJ * np.random.normal(size=len(x))
        y += self.sigmaJ * np.random.normal(size=len(x))
        z += self.sigmaJ * np.random.normal(size=len(x))
        if periodic:
            x, y, z = self._periodic(x, y, z)
        return x, y, z

    def _get_even_splits(self, length, N):
        """
        Finds the intervals to split any array into roughly N segments.

        Parameters
        ----------
        length : int
            Length of the array.
        N : int
            The number of splits.

        Returns
        -------
        split1, split2 : array_like
            Arrays corresponding to the minimum and maximum index for each N split.
        """
        split_equal = length / N
        split_floor = np.floor(split_equal)
        split_remain = split_equal - split_floor
        counts = split_floor * np.ones(N)
        counts[: int(np.round(split_remain * N, decimals=0))] += 1
        counts = counts.astype("int")
        splits = np.zeros(N + 1, dtype="int")
        splits[1:] = np.cumsum(counts)
        split1 = splits[:-1]
        split2 = splits[1:]
        return split1, split2

    def get_percolation_statistics(self, data_pos, useknn=True, k=20):
        """
        Computes the percolation statistics on the input galaxy data.

        Parameters
        ----------
        data_pos : array_like
            Coordinates of the input data set.

        Returns
        -------
        mstdict : dict
            Dictionary containing the percolations statistics.
        """
        x, y, z = data_pos[:, 0], data_pos[:, 1], data_pos[:, 2]
        # remove origin
        x, y, z = (data_pos[:, i] - self.origin[i] for i in range(3))
        # apply point-process smoothing
        if self.sigmaJ > 0.0:
            x, y, z = self._smooth(x, y, z, periodic=True)

        # Create subbox regions for segmenting the box.
        xedges = np.linspace(0.0, self.boxsize[0], self.split + 1)
        yedges = np.linspace(0.0, self.boxsize[1], self.split + 1)
        zedges = np.linspace(0.0, self.boxsize[2], self.split + 1)
        ixs = np.arange(self.split)
        ixs, iys, izs = np.meshgrid(ixs, ixs, ixs, indexing="ij")
        ixs = ixs.flatten()
        iys = iys.flatten()
        izs = izs.flatten()

        mstdict = {}
        for niter in range(self.iterations):
            for j in range(len(ixs)):
                xmin, xmax = xedges[ixs[j]], xedges[ixs[j] + 1]
                ymin, ymax = yedges[iys[j]], yedges[iys[j] + 1]
                zmin, zmax = zedges[izs[j]], zedges[izs[j] + 1]

                # create mask for only data-points within the subbox.
                inbox = np.where(
                    (x >= xmin)
                    & (x < xmax)
                    & (y >= ymin)
                    & (y < ymax)
                    & (z >= zmin)
                    & (z < zmax)
                )[0]

                if useknn:
                    # construct kNN tesselation
                    _graph = mist.graph.construct_knn3D(x[inbox], y[inbox], z[inbox], k)
                else:
                    # construct delaunay tesselation
                    _graph = mist.graph.construct_del3D(x[inbox], y[inbox], z[inbox])

                # graph -> minimum spanning tree
                mst_graph = mist.mst.construct_mst(_graph)

                # convert from scipy sparse graph matrix to just edge index and weights (euclidean distance)
                edge_idx, wei = mist.graph.graph2data(mst_graph)
                # Get the number of nodes.
                Nnodes = len(x[inbox])
                # construct edge index dictionary for fast edge weight finding.
                edge_dict = mist.tree.get_edge_dict(edge_idx, wei)
                # construct an adjacency tree/lists.
                adj_idx, adj_wei = mist.tree.get_adjacents(edge_idx, wei, Nnodes)
                # the 1-point MST edge length distribution.
                weiNpt = wei

                # sort the MST statistics to compute the mean in each quartile -- more efficient way of computing PDF/CDFs.
                sortID = np.argsort(weiNpt)
                split1, split2 = self._get_even_splits(len(weiNpt), self.quartiles)

                # compute the 1-point MST PDF
                meanperbin = np.array(
                    [
                        np.mean(weiNpt[sortID[split1[i] : split2[i]]])
                        for i in range(len(split1))
                    ]
                )

                # store the 1-point MST PDF into the vectors dictionary
                if niter == 0 and j == 0:
                    mstdict["mst1pt"] = meanperbin
                else:
                    mstdict["mst1pt"] += meanperbin

                # compute the N-point MST PDFs
                for N in range(2, self.Nthpoint + 1):
                    # get the percolation paths from each point into the tree for the N-th placed point.
                    if N == 2:
                        percpaths = mist.tree.perc_from_all_by_N(adj_idx, N)
                    else:
                        percpaths = mist.tree.perc_from_all_by_N(
                            adj_idx, N, percpaths=percpaths
                        )
                    # the N-point MST edge length distribution.
                    weiNpt = mist.tree.percpath2weight(percpaths, edge_dict)
                    # find the percolation end points
                    percends = mist.tree.percpath2percends(percpaths)
                    # the straight line distance to the end point
                    endNpt = mist.tree.percend_dist3D(
                        x[inbox], y[inbox], z[inbox], percends
                    )

                    # sort the MST statistics to compute the mean in each quartile -- more efficient way of computing PDF/CDFs.
                    sortID = np.argsort(weiNpt)
                    split1, split2 = self._get_even_splits(len(weiNpt), self.quartiles)

                    # compute the N-point MST PDF
                    meanperbin = np.array(
                        [
                            np.mean(weiNpt[sortID[split1[i] : split2[i]]])
                            for i in range(len(split1))
                        ]
                    )
                    # compute the N-endpoint MST PDF
                    endperbin = np.array(
                        [
                            np.mean(endNpt[sortID[split1[i] : split2[i]]])
                            for i in range(len(split1))
                        ]
                    )

                    # store the N-point MST PDF into the vectors dictionary
                    if niter == 0 and j == 0:
                        mstdict["mst%ipt" % N] = meanperbin
                        mstdict["end%ipt" % N] = endperbin
                    else:
                        mstdict["mst%ipt" % N] += meanperbin
                        mstdict["end%ipt" % N] += endperbin

        # Normalise by iterations and splits
        mstdict["mst1pt"] /= self.iterations * len(ixs)
        for N in range(2, self.Nthpoint + 1):
            mstdict["mst%ipt" % N] /= self.iterations * len(ixs)
            mstdict["end%ipt" % N] /= self.iterations * len(ixs)

        return mstdict

    @set_plot_style
    def plot_percolation_statistics(
        self, mstdict, cmap="viridis", figsize=(10, 4), fname=None
    ):
        """
        Plot the percolation statistics.

        Parameters
        ----------
        mstdict : dict
            Dictionary containing the percolations statistics.
        cmap : str, optional
            Colormap.
        figsize : list, optional
            Figsize aspect ratio.
        fname : str, optional
            Optional to save the plot output.
        """
        percentedge = np.linspace(0.0, 100.0, len(mstdict["mst1pt"]) + 1)
        percentmids = 0.5 * (percentedge[1:] + percentedge[:-1])
        colormap = plt.cm.get_cmap(cmap)
        plt.figure(figsize=figsize)
        plt.plot(
            percentmids,
            mstdict["mst1pt"],
            linestyle="-",
            color=colormap((1 - 1) / (10 - 1)),
        )
        xticks = [50.0]
        xlabel = ["%i" % 1]
        for N in range(2, self.Nthpoint + 1):
            plt.plot(
                100 * (N - 1) + percentmids,
                mstdict["mst%ipt" % N],
                linestyle="-",
                linewidth=2.0,
                color=colormap((N - 1) / (self.Nthpoint - 1)),
            )
            plt.plot(
                100 * (N - 1) + percentmids,
                mstdict["end%ipt" % N],
                linestyle="--",
                linewidth=2.0,
                color=colormap((N - 1) / (self.Nthpoint - 1)),
            )
            plt.axvline(100 * (N - 1), color="k", linestyle=":")
            xticks.append(100.0 * N - 50.0)
            xlabel.append("%i" % N)
        plt.plot([], [], color="k", linestyle="-", label="Percolation Distance")
        plt.plot([], [], color="k", linestyle="--", label="End-to-End Distance")
        plt.xlim(0.0, 100 * self.Nthpoint)
        plt.ylim(0.0, None)
        plt.xticks(xticks, xlabel)
        plt.xlabel(r"MST Percolation Order")
        plt.ylabel(r"Percolation Length [$h^{-1}Mpc$]")
        plt.legend(loc="best", framealpha=1.0)
        if fname is not None:
            plt.savefig(fname, bbox_inches="tight")
        plt.show()
