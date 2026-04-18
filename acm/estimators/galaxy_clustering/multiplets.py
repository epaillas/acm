import logging
from collections import defaultdict

import numpy as np
from pycorr import AnalyticTwoPointCounter, NaturalTwoPointEstimator, TwoPointCounter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


class GalaxyMultiplets:
    """
    Class to identify galaxy multiplets (singlets, pairs, triplets, quadruplets, etc.)
    based on 3D spatial separation and projected separation criteria, and compute
    their cross-correlation with the full galaxy sample.

    A multiplet is a group of galaxies that are connected by proximity:
    - Two galaxies are linked if their 3D separation is < r_max and their projected
      separation (perpendicular to LOS) is < r_perp_max.
    - Groups are formed by connecting all linked pairs using graph-based clustering.

    Parameters
    ----------
    r_max : float
        Maximum 3D separation for linking galaxies (default: 7 Mpc/h)
    r_perp_max : float
        Maximum projected separation perpendicular to line-of-sight (default: 1.05 Mpc/h)
    los : str
        Line-of-sight direction: 'x', 'y', or 'z' (default: 'z')
    nthreads : int
        Number of threads for correlation function computation (default: 4)
    """

    def __init__(self, r_max=7.0, r_perp_max=1.05, los="z", nthreads=4):
        self.logger = logging.getLogger("GalaxyMultiplets")
        self.logger.info("Initializing GalaxyMultiplets.")

        self.r_max = r_max
        self.r_perp_max = r_perp_max
        self.los = los
        self.nthreads = nthreads

        # Results storage
        self.singlet_ids = None
        self.singlet_coords = None
        self.groups_list = None
        self.group_centers = None
        self.group_sizes = None
        self.pair_coords = None
        self.triplet_coords = None
        self.quadruplet_coords = None

    def _get_los_axes(self):
        """Get the perpendicular axes for the given line-of-sight."""
        if self.los == "z":
            return [0, 1]
        elif self.los == "y":
            return [0, 2]
        elif self.los == "x":
            return [1, 2]
        else:
            raise ValueError(f"Invalid los: {self.los}. Must be 'x', 'y', or 'z'")

    def find_pairs(self, positions, boxsize):
        """
        Find all galaxy pairs within the specified 3D and projected separations.

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            Galaxy positions in Cartesian coordinates
        boxsize : float or array_like, shape (3,)
            Periodic box size(s)

        Returns
        -------
        pairs : ndarray, shape (M, 2)
            Array of galaxy pair indices
        """
        self.logger.info(
            f"Finding pairs with r_max={self.r_max}, r_perp_max={self.r_perp_max}"
        )

        if np.isscalar(boxsize):
            boxsize = np.array([boxsize, boxsize, boxsize])
        else:
            boxsize = np.asarray(boxsize)

        perp_axes = self._get_los_axes()

        # Build KD-tree and find all pairs within r_max
        kd_tree = cKDTree(positions, boxsize=boxsize[0])
        all_pairs = kd_tree.query_pairs(r=self.r_max, output_type="ndarray")

        self.logger.info(f"Found {len(all_pairs)} pairs within r_max={self.r_max}")

        # Separate boundary and non-boundary pairs
        sep_3d = np.sqrt(
            np.sum(
                (positions[all_pairs[:, 0]] - positions[all_pairs[:, 1]]) ** 2, axis=1
            )
        )
        boundary_mask = sep_3d > 100  # Heuristic for boundary detection

        # Non-boundary pairs: simple projected separation check
        non_boundary_pairs = all_pairs[~boundary_mask]
        if len(non_boundary_pairs) > 0:
            proj_sep_nb = np.sqrt(
                np.sum(
                    (
                        positions[non_boundary_pairs[:, 0]][:, perp_axes]
                        - positions[non_boundary_pairs[:, 1]][:, perp_axes]
                    )
                    ** 2,
                    axis=1,
                )
            )
            non_boundary_pairs = non_boundary_pairs[proj_sep_nb < self.r_perp_max]

        # Boundary pairs: account for periodic wrapping
        boundary_pairs = all_pairs[boundary_mask]
        valid_boundary = []

        for i, (a, b) in enumerate(boundary_pairs):
            diff = np.abs(positions[a] - positions[b])
            # Apply periodic boundary conditions
            diff = np.where(diff > boxsize / 2, boxsize - diff, diff)
            proj_sep = np.sqrt(np.sum(diff[perp_axes] ** 2))
            if proj_sep < self.r_perp_max:
                valid_boundary.append(i)

        if valid_boundary:
            boundary_pairs = boundary_pairs[valid_boundary]
        else:
            boundary_pairs = np.empty((0, 2), dtype=int)

        # Combine all valid pairs
        pairs = (
            np.vstack([non_boundary_pairs, boundary_pairs])
            if len(non_boundary_pairs) > 0
            else boundary_pairs
        )

        self.logger.info(f"Found {len(pairs)} pairs satisfying all criteria")
        self.logger.info(
            f"  Non-boundary: {len(non_boundary_pairs)}, Boundary: {len(boundary_pairs)}"
        )

        return pairs

    def form_multiplets(self, pairs, n_galaxies):
        """
        Form multiplets by connecting pairs using graph-based clustering.

        Parameters
        ----------
        pairs : array_like, shape (M, 2)
            Array of galaxy pair indices
        n_galaxies : int
            Total number of galaxies

        Returns
        -------
        groups_list : list of lists
            List where each element is a list of galaxy indices in that group
        """
        self.logger.info("Forming multiplets via connected components")

        # Create adjacency matrix
        row = pairs[:, 0]
        col = pairs[:, 1]
        data = np.ones(len(pairs), dtype=int)
        adj_matrix = csr_matrix((data, (row, col)), shape=(n_galaxies, n_galaxies))

        # Find connected components
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False)

        # Group galaxies by component label
        groups = defaultdict(set)
        for a, b in pairs:
            g = labels[a]
            groups[g].add(a)
            groups[g].add(b)

        # Convert to sorted lists
        groups_list = [sorted(list(groups[g])) for g in sorted(groups)]

        self.logger.info(f"Formed {len(groups_list)} multiplets")

        return groups_list

    def _periodic_mean(self, coords, boxsize):
        """
        Compute the mean position of a group accounting for periodic boundaries.

        Parameters
        ----------
        coords : array_like, shape (M, D)
            Coordinates of objects in one group
        boxsize : float
            Box size

        Returns
        -------
        mean : ndarray, shape (D,)
            Mean position wrapped into [0, boxsize)
        """
        ref = coords[0]
        shifted = coords.copy()

        # Minimum-image convention: shift by +/- boxsize to be closest to reference
        shifted -= np.round((shifted - ref) / boxsize) * boxsize

        mean = shifted.mean(axis=0)

        # Wrap back into [0, boxsize)
        mean = mean % boxsize

        return mean

    def compute_group_centers(self, positions, groups_list, boxsize):
        """
        Compute the center-of-mass for each multiplet group.

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            Galaxy positions
        groups_list : list of lists
            List of galaxy groups
        boxsize : float or array_like
            Box size(s)

        Returns
        -------
        group_centers : ndarray, shape (n_groups, 3)
            Center position for each group
        group_sizes : ndarray, shape (n_groups,)
            Number of galaxies in each group
        """
        if np.isscalar(boxsize):
            boxsize = np.array([boxsize, boxsize, boxsize])

        n_groups = len(groups_list)
        group_centers = np.zeros((n_groups, 3))
        group_sizes = np.zeros(n_groups, dtype=int)

        for i, obj_ids in enumerate(groups_list):
            coords = positions[obj_ids]
            group_centers[i] = self._periodic_mean(coords, boxsize[0])
            group_sizes[i] = len(obj_ids)

        return group_centers, group_sizes

    def identify_multiplets(self, positions, boxsize):
        """
        Complete pipeline to identify all multiplets in a galaxy catalog.

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            Galaxy positions
        boxsize : float or array_like
            Box size(s)

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'singlet_ids': indices of singlet galaxies
            - 'singlet_coords': coordinates of singlet galaxies
            - 'groups_list': list of multiplet groups
            - 'group_centers': center positions of multiplets
            - 'group_sizes': sizes of multiplets
            - 'pair_coords': centers of pairs (size=2)
            - 'triplet_coords': centers of triplets (size=3)
            - 'quadruplet_coords': centers of quadruplets (size=4)
        """
        self.logger.info("Starting multiplet identification")

        positions = np.asarray(positions)
        n_galaxies = len(positions)

        # Find pairs
        pairs = self.find_pairs(positions, boxsize)

        # Identify singlets (galaxies not in any pair)
        all_pair_ids = np.unique(np.concatenate([pairs[:, 0], pairs[:, 1]]))
        all_ids = np.arange(n_galaxies)
        self.singlet_ids = np.setdiff1d(all_ids, all_pair_ids)
        self.singlet_coords = positions[self.singlet_ids]

        self.logger.info(
            f"Found {len(self.singlet_ids)} singlets ({len(self.singlet_ids) / n_galaxies * 100:.2f}%)"
        )
        self.logger.info(
            f"{len(all_pair_ids) / n_galaxies * 100:.2f}% of galaxies are in multiplets"
        )

        # Form multiplets
        self.groups_list = self.form_multiplets(pairs, n_galaxies)

        # Compute group centers
        self.group_centers, self.group_sizes = self.compute_group_centers(
            positions, self.groups_list, boxsize
        )

        # Separate by multiplet size
        self.pair_coords = self.group_centers[self.group_sizes == 2]
        self.triplet_coords = self.group_centers[self.group_sizes == 3]
        self.quadruplet_coords = self.group_centers[self.group_sizes == 4]

        self.logger.info("Multiplet counts:")
        self.logger.info(f"  Pairs: {len(self.pair_coords)}")
        self.logger.info(f"  Triplets: {len(self.triplet_coords)}")
        self.logger.info(f"  Quadruplets: {len(self.quadruplet_coords)}")

        return {
            "singlet_ids": self.singlet_ids,
            "singlet_coords": self.singlet_coords,
            "groups_list": self.groups_list,
            "group_centers": self.group_centers,
            "group_sizes": self.group_sizes,
            "pair_coords": self.pair_coords,
            "triplet_coords": self.triplet_coords,
            "quadruplet_coords": self.quadruplet_coords,
        }

    def compute_cross_correlation(
        self, multiplet_coords, all_positions, boxsize, edges
    ):
        """
        Compute the cross-correlation between multiplet centers and all galaxies.

        Parameters
        ----------
        multiplet_coords : array_like, shape (M, 3)
            Multiplet center positions
        all_positions : array_like, shape (N, 3)
            All galaxy positions
        boxsize : float or array_like
            Box size(s)
        edges : tuple of arrays
            (rp_edges, pi_edges) for the correlation function bins

        Returns
        -------
        estimator : NaturalTwoPointEstimator
            Correlation function estimator
        """
        if np.isscalar(boxsize):
            boxsize = np.array([boxsize, boxsize, boxsize])

        DD_counts = TwoPointCounter(
            "rppi",
            edges,
            positions1=[
                multiplet_coords[:, 0],
                multiplet_coords[:, 1],
                multiplet_coords[:, 2],
            ],
            positions2=[all_positions[:, 0], all_positions[:, 1], all_positions[:, 2]],
            boxsize=boxsize,
            los=self.los,
            position_type="xyz",
            engine="corrfunc",
            nthreads=self.nthreads,
        )

        RR_counts = AnalyticTwoPointCounter(
            "rppi",
            edges,
            size1=len(multiplet_coords),
            size2=len(all_positions),
            boxsize=boxsize,
            los=self.los,
        )

        estimator = NaturalTwoPointEstimator(D1D2=DD_counts, R1R2=RR_counts)

        return estimator

    def compute_all_cross_correlations(self, positions, boxsize, edges, pimax=None):
        """
        Compute cross-correlations for all multiplet types (singlets, pairs, triplets, quadruplets).

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            All galaxy positions
        boxsize : float or array_like
            Box size(s)
        edges : tuple of arrays
            (rp_edges, pi_edges) for the correlation function bins
        pimax : float, optional
            Maximum pi for wp calculation. If None, returns 2D correlation

        Returns
        -------
        correlations : dict
            Dictionary with keys 'singlet', 'pair', 'triplet', 'quadruplet',
            each containing the correlation function values
        """
        if self.singlet_coords is None:
            raise ValueError("Must run identify_multiplets() first")

        self.logger.info("Computing cross-correlations for all multiplet types")

        correlations = {}

        # Singlets
        if len(self.singlet_coords) > 0:
            self.logger.info(
                f"Computing singlet cross-correlation ({len(self.singlet_coords)} objects)"
            )
            est_singlet = self.compute_cross_correlation(
                self.singlet_coords, positions, boxsize, edges
            )
            correlations["singlet"] = est_singlet(pimax=pimax, return_sep=False)

        # Pairs
        if self.pair_coords is not None and len(self.pair_coords) > 0:
            self.logger.info(
                f"Computing pair cross-correlation ({len(self.pair_coords)} objects)"
            )
            est_pair = self.compute_cross_correlation(
                self.pair_coords, positions, boxsize, edges
            )
            correlations["pair"] = est_pair(pimax=pimax, return_sep=False)

        # Triplets
        if self.triplet_coords is not None and len(self.triplet_coords) > 0:
            self.logger.info(
                f"Computing triplet cross-correlation ({len(self.triplet_coords)} objects)"
            )
            est_triplet = self.compute_cross_correlation(
                self.triplet_coords, positions, boxsize, edges
            )
            correlations["triplet"] = est_triplet(pimax=pimax, return_sep=False)

        # Quadruplets
        if self.quadruplet_coords is not None and len(self.quadruplet_coords) > 0:
            self.logger.info(
                f"Computing quadruplet cross-correlation ({len(self.quadruplet_coords)} objects)"
            )
            est_quadruplet = self.compute_cross_correlation(
                self.quadruplet_coords, positions, boxsize, edges
            )
            correlations["quadruplet"] = est_quadruplet(pimax=pimax, return_sep=False)

        return correlations

    def get_summary_table(self, edges, correlations):
        """
        Create a summary table with separation bins and correlation functions.

        Parameters
        ----------
        edges : tuple of arrays
            (rp_edges, pi_edges) used for the correlation function
        correlations : dict
            Dictionary of correlation functions from compute_all_cross_correlations()

        Returns
        -------
        table : ndarray
            Array with columns: [r, wp_singlet, wp_pair, wp_triplet, wp_quadruplet]
        """
        rp_edges = edges[0]
        nbins = len(rp_edges) - 1
        r = (rp_edges[:nbins] + rp_edges[1:]) / 2

        data = [r]
        for key in ["singlet", "pair", "triplet", "quadruplet"]:
            if key in correlations:
                data.append(correlations[key])
            else:
                data.append(np.zeros(nbins))

        table = np.vstack(data).T

        return table
