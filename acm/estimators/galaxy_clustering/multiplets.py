import logging
from collections import defaultdict
from pathlib import Path

import lsstypes
import matplotlib.pyplot as plt
import numpy as np
from lsstypes.base import ObservableTree
from lsstypes.external import from_pycorr
from pycorr import TwoPointCorrelationFunction
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from acm.estimators.galaxy_clustering.backends import EstimatorBackend
from acm.typing import LsstypeObject

from .base import BaseEstimator

logger = logging.getLogger(__name__)

MULTIPLET_NAMES = {1: "singlet", 2: "pair", 3: "triplet", 4: "quadruplet"}


class GalaxyMultiplets(BaseEstimator):
    """
    Galaxy multiplets (singlets, pairs, triplets, ...) and their cross-correlation with the full sample.

    Two galaxies are linked if their 3D separation is < ``r_max`` and their separation
    perpendicular to the line of sight is < ``r_perp_max``. Multiplets are the connected
    components of the resulting graph; singlets are galaxies with no link.

    The geometry follows the inputs, as in the other estimators:

    - box (no randoms): periodic boundaries with ``backend.boxsize``, fixed ``los`` axis,
      natural estimator with analytic randoms.
    - cutsky (randoms given): Cartesian positions with the observer at the origin, midpoint
      line of sight, Davis-Peebles estimator (randoms on the galaxy side only).
    """

    def __init__(
        self,
        backend: str | EstimatorBackend,
        data_positions: np.ndarray,
        randoms_positions: np.ndarray | None = None,
        data_weights: np.ndarray | None = None,
        randoms_weights: np.ndarray | None = None,
        r_max: float = 7.0,
        r_perp_max: float = 1.05,
        los: str = "z",
        **kwargs,
    ) -> None:
        """
        Initialize the multiplets estimator.

        Parameters
        ----------
        backend: str | EstimatorBackend
            The backend to use for the estimator.
        data_positions: np.ndarray
            Cartesian positions of data galaxies, of shape (N, 3).
        randoms_positions: np.ndarray, optional
            Cartesian positions of randoms, of shape (M, 3). If given, cutsky mode is used.
        data_weights: np.ndarray, optional
            Weights for data galaxies, of shape (N,). Only used in the cross-correlation.
        randoms_weights: np.ndarray, optional
            Weights for randoms, of shape (M,).
        r_max: float, optional
            Maximum 3D separation for linking galaxies. Defaults to 7 Mpc/h.
        r_perp_max: float, optional
            Maximum separation perpendicular to the line of sight. Defaults to 1.05 Mpc/h.
        los: str, optional
            Line-of-sight axis in box mode, one of 'x', 'y', 'z'. Ignored in cutsky mode,
            which uses the pair midpoint. Defaults to 'z'.
        **kwargs
            Additional keyword arguments for the backend.
        """
        super().__init__(
            backend,
            data_positions=data_positions,
            randoms_positions=randoms_positions,
            data_weights=data_weights,
            randoms_weights=randoms_weights,
            **kwargs,
        )
        if los not in ("x", "y", "z"):
            raise ValueError(f"Invalid los: {los}. Must be 'x', 'y', or 'z'.")
        self.r_max = r_max
        self.r_perp_max = r_perp_max
        self.los = los
        self._multiplets = None

    @property
    def cutsky(self) -> bool:
        """Whether the estimator runs in cutsky mode (randoms provided)."""
        return self.randoms_positions is not None

    @property
    def multiplets(self) -> dict[int, np.ndarray]:
        """Positions of each multiplet class, keyed by multiplicity (1 = singlets)."""
        if self._multiplets is None:
            raise ValueError(
                "Multiplets not identified yet, call identify_multiplets() first."
            )
        return self._multiplets

    def _positions(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Return data positions in the linking frame, and the periodic boxsize (None for cutsky)."""
        positions = np.asarray(self.data_positions, dtype=float)
        if self.cutsky:
            return positions, None
        boxsize = np.asarray(self.backend.boxsize, dtype=float)
        return positions % boxsize, boxsize  # wrap e.g. boxes centred on 0 into [0, L)

    # ------------------------------------------------------------------
    # Multiplet identification
    # ------------------------------------------------------------------
    def identify_multiplets(self, max_multiplicity: int = 4) -> dict[int, np.ndarray]:
        """
        Link galaxies into multiplets and store the positions of each class.

        Parameters
        ----------
        max_multiplicity: int, optional
            Largest multiplet size to keep as its own class. Defaults to 4 (quadruplets).

        Returns
        -------
        dict[int, np.ndarray]
            Positions keyed by multiplicity: galaxy positions for singlets (1),
            group centers for larger multiplets.
        """
        positions, boxsize = self._positions()
        n_galaxies = len(positions)
        logger.info(
            f"Finding pairs with r_max={self.r_max}, r_perp_max={self.r_perp_max}"
        )
        pairs = (
            self._find_pairs_cutsky(positions)
            if self.cutsky
            else self._find_pairs_box(positions, boxsize)
        )

        in_pairs = np.unique(pairs.ravel())
        singlet_ids = np.setdiff1d(np.arange(n_galaxies), in_pairs)
        groups_list = self._form_multiplets(pairs, n_galaxies)
        centers, sizes = self._compute_group_centers(positions, groups_list, boxsize)

        self._multiplets = {1: positions[singlet_ids]}
        for n in range(2, max_multiplicity + 1):
            self._multiplets[n] = centers[sizes == n]
        logger.info(
            f"{len(in_pairs) / n_galaxies * 100:.2f}% of galaxies are in multiplets. Counts: "
            + ", ".join(
                f"{MULTIPLET_NAMES.get(n, n)}={len(p)}"
                for n, p in self._multiplets.items()
            )
        )
        return self._multiplets

    def _find_pairs_box(self, positions: np.ndarray, boxsize: np.ndarray) -> np.ndarray:
        """Find linked pairs in a periodic box, with a fixed line of sight."""
        perp_axes = [i for i, ax in enumerate("xyz") if ax != self.los]
        kd_tree = cKDTree(positions, boxsize=boxsize)
        all_pairs = kd_tree.query_pairs(r=self.r_max, output_type="ndarray")
        diff = np.abs(positions[all_pairs[:, 0]] - positions[all_pairs[:, 1]])
        diff = np.where(diff > boxsize / 2, boxsize - diff, diff)  # minimum image
        r_perp2 = np.sum(diff[:, perp_axes] ** 2, axis=1)
        return all_pairs[r_perp2 < self.r_perp_max**2]

    def _find_pairs_cutsky(self, positions: np.ndarray) -> np.ndarray:
        """Find linked pairs in cutsky geometry, using the pair midpoint as line of sight."""
        kd_tree = cKDTree(positions)
        all_pairs = kd_tree.query_pairs(r=self.r_max, output_type="ndarray")
        a = positions[all_pairs[:, 0]]
        b = positions[all_pairs[:, 1]]
        sep = a - b
        mid = (a + b) / 2
        mid_unit = mid / np.linalg.norm(mid, axis=1, keepdims=True)
        r_perp2 = np.sum(sep**2, axis=1) - np.sum(sep * mid_unit, axis=1) ** 2
        return all_pairs[r_perp2 < self.r_perp_max**2]

    @staticmethod
    def _form_multiplets(pairs: np.ndarray, n_galaxies: int) -> list[list[int]]:
        """Group linked galaxies into connected components (multiplets)."""
        adj = csr_matrix(
            (np.ones(len(pairs), dtype=int), (pairs[:, 0], pairs[:, 1])),
            shape=(n_galaxies, n_galaxies),
        )
        _, labels = connected_components(csgraph=adj, directed=False)
        groups = defaultdict(set)
        for a, b in pairs:
            groups[labels[a]].update((a, b))
        return [sorted(groups[g]) for g in sorted(groups)]

    @staticmethod
    def _periodic_mean(coords: np.ndarray, boxsize: np.ndarray) -> np.ndarray:
        """Mean of positions in a periodic box (minimum image around the first one), wrapped to [0, L)."""
        shifted = coords - np.round((coords - coords[0]) / boxsize) * boxsize
        return shifted.mean(axis=0) % boxsize

    def _compute_group_centers(
        self,
        positions: np.ndarray,
        groups_list: list[list[int]],
        boxsize: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Center (periodic mean in box mode, Cartesian mean in cutsky) and size of each group."""
        centers = np.zeros((len(groups_list), 3))
        sizes = np.zeros(len(groups_list), dtype=int)
        for i, ids in enumerate(groups_list):
            coords = positions[ids]
            centers[i] = (
                coords.mean(axis=0)
                if boxsize is None
                else self._periodic_mean(coords, boxsize)
            )
            sizes[i] = len(ids)
        return centers, sizes

    # ------------------------------------------------------------------
    # BaseEstimator API
    # ------------------------------------------------------------------
    def compute(self, edges: tuple[np.ndarray, np.ndarray], **kwargs) -> ObservableTree:
        """
        Cross-correlate each multiplet class with the full galaxy sample in (rp, pi).

        Parameters
        ----------
        edges: tuple[np.ndarray, np.ndarray]
            (rp_edges, pi_edges) of the correlation function.
        **kwargs
            Additional keyword arguments for :class:`pycorr.TwoPointCorrelationFunction`
            (e.g. ``nthreads``).

        Returns
        -------
        ObservableTree
            One :class:`lsstypes.Count2Correlation` per non-empty multiplet class, labelled
            by ``multiplets``. Use ``.project()`` on a leaf (or ``load(..., project=True)``)
            to get :math:`w_p(r_p)`.
        """
        multiplets = self.multiplets  # raises if not identified yet
        positions, boxsize = self._positions()
        if self.cutsky:
            geometry = dict(
                randoms_positions2=self.randoms_positions,
                randoms_weights2=self.randoms_weights,
                los="midpoint",
                estimator="davispeebles",
            )
        else:
            # Without randoms, pycorr uses analytic RR in a periodic box
            geometry = dict(boxsize=boxsize, los=self.los)

        leaves, labels = [], []
        for n, pos in multiplets.items():
            label = MULTIPLET_NAMES.get(n, f"{n}-plet")
            if len(pos) == 0:
                logger.warning(f"No {label}s found, skipping.")
                continue
            logger.info(f"Computing {label} cross-correlation ({len(pos)} objects)")
            correlation = TwoPointCorrelationFunction(
                "rppi",
                edges,
                data_positions1=pos,
                data_positions2=positions,
                data_weights2=self.data_weights,
                position_type="pos",
                **geometry,
                **kwargs,
            )
            leaves.append(self._to_lsstypes(correlation))
            labels.append(label)

        if not leaves:
            raise ValueError("No multiplets found, nothing to correlate.")
        attrs = dict(
            name=self.__class__.__name__,
            r_max=self.r_max,
            r_perp_max=self.r_perp_max,
            los="midpoint" if self.cutsky else self.los,
            cutsky=self.cutsky,
        )
        return ObservableTree(leaves, multiplets=labels, attrs=attrs)

    @staticmethod
    def _to_lsstypes(
        correlation: TwoPointCorrelationFunction,
    ) -> lsstypes.Count2Correlation:
        """Convert a pycorr correlation to lsstypes, with a clear error for unsupported Davis-Peebles."""
        try:
            return from_pycorr(correlation)
        except AssertionError as exc:
            # TODO: remove once lsstypes supports Davis-Peebles (https://github.com/adematti/lsstypes/issues/6)
            if getattr(correlation, "name", None) == "davispeebles":
                raise NotImplementedError(
                    "Cutsky multiplets use the Davis-Peebles estimator, which the installed lsstypes "
                    "does not support yet. See https://github.com/adematti/lsstypes/issues/6."
                ) from exc
            raise

    @staticmethod
    def load(filename: str | Path, project: bool = False, **kwargs) -> LsstypeObject:
        """
        Load a multiplets result from file.

        Parameters
        ----------
        filename: str | Path
            Path to the file containing the result.
        project: bool, optional
            Whether to project each leaf onto :math:`w_p(r_p)`. Defaults to False.
        **kwargs
            Additional keyword arguments for :meth:`lsstypes.Count2Correlation.project`.

        Returns
        -------
        lsstypes.ObservableTree
            The loaded result, optionally projected.
        """
        obj = lsstypes.read(filename)
        if project:
            leaves = [leaf.project(**kwargs) for leaf in obj.flatten(level=1)]
            obj = ObservableTree(leaves, multiplets=obj.multiplets, attrs=obj.attrs)
        return obj

    @staticmethod
    def plot(
        obj: LsstypeObject,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot :math:`r_p w_p(r_p)` for each multiplet class.

        Parameters
        ----------
        obj: LsstypeObject
            Result of :meth:`compute` or :meth:`load` (projected or not).
        fig: plt.Figure, optional
            Figure to plot on. If None, a new one is created.
        ax: plt.Axes, optional
            Axes to plot on. If None, a new one is created.
        **kwargs
            Additional keyword arguments for :meth:`matplotlib.pyplot.plot`.

        Returns
        -------
        fig, ax: tuple[plt.Figure, plt.Axes]
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
            ax.set_ylabel(r"$r_p\,w_p(r_p)$ [(Mpc/$h$)$^2$]")
        for label in obj.multiplets:
            leaf = obj.get(multiplets=label)
            if isinstance(leaf, lsstypes.Count2Correlation):
                leaf = leaf.project()
            rp = leaf.coords("rp")
            ax.plot(rp, rp * leaf.value(), label=label, **kwargs)
        ax.legend()
        return fig, ax
