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


def find_pairs_box(
    positions: np.ndarray,
    boxsize: np.ndarray,
    r_max: float,
    r_perp_max: float,
    los: str,
) -> np.ndarray:
    """Find linked pairs in a periodic box, with a fixed line of sight."""
    if los not in ("x", "y", "z"):
        raise ValueError(f"Invalid los: {los}. Must be 'x', 'y', or 'z'.")
    perp_axes = [i for i, ax in enumerate("xyz") if ax != los]
    kd_tree = cKDTree(positions, boxsize=boxsize)
    all_pairs = kd_tree.query_pairs(r=r_max, output_type="ndarray")
    diff = np.abs(positions[all_pairs[:, 0]] - positions[all_pairs[:, 1]])
    diff = np.where(diff > boxsize / 2, boxsize - diff, diff)  # minimum image
    r_perp2 = np.sum(diff[:, perp_axes] ** 2, axis=1)
    return all_pairs[r_perp2 < r_perp_max**2]


def find_pairs_cutsky(
    positions: np.ndarray,
    r_max: float,
    r_perp_max: float,
) -> np.ndarray:
    """Find linked pairs in cutsky geometry, using the pair midpoint as line of sight."""
    kd_tree = cKDTree(positions)
    all_pairs = kd_tree.query_pairs(r=r_max, output_type="ndarray")
    a = positions[all_pairs[:, 0]]
    b = positions[all_pairs[:, 1]]
    sep = a - b
    mid = (a + b) / 2
    mid_unit = mid / np.linalg.norm(mid, axis=1, keepdims=True)
    r_perp2 = np.sum(sep**2, axis=1) - np.sum(sep * mid_unit, axis=1) ** 2
    return all_pairs[r_perp2 < r_perp_max**2]


def form_multiplets(pairs: np.ndarray, n_galaxies: int) -> list[list[int]]:
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


def group_center(coords: np.ndarray, boxsize: np.ndarray | None = None) -> np.ndarray:
    """Mean position of a group: periodic (minimum-image) mean in a box, Cartesian mean otherwise."""
    if boxsize is None:
        return coords.mean(axis=0)
    # Bring every member next to the first one, average, then wrap the mean back into [0, L)
    shifted = coords - boxsize * np.round((coords - coords[0]) / boxsize)
    return shifted.mean(axis=0) % boxsize


def compute_group_centers(
    positions: np.ndarray,
    groups_list: list[list[int]],
    boxsize: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Center and size of each group."""
    centers = [group_center(positions[ids], boxsize) for ids in groups_list]
    centers = np.array(centers).reshape(-1, 3)
    sizes = np.array([len(ids) for ids in groups_list], dtype=int)
    return centers, sizes


def identify_multiplets(
    positions: np.ndarray,
    boxsize: np.ndarray | None,
    r_max: float = 7.0,
    r_perp_max: float = 1.05,
    max_multiplicity: int = 4,
    los: str = "z",
) -> dict[int, np.ndarray]:
    """
    Link galaxies into multiplets and store the positions of each class.

    Parameters
    ----------
    r_max: float, optional
        Maximum 3D separation for linking galaxies. Defaults to 7 Mpc/h.
    r_perp_max: float, optional
        Maximum separation perpendicular to the line of sight. Defaults to 1.05 Mpc/h.
    los: str, optional
        Line-of-sight axis in box mode, one of 'x', 'y', 'z'.
        Ignored in cutsky mode, which uses the pair midpoint. Defaults to 'z'.
    max_multiplicity: int, optional
        Largest multiplet size to keep as its own class. Defaults to 4 (quadruplets).

    Returns
    -------
    dict[int, np.ndarray]
        Positions keyed by multiplicity: galaxy positions for singlets (1), group centers for larger multiplets.
    """
    n_galaxies = len(positions)
    logger.info(f"Finding pairs with r_max={r_max}, r_perp_max={r_perp_max}")
    if boxsize is None:
        pairs = find_pairs_cutsky(positions, r_max, r_perp_max)
    else:
        positions = positions % boxsize  # wrap e.g. boxes centred on 0 into [0, L)
        pairs = find_pairs_box(positions, boxsize, r_max, r_perp_max, los=los)

    in_pairs = np.unique(pairs.ravel())
    singlet_ids = np.setdiff1d(np.arange(n_galaxies), in_pairs)
    groups_list = form_multiplets(pairs, n_galaxies)
    centers, sizes = compute_group_centers(positions, groups_list, boxsize)

    multiplets = {1: positions[singlet_ids]}
    for n in range(2, max_multiplicity + 1):
        multiplets[n] = centers[sizes == n]
    logger.info(
        f"{len(in_pairs) / n_galaxies * 100:.2f}% of galaxies are in multiplets. Counts: "
        + ", ".join(
            f"{MULTIPLET_NAMES.get(n, n)}={len(p)}" for n, p in multiplets.items()
        )
    )
    return multiplets


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

    @property
    def is_cutsky(self) -> bool:
        """Whether the estimator runs in cutsky mode (randoms provided)."""
        return self.randoms_positions is not None

    def compute(
        self,
        r_max: float = 7.0,
        r_perp_max: float = 1.05,
        los: str = "z",
        **kwargs,
    ) -> ObservableTree:
        """
        Cross-correlate each multiplet class with the full galaxy sample in (rp, pi).

        Parameters
        ----------
        r_max: float, optional
            Maximum 3D separation for linking galaxies. Defaults to 7 Mpc/h.
        r_perp_max: float, optional
            Maximum separation perpendicular to the line of sight. Defaults to 1.05 Mpc/h.
        los: str, optional
            Line-of-sight axis in box mode, one of 'x', 'y', 'z'. Ignored in cutsky mode,
            which uses the pair midpoint. Defaults to 'z'.
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
        positions = self.data_positions
        if self.is_cutsky:
            boxsize = None
            geometry = dict(
                randoms_positions2=self.randoms_positions,
                randoms_weights2=self.randoms_weights,
                los="midpoint",
                estimator="davispeebles",
            )
        else:
            # Without randoms, pycorr uses analytic RR in a periodic box
            boxsize = np.asarray(self.backend.boxsize, dtype=float)
            positions = positions % boxsize  # wrap e.g. boxes centred on 0 into [0, L)
            geometry = dict(boxsize=boxsize, los=los)
        multiplets = identify_multiplets(
            positions, boxsize, r_max=r_max, r_perp_max=r_perp_max, los=los
        )

        leaves, labels = [], []
        for n, pos in multiplets.items():
            label = MULTIPLET_NAMES.get(n, f"{n}-plet")
            if len(pos) == 0:
                logger.warning(f"No {label}s found, skipping.")
                continue
            logger.info(f"Computing {label} cross-correlation ({len(pos)} objects)")
            correlation = TwoPointCorrelationFunction(
                "rppi",
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
            r_max=r_max,
            r_perp_max=r_perp_max,
            los="midpoint" if self.is_cutsky else los,
            is_cutsky=self.is_cutsky,
        )
        return ObservableTree(leaves, multiplets=labels, attrs=attrs)

    # TODO: remove once lsstypes supports Davis-Peebles (https://github.com/adematti/lsstypes/issues/6)
    @staticmethod
    def _to_lsstypes(
        correlation: TwoPointCorrelationFunction,
    ) -> lsstypes.Count2Correlation:
        """Convert a pycorr correlation to lsstypes, with a clear error for unsupported Davis-Peebles."""
        try:
            return from_pycorr(correlation)
        except AssertionError as exc:
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
