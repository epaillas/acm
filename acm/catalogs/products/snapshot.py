import logging
from typing import Self

import h5py
import numpy as np
import pandas as pd
from cosmoprimo import Cosmology
from numpy.random import RandomState

from acm.catalogs.dataclasses import Transform
from acm.catalogs.products import BaseGalaxyCatalog
from acm.catalogs.products.transforms import _apply_ap, _apply_downsample, _apply_rsd

logger = logging.getLogger(__name__)


# %% GalaxyCatalog classes
class SnapshotCatalog(BaseGalaxyCatalog):
    """
    Snapshot-based galaxy catalog at a fixed redshift.

    Expects galaxy positions in comoving Cartesian coordinates (x, y, z) in Mpc/h
    and velocities (vx, vy, vz) in km/s. Provides methods to apply RSD and AP transforms,
    as well as downsampling.
    """

    pos_columns = ("x", "y", "z")
    vel_columns = ("vx", "vy", "vz")

    def __init__(
        self,
        redshift: float,
        cosmo: Cosmology,
        cosmo_fid: Cosmology,
        boxsize: float | list[float] | np.ndarray,
    ) -> None:
        """
        Initialize the galaxy catalog with the given redshift and cosmologies.

        Extends BaseGalaxyCatalog by adding a redshift attribute and related cosmological properties.

        Parameters
        ----------
        redshift : float
            Redshift of the snapshot.
        cosmo : cosmoprimo.Cosmology, optional
            The cosmology for the simulation.
        cosmo_fid : cosmoprimo.Cosmology, optional
            The fiducial cosmology.
        boxsize : float or list of 3 floats
            Size of the simulation box in Mpc/h. Can be a single float (same for all dimensions) or a list of three floats for each dimension.
        """
        super().__init__(cosmo, cosmo_fid)
        self.redshift = redshift
        self.boxsize = boxsize  # uses the setter

    def __repr__(self) -> str:
        """Provide a string representation of the galaxy catalog, including redshift and tracer information."""
        return (
            f"{self.__class__.__name__}("
            f"redshift={self.redshift}, "
            f"boxsize={self.boxsize}, "
            f"tracers={list(self.tracers)})"
        )

    @property
    def boxsize(self) -> np.ndarray:
        """Size of the simulation box in each dimension, common to all tracers."""
        if "ap" in self._transforms:
            ap = self._transforms["ap"]
            los = ap.kwargs["los"]
            factors = np.array(
                [self.q_par if ax == los else self.q_perp for ax in self.pos_columns],
                dtype=float,
            )
            return self._boxsize / factors
        return self._boxsize

    @boxsize.setter
    def boxsize(self, value: float | list[float] | np.ndarray) -> None:
        """Set boxsize, broadcasting a scalar to all three dimensions."""
        value = np.asarray(value, dtype=float)
        if value.ndim == 0:  # scalar
            self._boxsize = np.full(3, value)
        elif value.shape == (3,):
            self._boxsize = value
        else:
            raise ValueError(
                f"boxsize must be a scalar or a 3-element array, got shape {value.shape}."
            )

    @property
    def az(self) -> float:
        """Scale factor at this snapshot's redshift."""
        return 1.0 / (1.0 + self.redshift)

    @property
    def hubble(self) -> float:
        """H(z) in km/s/(Mpc/h) for the simulation cosmology."""
        return 100.0 * self.cosmo.efunc(self.redshift)

    @property
    def hubble_fid(self) -> float:
        """H(z) in km/s/(Mpc/h) for the fiducial cosmology."""
        return 100.0 * self.cosmo_fid.efunc(self.redshift)

    @property
    def q_par(self) -> float:
        """AP parallel scaling factor."""
        return self.hubble_fid / self.hubble

    @property
    def q_perp(self) -> float:
        """AP perpendicular scaling factor."""
        return self.cosmo.angular_diameter_distance(
            self.redshift
        ) / self.cosmo_fid.angular_diameter_distance(self.redshift)

    def _check_data_columns(self, data: pd.DataFrame) -> bool:
        """Check that the position and velocity columns for a tracer are present in the data before assignment."""
        required_columns = set(self.pos_columns + self.vel_columns)
        missing_columns = required_columns - set(data.columns)
        return missing_columns == set()

    def rsd(self, los: str = "z", wrap: bool = True, offset: float = 0.0) -> None:
        """
        Add redshift-space distortion transform to the pipeline.

        Shifts positions along the line-of-sight axis by v_los / (H(z) * a(z)).

        Parameters
        ----------
        los : str
            Line-of-sight axis, one of 'x', 'y', 'z'.
        wrap : bool, optional
            If True, apply a boxsize periodic wrapping after RSD shifts. Default is True.
        offset : float, optional
            Offset to correct for periodic wrapping. Should be set to boxsize/2
            if positions are centered around zero, or 0 if positions are in [0, boxsize].
            Default is 0.0.
        """
        if los not in self.pos_columns:
            raise ValueError(f"los must be one of {self.pos_columns}, got '{los}'.")
        if "ap" in self._transforms:
            logger.warning(
                "AP transform exists: RSD transform will be registered with a distorted boxsize and may yield unexpected results. "
            )
        L = self.boxsize[self.pos_columns.index(los)]  # For periodic wrapping
        self._add_transform(
            Transform(
                name="rsd",
                func=_apply_rsd,
                kwargs={
                    "los": los,
                    "hubble": self.hubble,
                    "az": self.az,
                    "wrap": L if wrap else 0,
                    "offset": offset,
                },
            )
        )

    def ap(self, los: str = "z") -> None:
        """
        Add Alcock-Paczynski scaling transform to the pipeline.

        Scales positions along the line-of-sight by q_par and perpendicular
        axes by q_perp.

        Parameters
        ----------
        los : str
            Line-of-sight axis, one of 'x', 'y', 'z'.
        """
        if los not in self.pos_columns:
            raise ValueError(f"los must be one of {self.pos_columns}, got '{los}'.")
        self._add_transform(
            Transform(
                name="ap",
                func=_apply_ap,
                kwargs={
                    "los": los,
                    "q_par": self.q_par,
                    "q_perp": self.q_perp,
                    "pos_columns": self.pos_columns,
                },
            )
        )

    def downsample(
        self,
        tracer: str,
        n_gal: int | None = None,
        f_gal: float | None = None,
        nbar: float | None = None,
        seed: RandomState | int | None = None,
    ) -> None:
        """
        Add a downsampling transform for a specific tracer.

        Exactly one of n_gal, f_gal or nbar must be provided.

        Parameters
        ----------
        tracer : str
            Tracer to downsample.
        n_gal : int, optional
            Target number of galaxies.
        f_gal : float, optional
            Fraction of galaxies to keep, between 0 and 1.
        nbar : float, optional
            Target number density in (Mpc/h)^-3.

        Raises
        ------
        ValueError
            If not exactly one of n_gal, f_gal or nbar is provided.
        """
        provided = sum(p is not None for p in (n_gal, f_gal, nbar))
        if provided != 1:
            raise ValueError("Exactly one of n_gal, f_gal or nbar must be provided.")
        self._add_transform(
            Transform(
                name=f"downsample_{tracer}",
                func=_apply_downsample,
                tracer=tracer,
                kwargs={
                    "tracer": tracer,  # passed for logging purposes
                    "n_gal": n_gal,
                    "f_gal": f_gal,
                    "nbar": nbar,
                    "volume": lambda: np.prod(self.boxsize),  # evaluated at runtime
                    "seed": seed,
                },
            )
        )

    def _nbar(self, *tracers: str) -> float:
        """Return the number density of galaxies for specific tracers, or the full catalog if tracer is None."""
        n_gal = self._ngal(*tracers)
        boxsize = self.boxsize
        volume = np.prod(boxsize)
        return n_gal / volume if volume > 0 else 0.0

    @property
    def nbar(self) -> float:
        """Number density of galaxies in the entire catalog."""
        return self._nbar()

    def positions(self, raw: bool = False) -> pd.DataFrame:
        """
        Get the positions of galaxies in the full catalog.

        Parameters
        ----------
        raw : bool, optional
            If True, return the raw positions before any transformations.
            If False, return the positions after applying all transformations.
            Defaults to False.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the positions of galaxies.
        """
        if not self.tracers:
            raise RuntimeError(
                "No tracers loaded in the catalog, cannot get positions."
            )
        data = self.get_tracer_data(*self.tracers.keys(), raw=raw)
        pos = data[list(self.pos_columns)]
        return pos

    def _save_attrs(self, f: h5py.File) -> None:
        f.attrs["redshift"] = self.redshift
        f.attrs["boxsize"] = self._boxsize  # raw, pre-AP boxsize

        # Cosmology parameters (just in case; not used to reconstruct the class)
        f.attrs["cosmo_h"] = self.hubble
        f.attrs["cosmo_fid_h"] = self.hubble_fid
        f.attrs["az"] = self.az
        f.attrs["q_par"] = self.q_par
        f.attrs["q_perp"] = self.q_perp

    @classmethod
    def _from_attrs(cls, attrs: dict, cosmo: Cosmology, cosmo_fid: Cosmology) -> Self:
        return cls(
            redshift=float(attrs["redshift"]),
            cosmo=cosmo,
            cosmo_fid=cosmo_fid,
            boxsize=np.array(attrs["boxsize"]),
        )


class RandomSnapshotCatalog(SnapshotCatalog):
    """
    Snapshot catalog with randomized galaxy positions.

    Replaces true galaxy positions with uniform random positions within the
    simulation box. Intended for null tests and covariance estimation.
    Velocities and RSD are not available for random catalogs.
    """

    pos_columns = ("x", "y", "z")
    vel_columns = ()  # No velocities for random catalogs

    @classmethod
    def from_snapshot(cls, catalog: SnapshotCatalog, seed: int | None = None) -> Self:
        """
        Create a random catalog from an existing SnapshotCatalog.

        Inherits redshift, cosmology, boxsize and tracers from the source catalog,
        replacing all position data with uniform random draws.

        Parameters
        ----------
        catalog : SnapshotCatalog
            Source catalog to copy metadata and tracer counts from.
        seed : int | None
            Random seed for reproducibility.
        """
        ntracers = len(catalog.tracers)
        seeds = np.random.SeedSequence(seed).spawn(ntracers)

        random_catalog = cls(
            redshift=catalog.redshift,
            cosmo=catalog.cosmo,
            cosmo_fid=catalog.cosmo_fid,
            boxsize=catalog._boxsize,
        )
        for i, (tracer_name, tracer) in enumerate(catalog.tracers.items()):
            n_gal = len(catalog._data[tracer_name])
            random_catalog.set_tracer_data(
                tracer,
                cls._random_positions(
                    n_gal,
                    catalog._boxsize,
                    seed=seeds[i],
                ),
            )
        return random_catalog

    @staticmethod
    def _random_positions(
        n_gal: int,
        boxsize: np.ndarray,
        seed: int | np.random.SeedSequence | None,
    ) -> pd.DataFrame:
        """
        Generate a pandas DataFrame of uniform random positions within the box.

        Parameters
        ----------
        n_gal : int
            Number of random galaxies to generate.
        boxsize : np.ndarray
            Box dimensions in each axis, used to scale the random positions.
        seed : int | np.random.SeedSequence | None
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed=seed)
        return pd.DataFrame(
            {
                "x": rng.uniform(0, boxsize[0], n_gal),
                "y": rng.uniform(0, boxsize[1], n_gal),
                "z": rng.uniform(0, boxsize[2], n_gal),
            }
        )

    def rsd(self, los: str = "z", wrap: bool = False, offset: float = 0.0) -> None:
        """Raise an error if RSD is attempted on a random catalog, since velocities are not defined."""
        raise NotImplementedError("RSD is not available for random catalogs.")


# %% Helpers
def boundary_check(
    positions: np.ndarray,
    boxsize: np.ndarray | list[float] | float,
    check_in_float32: bool = True,
    center_at_zero: bool = False,
) -> None:
    """
    Check that all positions are within the specified box boundaries.

    Parameters
    ----------
    positions : np.ndarray
        Positions of the galaxies in the catalog.
        Should be of shape (N_galaxies, 3).
    boxsize : np.ndarray | list[float] | float
        Size of the periodic box.
        Can be a single float (same size for all dimensions) or an array of shape (3,).
    check_in_float32 : bool, optional
        If True, all checks are performed in single precision (float32).
        Default is True.
    center_at_zero : bool, optional
        If True, positions are required to be in the range [-L_i/2, L_i/2) for each axis.
        If False, positions should be in [0, L_i). Default is False.

    Raises
    ------
    ValueError
        If any of the positions fall outside the specified box boundaries.
    """
    # Pick precision
    _dtype = np.float32 if check_in_float32 else np.float64
    positions = positions.astype(_dtype)

    boxsize = np.atleast_1d(np.array(boxsize, dtype=_dtype))
    if len(boxsize) == 1:
        boxsize = np.repeat(boxsize, 3)
    elif len(boxsize) != 3:
        raise ValueError(
            f"boxsize should be a float or an array of shape (3,), but got {boxsize.shape}"
        )

    # Pick right and left edges for each dimension
    offset = boxsize / 2 if center_at_zero else 0.0
    L = np.array([0.0, 0.0, 0.0], dtype=_dtype) - offset
    R = boxsize - offset

    # Do checks
    for i in range(positions.shape[1]):
        left_bound_check = np.all(positions[:, i] >= L[i])
        right_bound_check = np.all(positions[:, i] < R[i])

        min_left = np.min(positions[:, i])
        max_right = np.max(positions[:, i])

        # Build error message:
        em = ""
        if not left_bound_check:
            em += f"{min_left!r} falls out of the box on the left edge {L[i]!r} along the {i}-th axis. "
        if not right_bound_check:
            em += f"{max_right!r} falls out of the box on the right edge {R[i]!r} along the {i}-th axis."
        if em:
            raise ValueError(em)
