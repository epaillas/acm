import logging
from collections.abc import Callable

import numpy as np
from cosmoprimo import Cosmology
from pandas import DataFrame

from acm.catalogs.dataclasses import Transform
from acm.catalogs.products import BaseGalaxyCatalog

logger = logging.getLogger(__name__)


# %% Pure transform functions that can be used in the pipeline
def _apply_rsd(data: DataFrame, los: str, hubble: float, az: float) -> DataFrame:
    """
    Apply RSD shift along the los axis.

    Parameters
    ----------
    data : DataFrame
        Galaxy data containing position and velocity columns.
    los : str
        Line-of-sight axis, one of 'x', 'y', 'z'.
    hubble : float
        Hubble parameter H(z) in km/s/(Mpc/h) for the simulation cosmology.
    az : float
        Scale factor a(z) at the snapshot's redshift.

    Returns
    -------
    DataFrame
        Transformed galaxy data with RSD applied.
    """
    data = data.copy()
    v_col = f"v{los}"
    data[los] = data[los] + data[v_col] / (hubble * az)
    return data


def _apply_ap(
    data: DataFrame,
    los: str,
    q_par: float,
    q_perp: float,
    pos_columns: tuple[str],
) -> DataFrame:
    """
    Apply AP scaling: q_par along los, q_perp along transverse axes.

    Parameters
    ----------
    data : DataFrame
        Galaxy data containing position columns.
    los : str
        Line-of-sight axis, one of 'x', 'y', 'z'.
    q_par : float
        AP scaling factor along the line-of-sight.
    q_perp : float
        AP scaling factor along the transverse directions.
    pos_columns : tuple[str]
        Names of the position columns, e.g. ('x', 'y', 'z').

    Returns
    -------
    DataFrame
        Transformed galaxy data with AP scaling applied.
    """
    data = data.copy()
    for ax in pos_columns:
        data[ax] = data[ax] * (q_par if ax == los else q_perp)
    return data


def _apply_downsample(
    data: DataFrame,
    tracer: str,
    n_gal: int | None,
    f_gal: float | None,
    nbar: float | None,
    boxsize: Callable[[], np.ndarray] | None = None,
) -> DataFrame:
    """
    Randomly downsample a tracer DataFrame.

    Parameters
    ----------
    data : DataFrame
        Galaxy data for a specific tracer.
    tracer : str
        Tracer name, used for logging.
    n_gal : int, optional
        Target number of galaxies.
    f_gal : float, optional
        Fraction of galaxies to keep, between 0 and 1.
    nbar : float, optional
        Target number density in (Mpc/h)^-3.
    boxsize : callable, optional
        Function that returns the current boxsize, needed to compute target n_gal when downsampling by nbar.

    Returns
    -------
    DataFrame
        Downsampled galaxy data.

    Raises
    ------
    ValueError
        If not exactly one of n_gal, f_gal or nbar is provided.
    ValueError
        If boxsize is needed but not provided.
    """
    provided = sum(p is not None for p in (n_gal, f_gal, nbar))
    if provided != 1:
        raise ValueError("Exactly one of n_gal,f_gal or nbar must be provided.")

    n_current = len(data)
    if f_gal is not None:
        n_target = int(n_current * f_gal)
    elif n_gal is not None:
        n_target = n_gal
    else:  # nbar
        if boxsize is None:
            raise ValueError(
                "boxsize function must be provided when downsampling by nbar."
            )
        # Callable to get current boxsize, which may include AP scaling
        volume = np.prod(boxsize())  
        n_target = int(nbar * volume)

    if n_target >= n_current:
        logger.warning(
            f"Target n_gal={n_target} >= current n_gal={n_current} for tracer '{tracer}', skipping downsample."
        )
        return data

    return data.sample(n=n_target).reset_index(drop=True)


# %% GalaxyCatalog classes
class SnapshotCatalog(BaseGalaxyCatalog):
    """Snapshot-based galaxy catalog at a fixed redshift."""

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
            f"tracers={list(self.tracers.keys())})"
        )

    @property
    def boxsize(self) -> np.ndarray[tuple[int]]:
        """Size of the simulation box in each dimension, common to all tracers."""
        if "ap" in self._transforms:
            ap = self._transforms["ap"]
            los = ap.kwargs["los"]
            factors = np.array(
                [self.q_par if ax == los else self.q_perp for ax in self.pos_columns],
                dtype=float,
            )
            return self._boxsize * factors
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

    def _check_data_columns(self, data: DataFrame) -> bool:
        """Check that the position and velocity columns for a tracer are present in the data before assignment."""
        required_columns = set(self.pos_columns + self.vel_columns)
        missing_columns = required_columns - set(data.columns)
        return missing_columns == set()

    def rsd(self, los: str = "z") -> None:
        """
        Add redshift-space distortion transform to the pipeline.

        Shifts positions along the line-of-sight axis by v_los / (H(z) * a(z)).

        Parameters
        ----------
        los : str
            Line-of-sight axis, one of 'x', 'y', 'z'.
        """
        if los not in self.pos_columns:
            raise ValueError(f"los must be one of {self.pos_columns}, got '{los}'.")
        self._add_transform(
            Transform(
                name="rsd",
                func=_apply_rsd,
                kwargs={"los": los, "hubble": self.hubble, "az": self.az},
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
                kwargs={
                    "tracer": tracer,
                    "n_gal": n_gal,
                    "f_gal": f_gal,
                    "nbar": nbar,
                    "boxsize": lambda: self.boxsize,  # evaluated at application time
                },
            )
        )

    def _ngal(self, tracer: str) -> int:
        """Return the total number of galaxies for a specific tracer."""
        d = self.get_tracer_data(tracer)
        return len(d)

    @property
    def ngal(self) -> int:
        """Total number of galaxies in the catalog across all tracers."""
        if not self.tracers:
            raise RuntimeError("No tracers loaded in the catalog, cannot compute ngal.")
        return sum(self._ngal(tracer) for tracer in self.tracers)

    def _nbar(self, tracer: str) -> float:
        """Return the number density of galaxies for a specific tracer."""
        n_gal = self._ngal(tracer)
        boxsize = self.boxsize
        volume = np.prod(boxsize)
        return n_gal / volume if volume > 0 else 0.0

    @property
    def nbar(self) -> float:
        """Number density of galaxies in the entire catalog."""
        if not self.tracers:
            raise RuntimeError("No tracers loaded in the catalog, cannot compute nbar.")
        volume = np.prod(self.boxsize)
        return self.ngal / volume if volume > 0 else 0.0

    # TODO: Add box replocation with padding for cutsky creation ?

    # TODO: Save and load methods (e.g. to hdf5)
