import logging
from collections.abc import Callable
from typing import Self, override

import h5py
import healpy as hp
import numpy as np
import pandas as pd
from cosmoprimo import Cosmology
from numpy.random import RandomState
from scipy.interpolate import interp1d

from acm.catalogs.dataclasses import Transform
from acm.catalogs.products.base import BaseGalaxyCatalog
from acm.catalogs.products.transforms import _add_distance_column, _apply_downsample

logger = logging.getLogger(__name__)


def _fsky(ra: np.ndarray, dec: np.ndarray, nside: int = 256) -> float:
    """
    Estimate the sky fraction covered by a set of angular positions using HEALPix pixel counting.

    Pixels are assigned at the given resolution; the estimate improves with denser
    sampling and higher nside. For sparse or small-area catalogs, consider increasing nside.

    Parameters
    ----------
    ra : np.ndarray
        Right ascension in degrees, in [0, 360).
    dec : np.ndarray
        Declination in degrees, in [-90, 90].
    nside : int
        HEALPix resolution parameter. Must be a power of 2. Default is 256.

    Returns
    -------
    float
        Fraction of the sky covered, in [0, 1].
    """
    npix = hp.nside2npix(nside)
    phi = np.radians(ra)
    theta = np.radians(90 - dec)  # Convert dec to theta for HEALPix

    pix = hp.ang2pix(nside, theta, phi)
    unique_pix = np.unique(pix)
    fsky = len(unique_pix) / npix
    return fsky


def _shell_volume(cosmo: Cosmology, z: np.ndarray) -> np.ndarray:
    """
    Compute the full-sky comoving volume of redshift shells between consecutive redshift values.

    Parameters
    ----------
    cosmo : cosmology object
        Must expose ``comoving_radial_distance(z)`` returning distances in Mpc/h.
    z : np.ndarray
        Redshift bin edges of shape (n_bins + 1,).

    Returns
    -------
    np.ndarray
        Shell volumes of shape (n_bins,) in (Mpc/h)³. Full-sky; multiply by fsky to get the actual volume fraction.
    """
    d = cosmo.comoving_radial_distance(z)
    dv = 4 / 3 * np.pi * np.diff(d**3)
    return dv


class CutskyCatalog(BaseGalaxyCatalog):
    """
    Galaxy catalog with cutsky geometry and redshift evolution.

    Expects galaxy positions in spherical coordinates (ra, dec, z) with angular coordinates
    in degrees and redshift as a dimensionless quantity.

    A hp_res parameter can be provided at initialization to control the resolution
    of the HEALPix grid used for estimating sky coverage and n(z) interpolation.

    Heavy computations like fsky and n(z) interpolation are cached based on the current
    set of transforms to avoid redundant calculations when applying multiple transforms sequentially.
    Caches are automatically invalidated when transforms are added, removed, or reset.
    """

    pos_columns = ("ra", "dec", "z")

    def __init__(
        self,
        cosmo: Cosmology,
        cosmo_fid: Cosmology,
        hp_res: int = 256,
    ) -> None:
        """
        Initialise the CutskyCatalog.

        Parameters
        ----------
        cosmo : Cosmology
            True cosmology used for distance calculations.
        cosmo_fid : Cosmology
            Fiducial cosmology.
        hp_res : int
            HEALPix resolution (nside) for sky coverage estimation. Must be a
            power of 2. Higher values improve footprint accuracy at the cost of
            memory. Default is 256 (~0.05 deg² per pixel).
        """
        super().__init__(cosmo, cosmo_fid)
        self.hp_res = hp_res
        # Caches for expensive computations keyed by transform state
        self._fsky_cache: dict[tuple, float] = {}
        self._interpolate_nz_cache: dict[tuple, Callable[[float], float]] = {}

    def _check_data_columns(self, data: pd.DataFrame) -> bool:
        """
        Check that all required position columns are present in the data.

        Parameters
        ----------
        data : pd.DataFrame
            Galaxy data to validate.

        Returns
        -------
        bool
            True if all required columns are present, False otherwise.
        """
        required_columns = set(self.pos_columns)
        missing_columns = required_columns - set(data.columns)
        return missing_columns == set()

    def clear_caches(self) -> None:
        """
        Clear all cached computations (fsky and n(z) interpolators).

        Should be called manually if the underlying data is modified in a way
        that bypasses the transform pipeline, though under normal usage caches
        are invalidated automatically via the transform state key.
        """
        self._fsky_cache.clear()
        self._interpolate_nz_cache.clear()

    def _range(
        self,
        coord: str,
        *tracers: str,
        periodic_wrap: float | None = None,
    ) -> tuple[float, float]:
        """
        Compute the range of a coordinate from the data.

        Parameters
        ----------
        coord : str
            Coordinate to compute the range for (e.g., "ra", "dec", "z").
        *tracers : str
            Specific tracer names to compute the range for. If no tracers are specified, computes the range across all tracers.
        periodic_wrap : float | None
            If the coordinate is periodic (e.g., "ra"), specify the period to account for wrap-around in the range calculation.
        """
        tracer_names = tracers or tuple(self.tracers)
        all_values = self.get_tracer_data(*tracer_names, raw=True)[coord]
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        cout = [min_val, max_val]
        if periodic_wrap is not None:
            # Accounting for periodicity, with wrap-around if the range crosses the periodic boundary
            cout = np.mod(cout, periodic_wrap).tolist()
        return tuple(cout)

    def _zrange(self, *tracers: str) -> tuple[float, float]:
        """Return the redshift range of specified tracers, or the full catalog if tracer is None."""
        return self._range("z", *tracers)

    @property
    def zrange(self) -> tuple[float, float]:
        """Return the redshift range of the full catalog."""
        return self._zrange()

    @property
    def fsky(self) -> float:
        """Return the fraction of the sky covered by the catalog footprint."""
        # Use cache if available
        cache_key = (self.ngal, self._transform_state)
        if cache_key in self._fsky_cache:
            return self._fsky_cache[cache_key]

        angles = self.get_tracer_data(*self.tracers)[["ra", "dec"]]
        ra = angles["ra"]
        dec = angles["dec"]
        result = _fsky(ra, dec, nside=self.hp_res)
        self._fsky_cache[cache_key] = result
        return result

    @property
    def area(self) -> float:
        """Return the area of the catalog's footprint in square degrees."""
        fsky = self.fsky
        return fsky * 4 * np.pi * (180 / np.pi) ** 2  # Steradians to square degrees

    def _nbar(self, *tracers: str) -> float:
        """
        Return the mean number density of a tracer in (Mpc/h)⁻³.

        Computed as total galaxy count divided by the comoving volume of the
        catalog's redshift range, weighted by sky fraction.

        Parameters
        ----------
        *tracers : str
            Tracer names. If no tracers are specified, aggregates across all tracers.

        Returns
        -------
        float
            Mean number density in (Mpc/h)⁻³.
        """
        n_gal = self._ngal(*tracers)
        zrange = self._zrange(*tracers)
        dv = self.fsky * _shell_volume(self.cosmo, np.array(zrange))
        return float(n_gal / dv[0])

    @property
    def nbar(self) -> float:
        """Return the mean number density across all tracers in (Mpc/h)⁻³."""
        return self._nbar()

    def _interpolate_nz(
        self,
        *tracers: str,
        bins: int = 50,
    ) -> Callable[[float], float]:
        # FIXME
        """Interpolate the number density on the full redshift range."""
        # Use cache if available.
        tracer_cache = "_".join(list(tracers)) or "all"
        cache_key = (tracer_cache, bins, self._transform_state)
        if cache_key in self._interpolate_nz_cache:
            return self._interpolate_nz_cache[cache_key]

        tracer_names = tracers or tuple(self.tracers)
        z_values = self.get_tracer_data(*tracer_names)["z"]

        # Compute histogram of redshift distribution
        counts, bin_edges = np.histogram(z_values, bins=bins)

        # Compute shell volumes for each redshift bin
        dv = self.fsky * _shell_volume(self.cosmo, bin_edges)

        nz = counts / dv  # Number density in each redshift bin
        z_centers = 0.5 * (
            bin_edges[:-1] + bin_edges[1:]
        )  # Centers of the redshift bins

        nofz = interp1d(
            z_centers, 
            nz, 
            kind="linear", 
            fill_value=0.0, 
            bounds_error=False,
        )
        self._interpolate_nz_cache[cache_key] = nofz
        return nofz

    def n(self, z: float, *tracers: str, bins: int = 50) -> float | np.ndarray:
        """
        Evaluate the interpolated number density at redshift z.

        Parameters
        ----------
        z : float | np.ndarray
            Redshift value(s) at which to evaluate n(z).
        *tracers : str
            Tracer names. If no tracers are specified, aggregates across all tracers.

        Returns
        -------
        float | np.ndarray
            Number density in (Mpc/h)⁻³. Returns 0 outside the catalog's redshift range.
        """
        return self._interpolate_nz(*tracers, bins=bins)(z)

    # TODO: are extra properties needed to define that catalog at init ? Must repr reflect that ?

    # Transforms:
    # Angular mask / footprint ?
    # downsampling ?
    def add_distance_column(self) -> None:
        """
        Register a transform that appends a comoving distance column to the tracer data.

        The column is computed using the true cosmology at call time of
        ``get_tracer_data``. Has no effect if the transform is already registered.
        """
        self._add_transform(
            Transform(
                name="add_distance",
                func=_add_distance_column,
                kwargs={"cosmo": self.cosmo},
            )
        )

    def downsample(
        self,
        tracer: str,
        n_gal: int | None = None,
        f_gal: float | None = None,
        seed: RandomState | None = None,
    ) -> None:
        """
        Add a downsampling transform for a specific tracer.

        Exactly one of n_gal or f_gal must be provided.

        Parameters
        ----------
        tracer : str
            Tracer to downsample.
        n_gal : int, optional
            Target number of galaxies.
        f_gal : float, optional
            Fraction of galaxies to keep, between 0 and 1.
        seed : RandomState | None
            Random seed for reproducibility. If None, uses the global random state.

        Raises
        ------
        ValueError
            If not exactly one of n_gal or f_gal is provided.
        """
        provided = sum(p is not None for p in (n_gal, f_gal))
        if provided != 1:
            raise ValueError("Exactly one of n_gal or f_gal must be provided.")

        self._add_transform(
            Transform(
                name=f"downsample_{tracer}",
                func=_apply_downsample,
                tracer=tracer,
                kwargs={
                    "tracer": tracer,  # passed for logging purposes
                    "n_gal": n_gal,
                    "f_gal": f_gal,
                    "seed": seed,
                },
            )
        )

    def _save_attrs(self, f: h5py.File) -> None:
        f.attrs["hp_res"] = int(self.hp_res)

    @override
    @classmethod
    def _from_attrs(cls, attrs: dict, cosmo: Cosmology, cosmo_fid: Cosmology) -> Self:
        return cls(
            cosmo=cosmo,
            cosmo_fid=cosmo_fid,
            hp_res=int(attrs.get("hp_res", 256)),
        )


class RandomCutskyCatalog(CutskyCatalog):
    """A random catalog with cutsky geometry and redshift evolution."""

    @classmethod
    def from_snapshot(cls, catalog: CutskyCatalog, seed: int | None = None) -> Self:
        """
        Create a random catalog from an existing CutskyCatalog.

        Inherits cosmology and tracers from the source catalog,
        replacing all position data with uniform random draws, assuming
        fullsky coverage and a uniform redshift distribution within the source catalog's redshift range.

        Parameters
        ----------
        catalog : CutskyCatalog
            Source catalog to copy metadata and tracer counts from.
        seed : int | None
            Random seed for reproducibility.
        """
        # Ensure independent random states for each tracer and between calls of this method (with spawn)
        ntracers = len(catalog.tracers)
        seeds = np.random.SeedSequence(seed).spawn(ntracers)

        random_catalog = cls(
            cosmo=catalog.cosmo,
            cosmo_fid=catalog.cosmo_fid,
        )
        for i, (tracer_name, tracer) in enumerate(catalog.tracers.items()):
            n_gal = len(catalog._data[tracer_name])
            zrange = catalog._zrange(tracer_name)
            random_catalog.set_tracer_data(
                tracer,
                cls._random_positions(
                    n_gal,
                    rarange=(0, 360),
                    decrange=(-90, 90),
                    zrange=zrange,
                    seed=seeds[i],
                ),
            )
        return random_catalog

    @staticmethod
    def _random_positions(
        n_gal: int,
        rarange: tuple[float, float],
        decrange: tuple[float, float],
        zrange: tuple[float, float],
        seed: int | np.random.SeedSequence | None = None,
    ) -> pd.DataFrame:
        """
        Generate a pandas DataFrame of uniform random positions across the specified ranges for right ascension, declination, and redshift.

        Parameters
        ----------
        n_gal : int
            Number of random galaxies to generate.
        zrange : tuple[float, float]
            Redshift range (z_min, z_max) for the random galaxies.
        rarange : tuple[float, float]
            Right ascension range (ra_min, ra_max) in degrees for the random galaxies.
        decrange : tuple[float, float]
            Declination range (dec_min, dec_max) in degrees for the random galaxies.
        seed : int | np.random.SeedSequence | None
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed=seed)

        # Handle cases where the rarange wraps around 360 degrees
        _rarange = list(rarange)  # Mutable copy to handle potential wrapping
        if rarange[0] > rarange[1]:
            _rarange[0] -= 360

        ra = rng.uniform(*_rarange, size=n_gal) % 360  # Wrap RA to [0, 360)
        # Uniform distribution in sin(dec) for proper area weighting
        u = np.sin(np.radians(decrange))
        dec = np.degrees(np.arcsin(rng.uniform(u[0], u[1], size=n_gal)))
        z = rng.uniform(*zrange, size=n_gal)  # TODO: match n(z) ?

        return pd.DataFrame({"ra": ra, "dec": dec, "z": z})


# %% Transforms between box and cutsky geometries


def box_to_cutsky(*args, **kwargs) -> pd.DataFrame:
    """Convert a box geometry to a cutsky geometry."""
    # Input: SnapshotCatalog (positions, cosmology & boxsize), observer position, redshift range
    # Depends on cosmology for distance-redshift conversion.
    # Depends on boxsize & observer position for angle values and eventual periodic wrapping.


def cutsky_to_box(*args, **kwargs) -> pd.DataFrame:
    """Convert a cutsky geometry to a box geometry."""
    # Input: CutskyCatalog (positions, cosmology & redshift range), observer position, boxsize
    # Depends on cosmology for distance-redshift conversion.
    # Depends on redshift range & observer position for angle values and eventual periodic wrapping.
