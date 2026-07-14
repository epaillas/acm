import logging
from collections.abc import Callable

import pandas as pd
from cosmoprimo import Cosmology
from numpy.random import RandomState
import numpy as np
import numpy.typing as npt
from mockfactory.desi import is_in_desi_footprint
from mockfactory.make_survey import DistanceToRedshift

from acm.catalogs.geometry import is_in_photometric_region, cartesian_to_spherical

logger = logging.getLogger(__name__)


def _apply_rsd(
    data: pd.DataFrame,
    los: str,
    hubble: float,
    az: float,
    wrap: float = 0,
    offset: float = 0,
) -> pd.DataFrame:
    """
    Apply RSD shift along the los axis.

    Expects velocity columns named 'vx', 'vy', 'vz'
    corresponding to the position columns 'x', 'y', 'z'.

    Parameters
    ----------
    data : pd.DataFrame
        Galaxy data containing position and velocity columns.
    los : str
        Line-of-sight axis, one of 'x', 'y', 'z'.
    hubble : float
        Hubble parameter H(z) in km/s/(Mpc/h) for the simulation cosmology.
    az : float
        Scale factor a(z) at the snapshot's redshift.
    wrap : float, optional
            Wrap size for periodic boundary conditions, default is 0.
    offset : float, optional
        Offset for periodic wrapping, default is 0.

    Returns
    -------
    pd.DataFrame
        Transformed galaxy data with RSD applied.
    """
    data = data.copy()
    if los == 'los':
        projection = (data["vx"]*data["x"] + data["vy"]*data["y"] + data["vz"]*data["z"])   / (data["x"]**2 + data["y"]**2 + data["z"]**2) 
        projection /= hubble * az
        data["x"] *= (1 + projection)
        data["y"] *= (1 + projection)
        data["z"] *= (1 + projection)
    else:
        v_col = f"v{los}"
        data[los] = data[los] + data[v_col] / (hubble * az)
    if wrap > 0:
        data[los] = (data[los] + offset) % wrap - offset
    return data
    
def _apply_ap(
    data: pd.DataFrame,
    los: str,
    q_par: float,
    q_perp: float,
    pos_columns: tuple[str, ...],
) -> pd.DataFrame:
    """
    Apply AP scaling: q_par along los, q_perp along transverse axes.

    Expects position columns named 'x', 'y', 'z' (or as specified in pos_columns).

    Parameters
    ----------
    data : pd.DataFrame
        Galaxy data containing position columns.
    los : str
        Line-of-sight axis, one of the columns specified in pos_columns.
    q_par : float
        AP scaling factor along the line-of-sight.
    q_perp : float
        AP scaling factor along the transverse directions.
    pos_columns : tuple[str]
        Names of the position columns, e.g. ('x', 'y', 'z').

    Returns
    -------
    pd.DataFrame
        Transformed galaxy data with AP scaling applied.
    """
    data = data.copy()
    for ax in pos_columns:
        data[ax] = data[ax] / (q_par if ax == los else q_perp)
    return data


def _apply_downsample(
    data: pd.DataFrame,
    tracer: str,
    n_gal: int | None = None,
    f_gal: float | None = None,
    nbar: float | None = None,
    volume: Callable[[], float] | None = None,
    seed: RandomState | None = None,
) -> pd.DataFrame:
    """
    Randomly downsample a tracer DataFrame.

    Volume is expected as a callable to allow for transforms that change the effective volume, e.g. AP scaling.

    Parameters
    ----------
    data : pd.DataFrame
        Galaxy data for a specific tracer.
    tracer : str
        Tracer name, used for logging.
    n_gal : int, optional
        Target number of galaxies.
    f_gal : float, optional
        Fraction of galaxies to keep, between 0 and 1.
    nbar : float, optional
        Target number density in (Mpc/h)^-3.
    volume : callable, optional
        Function that returns the current volume, needed to compute target n_gal when downsampling by nbar.
    seed : RandomState or int, optional
        Random seed or RandomState for reproducibility.

    Returns
    -------
    pd.DataFrame
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
        raise ValueError("Exactly one of n_gal, f_gal or nbar must be provided.")

    n_current = len(data)
    if f_gal is not None:
        n_target = round(n_current * f_gal)
    elif n_gal is not None:
        n_target = n_gal
    elif nbar is not None:
        if volume is None:
            raise ValueError(
                "volume function must be provided when downsampling by nbar."
            )
        # Callable to get current volume, which may include transforms, e.g. AP scaling
        n_target = round(nbar * volume())

    if n_target >= n_current:
        logger.warning(
            f"Target n_gal={n_target} >= current n_gal={n_current} for tracer '{tracer}', skipping downsample."
        )
        return data

    return data.sample(n=n_target, random_state=seed).reset_index(drop=True)


def _add_distance_column(df: pd.DataFrame, cosmo: Cosmology) -> pd.DataFrame:
    """Add a comoving distance column to the DataFrame based on the redshift column."""
    if "z" not in df.columns:
        raise ValueError("DataFrame must contain a 'z' column to compute distances.")
    df = df.copy()
    df["distance"] = cosmo.comoving_radial_distance(df["z"])
    return df


def _apply_r_cut(
    data: pd.DataFrame,
    r_min: float,
    r_max: float,
) -> pd.DataFrame:
    """
    
    """
    data = data.copy()

    distance = np.sqrt(data["x"]**2 + data["y"]**2 + data["z"]**2) 
    in_r_lims = (distance > r_min) * (distance < r_max)
    data = data[in_r_lims]

    return data
    

def _apply_angular_mask(
    data: pd.DataFrame,
    tracer: str,
    mask_fractions: dict,
    region: str = 'N+SNGC',
    release: str = 'Y1',
    npasses: int | None = None,
    program: str = 'dark',
    custom_mask_path: str | None = None,
    num_fibonacci_samples: int = 100000
) -> None:
    """
    Applies the angular mask to the cutsky catalog based on the specified region.

    Parameters
    ----------
    region : str
        The region to apply the angular mask. Options include 'N', 'DN', 'DS', 'N+SNGC', 'SNGC', 'SSGC', 'DES', 'NGC', 'SGC'.
    release : str
        The release of the data, e.g., 'Y1'.
    npasses : int, optional
        The number of passes for the mask. If None, defaults to 1.
    program : str
        The program to use for the mask, e.g., 'dark'.
    custom_mask_path : str
        If not set to None, a custom mask file is read for applying the angular mask. The file should be in FITS format
        and should include a column named IN_MASK that corresponds to a boolean healpix mask
    num_fibonacci_samples : int
        The number of points to evenly distribute on teh sky for calculating the survey mask sky fraction

    Returns
    -------
    None
        The cutsky catalog is modified in place.
    """
    data = data.copy()

    _, ra, dec = cartesian_to_spherical(data)
    
    if custom_mask_path is None:
        is_in_desi = is_in_desi_footprint(
            ra,
            dec,
            release=release,
            program=program,
            npasses=npasses
        )
        _, is_in_photo = is_in_photometric_region(
            ra,
            dec,
            region=region
        )
        data = data[is_in_desi & is_in_photo]

        # ==============================================================
        # Calculate survey mask sky fraction using Fibonacci method 
        # for populating RA and dec. points on the sky
        # ==============================================================
        
        # Fibonacci method 
        generate_fibonacci = np.arange(0, num_fibonacci_samples, dtype=float) + 0.5
        mask_dec = np.arccos(1 - 2 * generate_fibonacci / num_fibonacci_samples)
        mask_dec = 180 / np.pi * mask_dec - 90
        mask_ra = (4 * 180 * generate_fibonacci / (1 + np.sqrt(5))) % 360

        # Sky fraction calculation
        mask_in_desi = is_in_desi_footprint(
            mask_ra,
            mask_dec,
            release=release,
            program=program,
            npasses=npasses
        )
        _, mask_in_photo = is_in_photometric_region(
            mask_ra,
            mask_dec,
            region=region
        )

        in_mask = mask_in_desi & mask_in_photo

        mask_fractions[tracer] = np.sum(in_mask) / len(in_mask)

    else:
        mask = fitsio.read(custom_mask_path)
        nside = hp.npix2nside(len(mask['IN_MASK']))
        phi = np.radians(ra)
        theta = np.radians(90 - dec)
        target_pixels = hp.ang2pix(nside, theta, phi)
        is_in_mask = mask['IN_MASK'][target_pixels]
        data = data[is_in_desi & is_in_photo]

        mask_fractions[tracer] = np.sum(mask['IN_MASK']) / len(mask['IN_MASK'])

    return data
    
def _apply_sky_coords(data: pd.DataFrame, redshift_to_distance: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]):
    """
    """
    data = data.copy()
    distance, ra, dec = cartesian_to_spherical(data)
    distance_to_redshift = DistanceToRedshift(redshift_to_distance)
    redshift = distance_to_redshift(distance)
    # return distance, ra, dec # TODO: need to setup pandas dataframe
    