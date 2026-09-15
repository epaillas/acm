import logging
from collections.abc import Callable

import pandas as pd
from cosmoprimo import Cosmology
from numpy.random import RandomState
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

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
    #mask_fractions: dict,
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

        #mask_fractions[tracer] = np.sum(in_mask) / len(in_mask)

    else:
        mask = fitsio.read(custom_mask_path)
        nside = hp.npix2nside(len(mask['IN_MASK']))
        phi = np.radians(ra)
        theta = np.radians(90 - dec)
        target_pixels = hp.ang2pix(nside, theta, phi)
        is_in_mask = mask['IN_MASK'][target_pixels]
        data = data[is_in_desi & is_in_photo]

        #mask_fractions[tracer] = np.sum(mask['IN_MASK']) / len(mask['IN_MASK'])

    return data
    
def _apply_sky_coords(data: pd.DataFrame, cosmo: Cosmology):
    """
    """
    data = data.copy()
    distance, ra, dec = cartesian_to_spherical(data)
    distance_to_redshift = DistanceToRedshift(cosmo.comoving_radial_distance)
    redshift = distance_to_redshift(distance)
    data = pd.DataFrame({
        'ra': ra,
        'dec': dec,
        'redshift': redshift
    })
    return data


'''
# %% Transforms between box and cutsky geometries

def _apply_box_coords(*args, **kwargs) -> pd.DataFrame:  # ty:ignore[empty-body]
    """Convert a cutsky geometry to a box geometry."""
    # Input: CutskyCatalog (positions, cosmology & redshift range), observer position, boxsize
    # Depends on cosmology for distance-redshift conversion.
    # Depends on redshift range & observer position for angle values and eventual periodic wrapping.

'''

def _apply_radial_mask(data: pd.DataFrame, 
                       sky_fraction: float, 
                       cosmo: Cosmology, 
                       nz_filename: str, 
                       shape_only: bool = False, 
                       dz_new: float = 0.002
                      ) -> None:
    """
    Applies the radial selection function to a cutsky catalog based on 
    an input n(z) file (number desity as a function of redshift).

    Parameters
    ----------
    nz_filename : str
        Path to the n(z) file containing the target number density. Columns
        (1, 2, 3) are zbin_min, zbin_max, and target_nz respectively.
    shape_only : bool, optional
        If True, match only the shape of the n(z), disregarding the amplitude.
    dz_new : float
        redshift interval used for redshift bin edges. Should be small compared to 
        the expected fluctuations in the raw n(z)
        
    Returns
    -------
    None
        The cutsky catalog is modified in place.
    """
    data = data.copy()

    zmin_data = data['redshift'].min()
    zmax_data = data['redshift'].max()

    # read n(z) file
    zbin_min, zbin_max, target_nz = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
    zbin_mid = 0.5 * (zbin_min + zbin_max)
    if zbin_min[0] > zmin_data or zbin_max[-1] < zmax_data:
        raise ValueError('Provided n(z) file does not cover redshift range of data')

    # nz(z) interpolator (piecewise linear)
    nz_spline = InterpolatedUnivariateSpline(zbin_mid, target_nz, k=1, ext=3)

    # --- refine each coarse bin into dz=0.002 sub-bins ---
    nsub = int(round((zbin_max[0] - zbin_min[0]) / dz_new))  # should be 5 for 0.01->0.002
    if not np.allclose(zbin_max - zbin_min, nsub * dz_new, rtol=0, atol=1e-12):
        raise ValueError("Your input bins are not an integer multiple of dz_new.")
    
    # new bin edges per coarse bin: (Nbins, nsub+1)
    edges = zbin_min[:, None] + dz_new * np.arange(nsub + 1)[None, :]
    
    # flatten into sub-bins
    zbin_min = edges[:, :-1].ravel()
    zbin_max = edges[:,  1:].ravel()
    
    #impose lightcone redshift limits on zbins
    select_zbins = (zbin_max > zmin_data) * (zbin_min < zmax_data)
    zbin_min = zbin_min[select_zbins]
    zbin_max = zbin_max[select_zbins]
    zbin_min[0] = zmin_data
    zbin_max[-1] = zmax_data
    zbin_mid = (zbin_min + zbin_max) / 2
    target_nz = nz_spline(zbin_mid)
    
    #calculate volumes of shells
    zedges = np.insert(zbin_max, 0, zbin_min[0])
    dbin_max = cosmo.comoving_radial_distance(zbin_max)
    dedges =  np.insert(dbin_max, 0, cosmo.comoving_radial_distance(zbin_min[0]))
    volume = sky_fraction * 4/3 * np.pi * (dedges[1:]**3 - dedges[:-1]**3) 

    # calculate downsampling ratio
    data_nz = np.histogram(data['redshift'], bins=zedges)[0] / volume
    ratio = target_nz / data_nz
    
    if shape_only:
        max_ratio = np.max(ratio[~np.isinf(ratio)])
        ratio /= max_ratio
    ratio_spline = InterpolatedUnivariateSpline(zbin_mid, ratio, k=1, ext=3)
    # use the spline to get the number density at the redshift of every galaxy
    # then assign a random number to each and compare it to the ratio to determine
    # if the galaxy should be kept or not
    data_nz = nz_spline(data['redshift'])
    select_mask = np.random.uniform(size=len(data['redshift'])) < ratio_spline(data['redshift'])
    data = data[select_mask]
    data['NZ'] = data_nz[select_mask]
    if shape_only:
        data['NZ'] /= max_ratio
    return data