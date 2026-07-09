import numpy as np
import os
import logging
import healpy as hp

from acm.catalogs.factories import BaseCatalogFactory, SnapshotCatalogFactory
import desimodel.footprint
import desimodel.io

from mockfactory.make_survey import DistanceToRedshift
from mockfactory.desi import is_in_desi_footprint
from mockfactory.utils import cartesian_to_sky, sky_to_cartesian
from cosmoprimo import Cosmology

# Optional imports with better error handling
try:
    from regressis import DR9Footprint
    from regressis.utils import build_healpix_map
    HAS_REGRESSIS = True
except ImportError:
    DR9Footprint = None
    build_healpix_map = None
    HAS_REGRESSIS = False


logger = logging.getLogger('DESI footprint')


# check if the env variable DESI_SPECTRO_REDUX is defined, otherwise load default path:
try:
    redux_path = os.environ['DESI_SPECTRO_REDUX']
except KeyError:
    logger.debug("$DESI_SPECTRO_REDUX is not set in the current environment. No assurance for the existence of files. Default path will be used: /global/cfs/cdirs/desi/spectro/redux")
    redux_path = '/global/cfs/cdirs/desi/spectro/redux'

# check if the env variable DESI_SURVEYOPS is defined (needed for desimodel.io.load_tiles()):
    try:
        redux_path = os.environ['DESI_SURVEYOPS']
    except KeyError:
        # see: https://desisurvey.slack.com/archives/C025RHKPV8R/p1729735768040629?thread_ts=1729733422.550579&cid=C025RHKPV8R
        logger.debug("$DESI_SURVEYOPS is not set in the current environment. No assurance for the existence of files. Default path will be used: /global/cfs/cdirs/desi/survey/ops/surveyops/trunk")
        os.environ['DESI_SURVEYOPS'] = '/global/cfs/cdirs/desi/survey/ops/surveyops/trunk'

# Valid DESI photometric regions
# N = North, DN = Dark North, DS = Dark South, SNGC = South NGC, SSGC = South SGC
# DES = Dark Energy Survey, NGC = North Galactic Cap, SGC = South Galactic Cap
VALID_REGIONS = ['N', 'DN', 'DS', 'N+SNGC', 'SNGC', 'SSGC', 'DES', 'NGC', 'SGC']

def is_in_photometric_region(
        ra: np.ndarray,
        dec: np.ndarray,
        region: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine if the given RA/Dec coordinates are within the specified photometric region.

        Parameters
        ----------
        ra : np.ndarray
            Array of right ascension values in degrees.
        dec : np.ndarray
            Array of declination values in degrees.
        region : str
            The photometric region to check. Options include 'N', 'DN', 'DS', 'N+SNGC',
            'SNGC', 'SSGC', 'DES', 'NGC', 'SGC'.

        Returns
        -------
        pixels : np.ndarray
            Healpix pixel numbers corresponding to the RA/Dec coordinates.
        mask : np.ndarray
            Boolean mask indicating whether each RA/Dec coordinate is within the specified region.
        """
        region = region.upper()
        if region not in VALID_REGIONS:
            raise ValueError(f"Invalid region '{region}'. Must be one of: {', '.join(VALID_REGIONS)}")

        if not HAS_REGRESSIS:
            mask = np.ones_like(ra, dtype='?')
            if region == 'DES':
                raise ValueError('Do not know DES cuts, install regressis')
            dec_cut = 32.375
            if region == 'N':
                mask &= dec > dec_cut
            else:  # S
                mask &= dec < dec_cut
            if region in ['DN', 'DS', 'SNGC', 'SSGC']:
                mask_ra = (ra > 100 - dec)
                mask_ra &= (ra < 280 + dec)
                if region in ['DN', 'SNGC']:
                    mask &= mask_ra
                else:  # DS
                    mask &= dec > -25
                    mask &= ~mask_ra
            return np.nan * np.ones(ra.size), mask
        else:
            # Precompute the healpix number
            nside = 256
            _, pixels = build_healpix_map(nside, ra, dec, return_pix=True)

            # Load DR9 footprint and create corresponding mask
            dr9_footprint = DR9Footprint(
                nside,
                mask_lmc=False,
                clear_south=False,
                mask_around_des=False,
                cut_desi=False
            )
            convert_dict = {
                'N': 'north',
                'DN': 'south_mid_ngc',
                'N+SNGC': 'ngc', 'SNGC': 'south_mid_ngc',
                'DS': 'south_mid_sgc',
                'SSGC': 'south_mid_sgc',
                'DES': 'des',
                'NGC': 'ngc',
                'SGC': 'south_mid_sgc',
            }
            return pixels, dr9_footprint(convert_dict[region])[pixels]


def minmax_xyz_desi(zrange: tuple, 
                    cosmo : Cosmology,
                    region : str ='NGC', 
                    release : str ='Y1', 
                    program : str = 'dark', 
                    tracer : str ='LRG', 
                    num_fibonacci_samples : int =786432, 
                    custom_healpix_mask : np.typing.NDArray | None = None,
                    npasses: int | None = None,
                   ):
    """
    Get the minimum and maximum cartesian coordinates of
    the DESI survey volume for a given region and release.

    Parameters
    ----------
    zrange : tuple
        Tuple containing minimum and maximum redshift (zmin, zmax).
    cosmo : Cosmology
        The assumed cosmology
    region : str
        The DESI photometric region, e.g., 'N+SNGC'.
    release : str
        The DESI data release, e.g., 'y1'.
    program : str
        The DESI survey program, e.g. 'dark'
    tracer : str
        The DESI survey tracer, e.g. 'LRG'
    num_fibonacci_samples : int
        The number of points used for Fibonacci sampling the mask region. Defaults
        to 786432, the number of pixels in a HEALpix mask with NSIDE=256
    custom_healpix_mask : numpy array
        If not None, a custom mask is used to define
        the survey volume bounds

    Returns
    -------
    tuple
        A tuple containing the minimum and maximum coordinates
    """
    # Fibonacci method for populating RA DEC points on the sky
    generate_fibonacci = np.arange(0, num_fibonacci_samples, dtype=float) + 0.5
    mask_dec = np.arccos(1 - 2 * generate_fibonacci / num_fibonacci_samples)
    mask_dec = 180 / np.pi * mask_dec - 90
    mask_ra = (4 * 180 * generate_fibonacci / (1 + np.sqrt(5))) % 360

    
    if custom_healpix_mask is not None:
        nside = hp.npix2nside(len(custom_healpix_mask))
        galaxy_pixels = hp.ang2pix(nside, mask_ra, mask_dec, lonlat = True)
        in_mask = custom_healpix_mask[galaxy_pixels]
    else:
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

    # sample points in mask
    mask_ra = mask_ra[in_mask]
    mask_dec = mask_dec[in_mask]

    # place sampled points at inner and outer redshift limits
    zmin, zmax = zrange
    dist_lims = cosmo.comoving_radial_distance([zmin, zmax])
    dist = np.ones_like(mask_ra)
    dist[::2] = dist_lims[0]
    dist[1::2] = dist_lims[1]

    #get minimum and maximum points
    pos = sky_to_cartesian(dist, mask_ra, mask_dec)
    pos_min = np.min(pos, axis=0)
    pos_max = np.max(pos, axis=0)
    return pos_min, pos_max

def box_to_cutsky(snapshot_catalog_factory: SnapshotCatalogFactory,
                 cosmo: Cosmology,):
    """
    Convert a box catalog with cartesian positions and velocities to a cutsky catalog
    with sky coordinates and redshifts.

    Parameters
    ----------
    snapshot_catalog_factory : SnapshotCatalog
        The box catalog containing positions and velocities.
    zrsd : float, optional
        Redshift at which to evaluate the cosmology to apply the RSD, by default None.

    Returns
    -------
    cutsky : dict
        Dictionary containing the cutsky catalog with keys 'Distance', 'RA', 'DEC', and 'Z'.
    """

    cutsky = {}
    d2r = mockfactory.DistanceToRedshift(distance=cosmo.comoving_radial_distance)
    cutsky['Distance'], cutsky['RA'], cutsky['DEC'] = cartesian_to_sky(snapshot_catalog_factory.position)
    cutsky['Z'] = d2r(cutsky['Distance'])
    return cutsky