import fitsio
import numpy as np
import mockfactory
from cosmoprimo.fiducial import DESI


def minmax_xyz_desi(zrange, region='NGC', release='Y1', tracer='LRG'):
    """
    Get the minimum and maximum cartesian coordinates of
    the DESI survey volume for a given region and release.

    Parameters
    ----------
    zrange : tuple
        Tuple containing minimum and maximum redshift (zmin, zmax).
    region : str
        The DESI photometric region, e.g., 'N+SNGC'.
    release : str
        The DESI data release, e.g., 'y1'.

    Returns
    -------
    tuple
        A tuple containing the minimum and maximum coordinates
    """
    data_fn  = f'/global/cfs/cdirs/desi/survey/catalogs/{release}/mocks/SecondGenMocks/AbacusSummit_v4_1/'
    data_fn += f'altmtl0/mock0/LSScats/{tracer}_{region}_clustering.dat.fits'
    data = fitsio.read(data_fn)
    zmin, zmax = zrange
    chosen = np.logical_and(data['Z'] < zmax, data['Z'] > zmin)
    cosmo = DESI()
    dist = cosmo.comoving_radial_distance(data['Z'])
    pos = mockfactory.sky_to_cartesian(dist[chosen], data['RA'][chosen], data['DEC'][chosen])
    pos_min = np.min(pos, axis=0)
    pos_max = np.max(pos, axis=0)
    return pos_min, pos_max

def minmax_skycoord_desi(zrange, region='NGC', release='Y1', tracer='LRG'):
    """
    Get the minimum and maximum sky coordinates of
    the DESI survey volume for a given region and release.

    Parameters
    ----------
    zrange : tuple
        Tuple containing minimum and maximum redshift (zmin, zmax).
    region : str
        The DESI photometric region, e.g., 'N+SNGC'.
    release : str
        The DESI data release, e.g., 'y1'.

    Returns
    -------
    tuple
        Collection of tuples containing the minimum and maximum ra, dec, and dist.
    """
    if release == 'Y5':
        release = 'Y3'  # we don't have Y5 mocks yet, but the minmax should hopefully be the same
    data_fn  = f'/global/cfs/cdirs/desi/survey/catalogs/{release}/mocks/SecondGenMocks/AbacusSummit_v4_1/'
    data_fn += f'altmtl0/mock0/LSScats/{tracer}_{region}_clustering.dat.fits'
    data = fitsio.read(data_fn)
    zmin, zmax = zrange
    chosen = np.logical_and(data['Z'] < zmax, data['Z'] > zmin)
    ra_min = np.min(data['RA'][chosen])
    ra_max = np.max(data['RA'][chosen])
    dec_min = np.min(data['DEC'][chosen])
    dec_max = np.max(data['DEC'][chosen])
    cosmo = DESI()
    dist = cosmo.comoving_radial_distance(data['Z'])
    dist_min = np.min(dist[chosen], axis=0)
    dist_max = np.max(dist[chosen], axis=0)
    return (dist_min, dist_max), (ra_min, ra_max), (dec_min, dec_max)