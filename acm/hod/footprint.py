import numpy as np
import mockfactory
from cosmoprimo.fiducial import DESI


def read_catalog(fn):
    """Wrapper around :meth:`Catalog.read` to read catalog(s)."""
    kw = {}
    if fn.endswith(".h5"):
        kw["group"] = "LSS"
    catalog = mockfactory.Catalog.read(fn, **kw)
    if fn.endswith(".fits"):
        catalog.get(catalog.columns())  # Faster to read all columns at once
    return catalog


def minmax_xyz_desi(
    zrange, region="NGC", release="Y1", tracer="LRG", custom_xyz_file=None
):
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
    custom_xyz_file : str
        If not None, a custom file is read for the positions of the tracers that define
        the survey volume bounds

    Returns
    -------
    tuple
        A tuple containing the minimum and maximum coordinates
    """
    if custom_xyz_file is not None:
        data_fn = custom_xyz_file
    else:
        # data_fn  = f'/global/cfs/cdirs/desi/survey/catalogs/{release}/mocks/SecondGenMocks/AbacusSummit_v4_1/'
        # data_fn += f'altmtl0/mock0/LSScats/{tracer}_{region}_clustering.dat.fits'
        data_fn = "/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/AbacusHighFidelity/altmtl0/loa-v1/mock0/LSScats/"
        data_fn += f"{tracer}_{region}_clustering.dat.h5"
    # data = fitsio.read(data_fn)
    data = read_catalog(data_fn)
    zmin, zmax = zrange
    chosen = np.logical_and(data["Z"] < zmax, data["Z"] > zmin)
    cosmo = DESI()
    dist = cosmo.comoving_radial_distance(data["Z"])
    pos = mockfactory.sky_to_cartesian(
        dist[chosen], data["RA"][chosen], data["DEC"][chosen]
    )
    pos_min = np.min(pos, axis=0)
    pos_max = np.max(pos, axis=0)
    return pos_min, pos_max


def minmax_skycoord_desi(zrange, region="NGC", release="Y1", tracer="LRG"):
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
    if release == "Y5":
        release = "Y3"  # we don't have Y5 mocks yet, but the minmax should hopefully be the same
    data_fn = "/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/AbacusHighFidelity/altmtl0/loa-v1/mock0/LSScats/"
    data_fn += f"{tracer}_{region}_clustering.dat.h5"
    data = read_catalog(data_fn)
    zmin, zmax = zrange
    chosen = np.logical_and(data["Z"] < zmax, data["Z"] > zmin)
    ra_min = np.min(data["RA"][chosen])
    ra_max = np.max(data["RA"][chosen])
    dec_min = np.min(data["DEC"][chosen])
    dec_max = np.max(data["DEC"][chosen])
    cosmo = DESI()
    dist = cosmo.comoving_radial_distance(data["Z"])
    dist_min = np.min(dist[chosen], axis=0)
    dist_max = np.max(dist[chosen], axis=0)
    return (dist_min, dist_max), (ra_min, ra_max), (dec_min, dec_max)
