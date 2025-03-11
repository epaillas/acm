# Based on routines implemented in the mockfactory repository
# https://github.com/echaussidon/mockfactory/blob/852ef55e1113986ac2df294a8902c473b5295a83/desi/from_box_to_desi_cutsky.py

from pathlib import Path
import fitsio
import numpy as np
from cosmoprimo.fiducial import DESI
from mockfactory import BoxCatalog, DistanceToRedshift
from mockfactory.desi import get_brick_pixel_quantities


def get_mocks_fn(hod_idx=30, cosmo_idx=0, phase_idx=0, seed_idx=0, redshift=0.5):
    return Path(f'/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z{redshift:.1f}/yuan23_prior', 
                f'c{phase_idx:03}_ph{phase_idx:03}/seed{seed_idx:01}',
                f'hod{hod_idx:03}_raw.fits')

def get_clustering_positions(data_fn):
    data = fitsio.read(data_fn)
    pos = np.c_[data['X'], data['Y'], data['Z']]
    vel = np.c_[data['VX'], data['VY'], data['VZ']]
    print(len(pos) / 2000**3)
    return pos, vel

def get_box_replications(pos, vel, mappings=[-1, 0, 1]):
    rep_pos = []
    rep_vel = []
    for i in mappings:
        for j in mappings:
            for k in mappings:
                rep_pos.append(pos + [boxsize * idx for idx in [i, j, k]])
                rep_vel.append(vel)
                # print(f'{i}{j}{k}')
    rep_pos = np.concatenate(rep_pos)
    rep_vel = np.concatenate(rep_vel)
    return rep_pos, rep_vel

def photometric_region_center(region):
    if region == 'N':
        ra, dec = 192.3, 56.0
    elif region in ['N+DN', 'N+SNGC']:
        ra, dec = 192, 35
    elif region in ['DN', 'SNGC']:
        ra, dec = 192, 13.0
    elif region in ['DS', 'SSGC']:
        ra, dec = 6.4, 5.3
    else:
        ValueError(f'photometric_region_center is not defined for region={region}')
    return ra, dec

def remap_the_box(catalog):
    """
    Since the box is periodic, we can transform the box into a parallelepiped
    following Jordan Carlson and Martin White's algorithm of arXiv:1003.3178.

    It is purely geometric.
    """
    from mockfactory.remap import Cuboid
    # Remap the box, see nb/remap_examples.ipynb to see how we choose the vector
    lattice = Cuboid.generate_lattice_vectors(maxint=1, maxcomb=1, sort=False,
                                              boxsize=[5500, 5500, 5500],
                                              cuboidranges=[[8000, 10000], [4000, 5000], [2000, 4000]])
    # Collect the desired lattice.values:
    u = list(lattice.values())[1][0]
    # Let's remap our catalog!
    catalog = catalog.remap(*u)
    # use z as depth to maximize the sky surface with remapped box: (x, y, z) --> (z, x, y)
    # (x, y, z) --> (z, y,-x)
    catalog.rotate(1, axis='y')  # 1 in units of pi / 2
    # (z, y, -x) --> (z, x, y)
    catalog.rotate(1, axis='x')

    return catalog


def to_cutsky(catalog, dmin, dmax, rsd_factor=None, center_ra=0, center_dec=0, apply_rsd=True):
    """
    Rotate the box to the final position, apply RSD and masks.

    Note
    ----
    RSD needs to be applied before applying the distance cuts.

    Parameters
    ----------
    catalog: BoxCatalog
        Box containing the simulation. Must be large enough for the desired ``dmax`` and ``dmin``.

    dmin : float
        Minimal distance desired for the cutsky. Can be computed with `cosmo.comoving_radial_distance(zmin)`.

    dmax : float
        Maximal distance desired for the cutsky. Can be computed with `cosmo.comoving_radial_distance(zmax)`.

    rsd_factor: float
        Factor to apply to ``catalog.velocity`` to obtain RSD displacement in positions units, to be added to ``catalog.position``.
        It depends on the choice of velocity units in ``catalog``.

    center_ra, center_dec : float, default=0.
        Add angles to rotate the box. The box is centered around (RA, Dec) = (center_ra, center_dec).

    Returns
    -------
    cutsky : CutskyCatalog
        Catalog with desired cutsky and RSD positions.
    """
    from mockfactory import box_to_cutsky, utils

    # Collect limit for the cone
    drange, rarange, decrange = box_to_cutsky(catalog.boxsize, dmax, dmin=dmin)

    # Slice rarange et decrange:
    # rarange = np.array(rarange) + center_ra
    # decrange = np.array(decrange) + center_dec

    isometry, mask_radial, mask_angular = catalog.isometry_for_cutsky(drange, rarange, decrange)
    data_cutsky = catalog.cutsky_from_isometry(isometry, rdd=None)
    if apply_rsd:
        data_cutsky['RSDPosition'] = data_cutsky.rsd_position(f=rsd_factor)
        data_cutsky['DISTANCE'], data_cutsky['RA'], data_cutsky['DEC'] = utils.cartesian_to_sky(data_cutsky['RSDPosition'])
    else:
        data_cutsky['DISTANCE'], data_cutsky['RA'], data_cutsky['DEC'] = utils.cartesian_to_sky(data_cutsky['Position'])
    return data_cutsky

def apply_radial_mask(cutsky, zmin=0., zmax=6., nz_filename='nz_qso_final.dat',
                      apply_redshift_smearing=False, tracer_smearing='QSO',
                      cosmo=None, seed=145, norm=None):
    """
    Match the input n(z) distribution between ``zmin`` and ``zmax``.
    Here, we extract the largest number of galaxy as possible (as default).

    Parameters
    ----------
    cutsky: CutskyCatalog
        Catalog containing at least a column 'Z'.

    zmin: float, default=0.
        Minimal redshift to consider in the n(z).

    zmax: float, default=6.
        Maximum redshift to consider in the n(z).

    nz_filename: string, default='nz_qso_final.dat'
        Where the n(z) is saved, in ``cutsky.position`` units, e.g. (Mpc/h)^(-3). For now, only the final TS format is accepted.

    apply_redshift_smearing: bool, default=False
        If true, apply redshift smearing as in https://github.com/echaussidon/mockfactory/blob/341d915bd37c725e10c0b2f490960efc916a56dd/mockfactory/desi/redshift_smearing.py

    tracer_smearing: str, default='QSO'
        What king of smearing you want to apply. Use the default filename used in mockfactory/desi/redshift_smearing.py

    cosmo : Cosmology
        Cosmology of the input mock, to convert n(z) in ``nz_filename`` to mock units.

    seed : int, default=145
        Random seed, for reproductibility during the masking.

    Returns
    -------
    cutsky : CutskyCatalog
        Catalog with matched n(z) distribution.
    """
    from mockfactory import TabulatedRadialMask

    # Load nz
    zbin_min, zbin_max, n_z = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
    zbin_mid = (zbin_min + zbin_max) / 2
    # Compute comobile volume
    zedges = np.insert(zbin_max, 0, zbin_min[0])
    dedges = cosmo.comoving_radial_distance(zedges)
    volume = dedges[1:]**3 - dedges[:-1]**3
    mask_radial = TabulatedRadialMask(z=zbin_mid, nbar=n_z, interp_order=2, zrange=(zmin, zmax), norm=norm)

    if apply_redshift_smearing:
        from mockfactory.desi import TracerRedshiftSmearing
        # Note: apply redshift smearing before the n(z) match since n(z) is what we observe (ie) containing the smearing
        cutsky['Z'] = cutsky['Z'] + TracerRedshiftSmearing(tracer=tracer_smearing).sample(cutsky['Z'], seed=seed + 13)

    return cutsky[mask_radial(cutsky['Z'], seed=seed)]

def is_in_photometric_region(ra, dec, region, rank=0):
    """DN=NNGC and DS = SNGC"""
    region = region.upper()
    assert region in ['N', 'DN', 'DS', 'N+SNGC', 'SNGC', 'SSGC', 'DES']

    DR9Footprint = None
    try:
        from regressis import DR9Footprint
    except ImportError:
        if rank == 0: logger.info('Regressis not found, falling back to RA/Dec cuts')

    if DR9Footprint is None:
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
        from regressis.utils import build_healpix_map
        # Precompute the healpix number
        nside = 256
        _, pixels = build_healpix_map(nside, ra, dec, return_pix=True)

        # Load DR9 footprint and create corresponding mask
        dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=False, mask_around_des=False, cut_desi=False, verbose=(rank == 0))
        convert_dict = {'N': 'north', 'DN': 'south_mid_ngc', 'N+SNGC': 'ngc', 'SNGC': 'south_mid_ngc', 'DS': 'south_mid_sgc', 'SSGC': 'south_mid_sgc', 'DES': 'des'}
        return pixels, dr9_footprint(convert_dict[region])[pixels]


def apply_photo_desi_footprint(cutsky, region, release, program='dark', npasses=None, rank=0):
    """
    Remove part of the cutsky to match as best as possible (precision is healpix map at nside)
    the DESI release (e.g. y1) footprint and DR9 photometric footprint.
    """
    from mockfactory.desi import is_in_desi_footprint

    # Mask objects outside DESI footprint:
    is_in_desi = is_in_desi_footprint(cutsky['RA'], cutsky['DEC'], release=release, program=program, npasses=npasses)
    cutsky['HPX'], is_in_photo = is_in_photometric_region(cutsky['RA'], cutsky['DEC'], region, rank=rank)
    return cutsky[is_in_desi & is_in_photo]

def generate_redshifts(size, zmin=0., zmax=6., nz_filename='nz_qso_final.dat', cosmo=None, seed=145):
    """
    Generate redshifts following the input n(z) distribution between ``zmin`` and ``zmax``.

    Note
    ----
     * This uses a naive implementation from `RadialMask`, can be improved if it takes too long.
     * Do not need to apply any redshift smearing since the generated redshift will follow the observed n(z) containing the smearing.

    Parameters
    ----------
    size : int
        Number of redshifts to generate.

    zmin : float, default=0.
        Minimal redshift to consider in the n(z).

    zmax : float, default=6.
        Maximum redshift to consider in the n(z).

    nz_filename: string, default='nz_qso_final.dat'
        Where the n(z) is saved, in ``cutsky.position`` units, e.g. (Mpc/h)^(-3). For now, only the final TS format is accepted.

    cosmo : Cosmology
        Cosmology of the input mock, to convert n(z) in ``nz_filename`` to mock units.

    seed : int, default=145
        Random seed, for reproductibility during the masking.

    Returns
    -------
    z : array
        Array of size ``size`` of redshifts following the input tabulated n(z).
    """
    from mockfactory import TabulatedRadialMask

    # Load nz
    zbin_min, zbin_max, n_z = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
    zbin_mid = (zbin_min + zbin_max) / 2
    # Compute comobile volume
    zedges = np.insert(zbin_max, 0, zbin_min[0])
    dedges = cosmo.comoving_radial_distance(zedges)
    volume = dedges[1:]**3 - dedges[:-1]**3
    mask_radial = TabulatedRadialMask(z=zbin_mid, nbar=n_z, interp_order=2, zrange=(zmin, zmax))

    # We generate randomly points in redshift space directly, as this is the unit of n_z file
    return mask_radial.sample(size, cosmo.comoving_radial_distance, seed=seed)

def get_fkp_weights(catalog, nz_filename='nz_qso_final.dat', P0=20000):
    """
    Compute the FKP weights for the input catalog.

    Parameters
    ----------
    catalog : CutskyCatalog
        Catalog containing at least a column 'Z'.

    nz_filename: string, default='nz_qso_final.dat'
        Where the n(z) is saved, in ``cutsky.position`` units, e.g. (Mpc/h)^(-3). For now, only the final TS format is accepted.

    cosmo : Cosmology
        Cosmology of the input mock, to convert n(z) in ``nz_filename`` to mock units.

    Returns
    -------
    w : array
        Array of size ``catalog.size`` of FKP weights.
    """
    from mockfactory import TabulatedRadialMask
    from scipy.interpolate import InterpolatedUnivariateSpline

    zbin_min, zbin_max, n_z = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
    zbin_mid = (zbin_min + zbin_max) / 2
    nz_spline = InterpolatedUnivariateSpline(zbin_mid, n_z, k=1)
    return 1 / (1 + P0 * nz_spline(catalog['Z']))


    return mask_radial.compute_fkp_weight(catalog['Z'], cosmo.comoving_radial_distance)

def plot_footprint(catalog, save_fn=None):
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(catalog['RA'], catalog['DEC'], s=0.1)
    ax.set_xlabel(r'$\textrm{right ascension}$')
    ax.set_ylabel(r'$\textrm{declination}$')
    if save_fn is not None:
        plt.savefig(save_fn, dpi=300, bbox_inches='tight')

def plot_redshift_distribution(catalog, save_fn=None):
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(catalog['Z'], histtype='step', lw=2)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$N(z)$')
    if save_fn is not None:
        plt.savefig(save_fn, dpi=300, bbox_inches='tight')

def concatenate_catalogs(catalogs):
    """
    Concatenate multiple catalogs.
    """
    return {key: np.concatenate([cat[key] for cat in catalogs]) for key in catalogs[0].keys()}

def save_catalog(save_fn, catalog):
    from astropy.table import Table
    from astropy.io import fits
    table = Table(catalog)
    # header = fits.Header({'N_cent': Ncent, 'gal_type': tracer, **self.ball.tracers[tracer]})
    myfits = fits.BinTableHDU(data=table)
    myfits.writeto(save_fn, overwrite=True)
    print(f'Saving {save_fn}.')
    return

if __name__ == '__main__':
    # snapshots = [0.5, 0.8]
    # zranges = [[0.41, 0.6], [0.6, 0.8]]
    snapshots = [0.8]
    zranges = [[0.6, 0.8]]
    hods = list(range(30, 31))
    cosmo_idx, phase_idx, seed_idx = 0, 0, 0
    generate_randoms = True

    for hod_idx in hods:
        data_catalogs = []
        randoms_catalogs = []
        for zsnap, zrange in zip(snapshots, zranges):
            boxsize = 2000  # size of the subbox

            data_fn = get_mocks_fn(phase_idx=phase_idx, seed_idx=seed_idx,
                                hod_idx=hod_idx, cosmo_idx=cosmo_idx,
                                redshift=zsnap)
            data_positions, data_velocities = get_clustering_positions(data_fn)
            data_nbar = len(data_positions) / boxsize ** 3
            print(data_positions[:,:].min(), data_positions[:,:].max())

            # Take an HOD catalog measured from a 2 Gpc/h box and replicate it in a 6 Gpc/h box
            # data_positions, data_velocities = get_box_replications(data_positions, data_velocities, mappings=[-1, 0, 1])

            box = BoxCatalog(
                data={'Position': data_positions, 'Velocity': data_velocities},
                position='Position',
                velocity='Velocity',
                boxsize=[2000, 2000, 2000],  # size of the box with replicas
                boxcenter=[0, 0, 0],
            )
            box.recenter()

            # Apply the DESI Y1 LRG cuts

            release, program, npasses = 'y1', 'dark', None
            tracer = 'LRG'
            region = 'N+SNGC'  # NGC is composed of two photometric regions

            # Add maskbits?
            # This step can be long. Large sky coverage need several nodes to be executed in small amounts of time ( ~50 bricks per process)
            # collect only maskbits, see mockfactory/desi/brick_pixel_quantities for other quantities as PSFSIZE_R or LRG_mask
            add_brick_quantities = {'maskbits': {'fn': '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/{region}/coadd/{brickname:.3s}/{brickname}/legacysurvey-{brickname}-maskbits.fits.fz', 'dtype': 'i2', 'default': 1}}
            # Set as False to save time.
            add_brick_quantities = False

            cosmo = DESI()
            d2z = DistanceToRedshift(cosmo.comoving_radial_distance)
            zmin, zmax = zrange
            z = (zmin + zmax) / 2
            rsd_factor = cosmo.sigma8_z(z=z, of='theta_cb') / cosmo.sigma8_z(z=z, of='delta_cb')
                    
            center_ra, center_dec = photometric_region_center(region=region)

            cutsky = to_cutsky(box, cosmo.comoving_radial_distance(zmin),
                               cosmo.comoving_radial_distance(zmax), rsd_factor,
                               center_ra=center_ra, center_dec=center_dec)
            cutsky['Z'] = d2z(cutsky['DISTANCE'])

            # Match the nz distribution
            nz_filename = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/{tracer}_NGC_nz.txt'
            cutsky = apply_radial_mask(cutsky, zmin, zmax, nz_filename=nz_filename,
                                    apply_redshift_smearing=False, cosmo=cosmo, norm=1/data_nbar)

            desi_cutsky = apply_photo_desi_footprint(cutsky, region, release, program, npasses=npasses)
            print(desi_cutsky)

            if add_brick_quantities:
                tmp = get_brick_pixel_quantities(desi_cutsky['RA'], desi_cutsky['DEC'], add_brick_quantities)
                for key, value in tmp.items(): desi_cutsky[key.upper()] = value

            # plot_footprint(desi_cutsky, save_fn=f'footprint_{zsnap:.1f}.png')
            # plot_redshift_distribution(desi_cutsky, save_fn=f'zdist_{zsnap:.1f}.png')

            data_catalogs.append(
                {'RA': desi_cutsky['RA'], 'DEC': desi_cutsky['DEC'], 'Z': desi_cutsky['Z']}
            )


        # Concatenate catalogs
        data_catalog = concatenate_catalogs(data_catalogs)

        save_dir = Path(f'/pscratch/sd/e/epaillas/desi-y1-acm/hods/cutsky/yuan23/',
                        f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx:01}/')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fn = Path(save_dir) / f'LRG_NGC_hod{hod_idx:03}.fits'
        save_catalog(save_fn, data_catalog)
        
        plot_footprint(data_catalog, save_fn=f'footprint_hod{hod_idx:03}.png')
        plot_redshift_distribution(data_catalog, save_fn=f'zdist_hod{hod_idx:03}.png')

        if generate_randoms:
            # Generate associated randoms
            from mockfactory import RandomCutskyCatalog, RandomBoxCatalog, box_to_cutsky

            # We want 10 times more than the cutsky mock
            nrand_over_data = 100

            # randoms_nbar = 10 * 6e-4
            # randoms = RandomBoxCatalog(boxsize=[6000, 6000, 6000], nbar=randoms_nbar)
            # randoms.recenter()

            # randoms = to_cutsky(randoms, cosmo.comoving_radial_distance(zmin),
            #                     cosmo.comoving_radial_distance(zmax), apply_rsd=False,
            #                     center_ra=center_ra, center_dec=center_dec)

            # Since random are generated not directly on DESI footprint, we take the size of cutsky and not desi_cutsky
            nrandoms = int(len(data_catalog['RA']) * nrand_over_data + 0.5)
            # Collect limit for the cone
            zmin, zmax = np.min(zranges), np.max(zranges)
            _, rarange, decrange = box_to_cutsky(box.boxsize, cosmo.comoving_radial_distance(zmax), dmin=cosmo.comoving_radial_distance(zmin))

            print(np.array(rarange) + 180, decrange)

            randoms = RandomCutskyCatalog(rarange=np.array(rarange) + 180, decrange=np.array(decrange), csize=nrandoms)
            # # randoms = RandomCutskyCatalog(rarange=np.array(rarange), decrange=np.array(decrange), csize=nbr_randoms)

            # Match the desi footprint and apply the DR9 mask
            randoms = apply_photo_desi_footprint(randoms, region, release, program, npasses=npasses)
            # if add_brick_quantities:
            #     tmp = get_brick_pixel_quantities(randoms['RA'], randoms['DEC'], add_brick_quantities, mpicomm=mpicomm)
            #     for key, value in tmp.items(): randoms[key.upper()] = value
            # if rank == 0: logger.info(f'Match region: {region} and release footprint: {release} + bricks done in {MPI.Wtime() - start:2.2f} s.')

            # Use the naive implementation of mockfactory/make_survey/BaseRadialMask
            # draw numbers according to a uniform law until to find enough correct numbers
            # basically, this is the so-called 'methode du rejet'
            randoms['Z'] = generate_redshifts(randoms.size, zmin, zmax, nz_filename=nz_filename, cosmo=cosmo)

            randoms = {'RA': randoms['RA'], 'DEC': randoms['DEC'], 'Z': randoms['Z']}
            save_fn = Path(save_dir) / f'LRG_NGC_hod{hod_idx:03}_randoms.fits'
            save_catalog(save_fn, randoms)

            plot_footprint(randoms, save_fn=f'footprint_hod{hod_idx:03}_randoms.png')
            plot_redshift_distribution(randoms, save_fn=f'zdist_hod{hod_idx:03}_randoms.png')