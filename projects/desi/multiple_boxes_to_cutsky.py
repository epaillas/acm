from pathlib import Path
import fitsio
import pandas
import numpy as np
from cosmoprimo.fiducial import AbacusSummit
from acm.hod import BoxHOD
from mockfactory import BoxCatalog, RandomBoxCatalog, DistanceToRedshift, box_to_cutsky, cartesian_to_sky
import matplotlib.pyplot as plt


def get_data_fn(hod_idx=30, cosmo_idx=0, phase_idx=0, seed_idx=0):
    return Path(f'/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z{zsnap:.1f}/yuan23_prior', 
                f'c{phase_idx:03}_ph{phase_idx:03}/seed{seed_idx:01}',
                f'hod{hod_idx:03}_raw.fits')

def get_hod_params(cosmo_idx=0):
    hod_dir = Path(f'/pscratch/sd/e/epaillas/emc/hod_params/yuan23/cosmo_split')
    hod_fn = hod_dir / f'hod_params_yuan23_c{cosmo_idx:03}.csv'
    df = pandas.read_csv(hod_fn, delimiter=',')
    df.columns = df.columns.str.strip()
    df.columns = list(df.columns.str.strip('# ').values)
    return df.to_dict('list')

def setup_hod(hod_params, cosmo_idx=0, phase_idx=0):
    hod_params = get_hod_params(cosmo_idx)
    return BoxHOD(varied_params=hod_params.keys(),
                  sim_type='base', redshift=zsnap,
                  cosmo_idx=cosmo_idx, phase_idx=phase_idx)

def run_hod(abacus, hod_params, hod_idx=30, seed=0):
    hod = {key: hod_params[key][hod_idx] for key in hod_params.keys()}
    return abacus.run(hod, nthreads=64, seed=seed)['LRG']

def get_data_box(cosmo_idx=0, phase_idx=0, hod_idx=30):
    print('Reading data in a box')
    hod_lhc = get_hod_params(cosmo_idx)
    abacus = setup_hod(hod_lhc, cosmo_idx=cosmo_idx, phase_idx=phase_idx)
    data = run_hod(abacus, hod_lhc, hod_idx)
    pos = np.c_[data['X'], data['Y'], data['Z']]
    vel = np.c_[data['VX'], data['VY'], data['VZ']]
    data = BoxCatalog(
        data={'Position': pos, 'Velocity': vel},
        position='Position',
        velocity='Velocity',
        boxsize=boxsize,
        boxcenter=boxcenter,
    )
    data.recenter()
    return data

def get_randoms_box(nbar, boxsize, boxcenter, seed=42):
    print('Generating randoms with nbar =', nbar)
    return RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)

def apply_geometric_cuts(catalog, boxsize, dmax):
    print('Applying geometric cuts.')
    # largest (RA, Dec) range we can achieve for a maximum distance of dist + boxsize / 2.
    drange, rarange, decrange = box_to_cutsky(boxsize=boxsize, dmax=dist + boxsize / 2.)
    rarange = np.array(rarange) + 192
    decrange = np.array(decrange) + 35
    # returned isometry corresponds to a displacement of the box along the x-axis to match drange, then a rotation to match rarange and decrange
    isometry, mask_radial, mask_angular = catalog.isometry_for_cutsky(drange=drange, rarange=rarange, decrange=decrange)
    return catalog.cutsky_from_isometry(isometry, rdd=None)

def apply_rsd(catalog):
    print('Applying RSD.')
    a = 1 / (1 + zsnap) # scale factor
    H = 100.0 * cosmo.efunc(zsnap)  # Hubble parameter in km/s/Mpc
    rsd_factor = 1 / (a * H)  # multiply velocities by this factor to convert to Mpc/h
    catalog['RSDPosition'] = catalog.rsd_position(f=rsd_factor)
    return catalog

def get_sky_positions(catalog, use_rsd=False):
    print('Converting to sky positions.')
    pos = 'RSDPosition' if use_rsd else 'Position'
    catalog['Distance'], catalog['RA'], catalog['DEC'] = cartesian_to_sky(catalog[pos])
    catalog['Z'] = distance_to_redshift(catalog['Distance'])
    return catalog

def apply_radial_mask(catalog, zmin=0., zmax=6., seed=42, norm=None):
    print('Applying radial mask.')
    from mockfactory import TabulatedRadialMask
    nz_filename = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/LRG_NGC_nz.txt'
    zbin_min, zbin_max, n_z = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
    zbin_mid = (zbin_min + zbin_max) / 2
    zedges = np.insert(zbin_max, 0, zbin_min[0])
    dedges = cosmo.comoving_radial_distance(zedges)
    volume = dedges[1:]**3 - dedges[:-1]**3
    mask_radial = TabulatedRadialMask(z=zbin_mid, nbar=n_z, interp_order=2, zrange=(zmin, zmax), norm=norm)
    return catalog[mask_radial(catalog['Z'], seed=seed)]

def apply_footprint_mask(catalog):
    print('Applying footprint mask.')
    from mockfactory.desi import is_in_desi_footprint
    is_in_desi = is_in_desi_footprint(catalog['RA'], catalog['DEC'], release='y1', program='dark', npasses=None)
    return catalog[is_in_desi]

def get_data_cutsky():
    print('Getting data cutsky.')
    data_cutsky = apply_geometric_cuts(data, boxsize, dist)
    data_cutsky = apply_rsd(data_cutsky)
    data_cutsky = get_sky_positions(data_cutsky, use_rsd=True)
    data_cutsky = apply_radial_mask(data_cutsky, zmin=zmin, zmax=zmax, norm=1/data_nbar)
    data_cutsky = apply_footprint_mask(data_cutsky)
    return data_cutsky

def get_randoms_cutsky():
    print('Getting randoms cutsky.')
    randoms_cutsky = apply_geometric_cuts(randoms, boxsize, dist)
    randoms_cutsky = get_sky_positions(randoms_cutsky, use_rsd=False)
    randoms_cutsky = apply_radial_mask(randoms_cutsky, zmin=zmin, zmax=zmax, norm=1/data_nbar)
    randoms_cutsky = apply_footprint_mask(randoms_cutsky)
    return randoms_cutsky

def get_twopoint_clustering(save_fn=False):
    print('Computing clustering.')
    from pycorr import setup_logging, TwoPointCorrelationFunction
    setup_logging()
    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)
    return TwoPointCorrelationFunction(
        data_positions1=data_cutsky['RSDPosition'], randoms_positions1=randoms_cutsky['Position'],
        position_type='pos', edges=edges, mode='smu', gpu=False, nthreads=256, estimator='landyszalay',
    )

def plot_footprint():
    print('Plotting footprint.')
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].scatter(data_cutsky['RA'], data_cutsky['DEC'], s=0.1)
    ax[1].scatter(randoms_cutsky['RA'], randoms_cutsky['DEC'], s=0.1)
    ax[0].set_title('Data')
    ax[1].set_title('Randoms')
    for aa in ax:
        aa.set_xlabel('right ascension [deg]')
        aa.set_ylabel('declination [deg]')
    plt.tight_layout()
    plt.savefig('footprint_multisnap.png', dpi=300)
    plt.close()

def plot_redshift_distribution():
    print('Plotting redshift distribution.')
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].hist(data_cutsky['Z'], density=True, label='data')
    ax[0].hist(randoms_cutsky['Z'], density=True, label='randoms', ls='--', histtype='step')
    ax[1].hist(data_cutsky['Z'], density=False, label='data')
    ax[1].hist(randoms_cutsky['Z'], density=False, label='randoms', ls='--', histtype='step')
    ax[0].set_title('Normalized')
    ax[1].set_title('Counts')
    for aa in ax:
        aa.set_xlabel('redshift')
        aa.legend()
    plt.tight_layout()
    plt.savefig('redshift_distribution_multisnap.png', dpi=300)
    plt.close()

def plot_multipoles():
    print('Plotting multipoles.')
    import matplotlib.pyplot as plt
    s, multipoles = tpcf(ells=(0, 2), return_sep=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(s, s**2 * multipoles[0], ls='--')
    ax.plot(s, s**2 * multipoles[1], ls='--')
    ax.grid()
    ax.set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$')
    ax.set_ylabel(r'$s^2\xi_{\ell}(s)\,[h^{-2}{\rm Mpc}^2]$')
    plt.tight_layout()
    plt.savefig('multipoles_multisnap.png', dpi=300)
    plt.close()

def save_catalog(catalog, filename):
    # save dictionary as fits file
    from astropy.io import fits
    from astropy.table import Table
    myfits = fits.BinTableHDU(Table(catalog))
    myfits.writeto(filename, overwrite=True)
    print(f'Saving {filename}.')


generate_randoms = False

# define cosmology and redshifts
# phase_idx = 0
for phase_idx in range(1, 8):
    cosmo_idx = 0
    hod_idx = 2
    cosmo = AbacusSummit(cosmo_idx)
    snapshots = [0.5, 0.8]
    zranges = [(0.41, 0.6), (0.6, 1.09)]

    # loop over different box snapshots and convert to cutsky shells
    data_cutsky = {}
    randoms_cutsky = {}
    for zsnap, zranges in zip(snapshots, zranges):
        print(f'Processing snapshot z={zsnap:.1f}, zranges={zranges}.')
        zmin, zmax = zranges
        z = (zmin + zmax) / 2
        dist = cosmo.comoving_radial_distance(z)
        distance_to_redshift = DistanceToRedshift(distance=cosmo.comoving_radial_distance)

        # define data and randoms catalogs in a box
        boxsize = 2000
        boxcenter = 0.0 
        data = get_data_box(cosmo_idx=cosmo_idx, phase_idx=phase_idx, hod_idx=hod_idx)
        data_nbar = len(data) / boxsize ** 3
        randoms_nbar = data_nbar * 5
        randoms = get_randoms_box(randoms_nbar, boxsize, boxcenter)

        # convert to cutsky shell
        tmp_data_cutsky = get_data_cutsky()
        tmp_randoms_cutsky = get_randoms_cutsky()

        # concatenate to previous shell, if any
        data_keys = ['RA', 'DEC', 'Z', 'RSDPosition', 'Distance', 'Position']
        randoms_keys = ['RA', 'DEC', 'Z', 'Position', 'Distance']
        if data_cutsky:
            for key in data_keys:
                data_cutsky[key] = np.concatenate([data_cutsky[key], tmp_data_cutsky[key]])
            for key in randoms_keys:
                randoms_cutsky[key] = np.concatenate([randoms_cutsky[key], tmp_randoms_cutsky[key]])
        else:
            for key in data_keys:
                data_cutsky[key] = tmp_data_cutsky[key]
            for key in randoms_keys:
                randoms_cutsky[key] = tmp_randoms_cutsky[key]

    # save catalogs
    save_dir = Path('/pscratch/sd/e/epaillas/acm/desi/abacus/cutsky/')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_catalog(data_cutsky, save_dir / f'LRG_NGC_c{cosmo_idx:03}_ph{phase_idx:03}_hod{hod_idx:03}.dat.fits')
    save_catalog(randoms_cutsky, save_dir / f'LRG_NGC_c{cosmo_idx:03}_ph{phase_idx:03}_hod{hod_idx:03}.ran.fits')

    # # get clustering
    tpcf = get_twopoint_clustering()
    save_fn = save_dir / f'tpcf_LRG_NGC_c{cosmo_idx:03}_ph{phase_idx:03}_hod{hod_idx:03}.fits'
    tpcf.save(save_fn)

    # plot results
    plot_footprint()
    plot_redshift_distribution()
    plot_multipoles()
