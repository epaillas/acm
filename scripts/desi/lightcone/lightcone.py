from acm.hod import LightconeHOD
from acm import setup_logging
from pyrecon.utils import sky_to_cartesian
import numpy as np
from pathlib import Path
import pandas
from cosmoprimo.fiducial import AbacusSummit

setup_logging()

def get_hod_priors():
    from sunbird.inference.priors import Yuan23
    prior = Yuan23()
    return prior.ranges

def get_hod_params(nrows=None):
    """Some example HOD parameters."""
    hod_dir = Path(f'/pscratch/sd/e/epaillas/emc/hod_params/yuan23/')
    hod_fn = hod_dir / f'hod_params_yuan23_c000.csv'
    df = pandas.read_csv(hod_fn, delimiter=',')
    df.columns = df.columns.str.strip()
    df.columns = list(df.columns.str.strip('# ').values)
    return df.to_dict('list')

def get_clustering_positions(data):
    ra = data['RA']
    dec = data['DEC']
    dist = cosmo.comoving_radial_distance(data['Z'])
    pos = sky_to_cartesian(ra=ra, dec=dec, dist=dist)
    weights = np.ones(len(pos))
    return pos

def plot_footprint():
    import matplotlib.pyplot as plt
    print('Plotting footprint.')
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].scatter(data['RA'], data['DEC'], s=0.1)
    ax[1].scatter(randoms['RA'], randoms['DEC'], s=0.1)
    ax[0].set_title('Data')
    ax[1].set_title('Randoms')
    for aa in ax:
        aa.set_xlabel('right ascension [deg]')
        aa.set_ylabel('declination [deg]')
    plt.tight_layout()
    plt.savefig('footprint_multisnap.png', dpi=300)
    plt.close()

def plot_redshift_distribution():
    import matplotlib.pyplot as plt
    print('Plotting redshift distribution.')
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].hist(data['Z'], density=True, label='data')
    ax[0].hist(randoms['Z'], density=True, label='randoms', ls='--', histtype='step')
    ax[1].hist(data['Z'], density=False, label='data')
    ax[1].hist(randoms['Z'], density=False, label='randoms', ls='--', histtype='step')
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


def get_twopoint_clustering(save_fn=False):
    print('Computing clustering.')
    from pycorr import setup_logging, TwoPointCorrelationFunction
    setup_logging()
    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)
    return TwoPointCorrelationFunction(
        data_positions1=data_positions, randoms_positions1=randoms_positions,
        position_type='pos', edges=edges, mode='smu', gpu=False, nthreads=128, estimator='landyszalay',
    )


# redshift range (in the cutsky) that will be covered by each snapshot
zrange = [0.41, 0.8]

# fiducial cosmology for the redshift-distance relation and RSD
cosmo = AbacusSummit(0)

# read example HOD parameters
hod_params = get_hod_params()

# load abacusHOD class
lightcone = LightconeHOD(varied_params=hod_params.keys(),
                   zrange=zrange, cosmo_idx=0, phase_idx=0,
                   load_existing_hod=False)
# sample HOD parameters and build the lightcone mock
hod = {key: hod_params[key][30] for key in hod_params.keys()}

# Run HOD code
lightcone.sample_hod(hod, nthreads=nthreads,)

# apply angular and radial masks
#lightcone.apply_angular_mask(region='NGC', release='Y1', npasses=None, program='dark')
nz_filename='/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/LRG_NGC_nz.txt'
lightcone.apply_radial_mask(nz_filename=nz_filename)

# generate a random catalog with the same angular and radial masks
lightcone_randoms = LightconeRandoms(
    rarange=(lightcone.catalog['RA'].min(), lightcone.catalog['RA'].max()),
    decrange=(lightcone.catalog['DEC'].min(), lightcone.catalog['DEC'].max()),
    zrange=zrange,
    nbar=2000,  # this is *surface area* density, in (deg^2)^-1
    # csize=10_000_000,  # alternatively, pass the desired number of randoms
)
lightcone_randoms.set_sim_type('base')
#lightcone_randoms.apply_angular_mask(region='NGC', release='Y1', npasses=None, program='dark')
# we use `shape_only=True` to only match the n(z) shape, keeping the randoms amplitude
lightcone_randoms.apply_radial_mask(nz_filename=nz_filename, shape_only=True)

# catalog is a dict that contains sky coordinates, redshifts and number density
data = lightcone.catalog
randoms = lightcone_randoms.catalog

# # get positions for clustering analysis
data_positions = get_clustering_positions(data)
randoms_positions = get_clustering_positions(randoms)

# plot footprint and redshift distribution
plot_footprint()
plot_redshift_distribution()

# # compute the two-point clustering multipoles
tpcf = get_twopoint_clustering()
plot_multipoles()
