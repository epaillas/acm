import numpy as np
from pathlib import Path
from acm.estimators.galaxy_clustering.bispectrum import Bispectrum
from acm import setup_logging
from astropy.table import Table
from cosmoprimo.cosmology import Cosmology


def read_lrg(filename, apply_rsd=True, los='z',):
    data = Table.read(filename)
    pos = data['pos']
    if apply_rsd:
        vel = data['vel']
        pos_rsd = pos + vel / (hubble * scale_factor)
        los_dict = {'x': 0, 'y': 1, 'z': 2}
        pos[:, los_dict[los]] = pos_rsd[:, los_dict[los]]
    is_lrg = data["diffsky_isLRG"].astype(bool)
    return pos[is_lrg]

def compute_tpcf(data_positions, edges, boxsize, nthreads=4, gpu=True, los='z'):
    return TwoPointCorrelationFunction(
        'smu', edges=edges, data_positions1=data_positions,
        engine='corrfunc', boxsize=boxsize, nthreads=nthreads, gpu=gpu,
        compute_sepsavg=False, position_type='pos', los=los,
    )

setup_logging()

# define cosmology
boxsize = 1000
redshift = 0.5
phases = [1, 2]
cosmo = Cosmology(Omega_m=0.3089, h=0.6774, n_s=0.9667,
                  sigma8=0.8147, engine='class')  # UNIT cosmology
hubble = 100 * cosmo.efunc(redshift)
adict = {0.5: '67120', 0.8: '54980'}  # redshift to scale factor string for UNIT
scale_factor = 1 / (1 + redshift)
los = 'z'
galsample = 'mass_conc'
version = 0.3

bspec = Bispectrum(boxsize=boxsize, boxcenter=0, nmesh=160,
                sightline='global', nthreads=128)
    
bspec.set_binning(k_bins=np.arange(0.013, 0.27, 0.02), lmax=2,
                  k_bins_squeeze=np.arange(0.013, 0.27, 0.02),
                  include_partial_triangles=False)

# loop over the different lines of sight
for phase_idx in phases:
    # read simulation
    data_dir = '/global/cfs/cdirs/desicollab/users/gbeltzmo/C3EMC/UNIT'
    data_fn = Path(data_dir) / f'galsampled_diffsky_mock_{adict[redshift]}_fixedAmp_{phase_idx:03}_{galsample}_v{version}.hdf5'
    data_positions = read_lrg(data_fn, apply_rsd=True, los='z')

    bspec.assign_data(positions=data_positions, wrap=True)
    bspec.set_density_contrast()
    bk = bspec.Bk_ideal(discreteness_correction=False)

    k123 = bspec.get_ks()

    weight = k123.prod(axis=0) / 1e5
    multipoles = np.concatenate([weight * bk[f'b{i}'] for i in [0, 2]])


    # save the results
    output_dir = f'/pscratch/sd/e/epaillas/emc/v1.1/diffsky/data_vectors/galsampled_{adict[redshift]}_fixedAmp_{phase_idx:03}_{galsample}_v{version}'
    output_fn = Path(output_dir) / f'bk.npy'
    cout = {'k123': k123, 'diffsky_y': multipoles}
    np.save(output_fn, cout)