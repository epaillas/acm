import fitsio
from pathlib import Path
import argparse
import numpy as np
from acm.estimators.galaxy_clustering.bispectrum import Bispectrum
from acm import setup_logging
from cosmoprimo.fiducial import AbacusSummit
import time


def get_hod_positions(input_fn, los='z'):
    hod = fitsio.read(input_fn)
    pos = np.c_[hod['X'], hod['Y'], hod['Z']]
    hubble = 100 * cosmo.efunc(redshift)
    scale_factor = 1 / (1 + redshift)
    if los == 'x':
        pos[:, 0] += hod['VX'] / (hubble * scale_factor)
    elif los == 'y':
        pos[:, 1] += hod['VY'] / (hubble * scale_factor)
    elif los == 'z':
        pos[:, 2] += hod['VZ'] / (hubble * scale_factor)
    return pos


parser = argparse.ArgumentParser()
parser.add_argument("--start_phase", type=int, default=3000)
parser.add_argument("--n_phase", type=int, default=1900)

args = parser.parse_args()
start_phase = args.start_phase
n_phase = args.n_phase

phases = list(range(start_phase, start_phase + n_phase))
redshift = 0.5
hod = 466
cosmo = AbacusSummit(0)

setup_logging()

bspec = Bispectrum(boxsize=500, boxcenter=0, nmesh=80,
                sightline='global', nthreads=128)
    
bspec.set_binning(k_bins=np.arange(0.013, 0.27, 0.02), lmax=2,
                  k_bins_squeeze=np.arange(0.013, 0.27, 0.02),
                  include_partial_triangles=False)

for phase_idx in phases:
    hod_dir = f'/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior/small/hod{hod:03}/'
    hod_fn = Path(hod_dir) / f'ph{phase_idx:03}_hod{hod:03}.fits'
    if not hod_fn.exists():
        print(f'{hod_fn} not found')
        continue

    hod_positions = get_hod_positions(hod_fn, los='z')

    bspec.assign_data(positions=hod_positions, wrap=True)
    bspec.set_density_contrast()
    bk = bspec.Bk_ideal(discreteness_correction=False)

    k123 = bspec.get_ks()

    save_dir = f'/pscratch/sd/e/epaillas/emc/v1.1/abacus/covariance_sets/small_box/raw/bispectrum/kmax0.25_dk0.02/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / f'bispectrum_ph{phase_idx:03}_hod{hod:03}.npy'
    np.save(save_fn, {'k123': k123, 'bk': bk})
