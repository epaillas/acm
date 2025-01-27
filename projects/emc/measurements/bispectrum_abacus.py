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
parser.add_argument("--start_hod", type=int, default=0)
parser.add_argument("--n_hod", type=int, default=1)
parser.add_argument("--start_cosmo", type=int, default=0)
parser.add_argument("--n_cosmo", type=int, default=1)
parser.add_argument("--start_phase", type=int, default=0)
parser.add_argument("--n_phase", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()
start_hod = args.start_hod
n_hod = args.n_hod
start_cosmo = args.start_cosmo
n_cosmo = args.n_cosmo
start_phase = args.start_phase
n_phase = args.n_phase

hods = list(range(start_hod, start_hod + n_hod))
phases = list(range(start_phase, start_phase + n_phase))
redshift = 0.5

setup_logging()

bspec = Bispectrum(boxsize=2000, boxcenter=0, nmesh=320,
                sightline='global', nthreads=128)
    
bspec.set_binning(k_bins=np.arange(0.013, 0.27, 0.02), lmax=2,
                  k_bins_squeeze=np.arange(0.013, 0.27, 0.02),
                  include_partial_triangles=False)

for cosmo_idx in range(start_cosmo, start_cosmo + n_cosmo):
    for phase_idx in phases:
        for hod in hods:
            t0 = time.time()
            print(f'Processing c{cosmo_idx:03} hod{hod:03}')
            hod_dir = f'/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z0.5/yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed0/'
            hod_fn = Path(hod_dir) / f'hod{hod:03}.fits'
            if not hod_fn.exists():
                # print(f'{hod_fn} does not exist')
                continue

            cosmo = AbacusSummit(cosmo_idx)

            hod_positions = get_hod_positions(hod_fn, los='z')

            bspec.assign_data(positions=hod_positions, wrap=True)
            bspec.set_density_contrast()
            bk = bspec.Bk_ideal(discreteness_correction=False)

            print(f'Elapsed time: {time.time() - t0:.2f} s')

            k123 = bspec.get_ks()

            save_dir = f'/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/raw/bispectrum/kmin0.013_kmax0.253_dk0.020/c{cosmo_idx:03}_ph{phase_idx:03}/seed0/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_dir) / f'bispectrum_hod{hod:03}.npy'
            np.save(save_fn, {'k123': k123, 'bk': bk})
