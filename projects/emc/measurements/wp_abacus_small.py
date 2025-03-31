import fitsio
from pathlib import Path
import argparse
import numpy as np
from pycorr import TwoPointCorrelationFunction
from acm import setup_logging
from cosmoprimo.fiducial import AbacusSummit


def get_hod_positions(input_fn, los='z'):
    hod = fitsio.read(input_fn)
    pos = np.c_[hod['X'], hod['Y'], hod['Z']]
    hubble = 100 * cosmo.efunc(redshift)
    scale_factor = 1 / (1 + redshift)
    pos += boxsize / 2
    if los == 'x':
        pos[:, 0] += hod['VX'] / (hubble * scale_factor)
    elif los == 'y':
        pos[:, 1] += hod['VY'] / (hubble * scale_factor)
    elif los == 'z':
        pos[:, 2] += hod['VZ'] / (hubble * scale_factor)
    return pos

def compute_tpcf(data_positions, edges, boxsize, nthreads=4, gpu=True, los='z'):
    return TwoPointCorrelationFunction(
        'rppi', edges=edges, data_positions1=data_positions, wrap=True,
        engine='corrfunc', boxsize=boxsize, nthreads=nthreads, gpu=gpu,
        compute_sepsavg=False, position_type='pos', los=los,
    )

parser = argparse.ArgumentParser()
parser.add_argument("--start_phase", type=int, default=0)
parser.add_argument("--n_phase", type=int, default=1)

args = parser.parse_args()
start_phase = args.start_phase
n_phase = args.n_phase

# hod = 9971
hod = 466
phases = list(range(start_phase, start_phase + n_phase)) 

boxsize = 500
rp_edges = np.logspace(-1, 1.5, num = 19, endpoint = True, base = 10.0)
pi_edges = np.linspace(-40, 40, 41)
edges = (rp_edges, pi_edges)
los = 'z'

cosmo = AbacusSummit(0)
redshift = 0.5

setup_logging()

for phase in phases:
    hod_dir = f'/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior/small/hod{hod:03}/'
    hod_fn = Path(hod_dir) / f'ph{phase:03}_hod{hod:03}.fits'
    if not hod_fn.exists():
        print(f'{hod_fn} not found')
        continue

    hod_positions = get_hod_positions(hod_fn, los=los)

    tpcf = compute_tpcf(hod_positions, edges, boxsize, nthreads=128, gpu=False, los=los)

    save_dir = '/pscratch/sd/e/epaillas/emc/covariance_sets/xi_rppi/z0.5/yuan23_prior/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / f'xi_rppi_ph{phase:03}_hod{hod:03}.npy'
    tpcf.save(save_fn)