import numpy as np
import argparse
import fitsio
from pathlib import Path
from pathlib import Path
from pypower import setup_logging
from pycorr import TwoPointCorrelationFunction
from cosmoprimo.fiducial import AbacusSummit
import matplotlib.pyplot as plt
import logging
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def get_hod_positions(input_fn, los='z'):
    hod = fitsio.read(input_fn)
    pos = np.c_[hod['X'], hod['Y'], hod['Z']] + boxsize / 2
    hubble = 100 * cosmo.efunc(redshift)
    scale_factor = 1 / (1 + redshift)
    if los == 'x':
        pos[:, 0] += hod['VX'] / (hubble * scale_factor)
    elif los == 'y':
        pos[:, 1] += hod['VY'] / (hubble * scale_factor)
    elif los == 'z':
        pos[:, 2] += hod['VZ'] / (hubble * scale_factor)
    return pos


def compute_tpcf(data_positions, edges, boxsize, nthreads=4, gpu=True, los='z'):
    return TwoPointCorrelationFunction(
        'smu', edges=edges, data_positions1=data_positions, wrap=True,
        engine='corrfunc', boxsize=boxsize, nthreads=nthreads, gpu=gpu,
        compute_sepsavg=False, position_type='pos', los=los,
    )


if __name__ == '__main__':
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
    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)
    los = 'z'

    cosmo = AbacusSummit(0)
    redshift = 0.5

    setup_logging()

    for phase in phases:
        hod_dir = '/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior/small'
        hod_fn = Path(hod_dir) / f'ph{phase:03}_hod{hod:03}.fits'
        if not hod_fn.exists():
            print(f'{hod_fn} not found')
            continue

        hod_positions = get_hod_positions(hod_fn, los=los)
        tpcf = compute_tpcf(hod_positions, edges, boxsize, nthreads=256, gpu=False, los=los)

        save_dir = '/pscratch/sd/e/epaillas/emc/covariance_sets/tpcf/z0.5/yuan23_prior/'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_fn = Path(save_dir) / f'tpcf_ph{phase:03}_hod{hod:03}.npy'
        tpcf.save(save_fn)