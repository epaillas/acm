import numpy as np
import argparse
import fitsio
from pathlib import Path
from pathlib import Path
from pypower import setup_logging
from pycorr import TwoPointCorrelationFunction
from cosmoprimo.fiducial import AbacusSummit
import logging
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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


def compute_tpcf(data_positions, edges, boxsize, nthreads=4, gpu=True, los='z'):
    return TwoPointCorrelationFunction(
        'rppi', edges=edges, data_positions1=data_positions,
        engine='corrfunc', boxsize=boxsize, nthreads=nthreads, gpu=gpu,
        compute_sepsavg=False, position_type='pos', los=los,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--n_seed", type=int, default=1)

    args = parser.parse_args()

    hods = list(range(args.start_hod, args.start_hod + args.n_hod))
    phases = list(range(args.start_phase, args.start_phase + args.n_phase))
    cosmos = list(range(args.start_cosmo, args.start_cosmo + args.n_cosmo))
    seeds = list(range(args.start_seed, args.start_seed + args.n_seed))

    setup_logging(level='INFO')
    logger = logging.getLogger('tpcf_abacus')
    boxsize = 2000
    redshift = 0.5
    seed = 0
    rp_edges = np.logspace(-1, 1.5, num = 19, endpoint = True, base = 10.0)
    pi_edges = np.linspace(-40, 40, 41)
    edges = (rp_edges, pi_edges)
    # edges = np.logspace(-1, 1.5, num = 19, endpoint = True, base = 10.0)

    for cosmo_idx in cosmos:
        for phase_idx in phases:
            for hod_idx in hods:
                hod_dir = f'/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z0.5/yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed}/'
                hod_fn = Path(hod_dir) / f'hod{hod_idx:03}.fits'
                if not hod_fn.exists():
                    # print(f'{hod_fn} does not exist')
                    continue
                logger.info(f'Reading {hod_fn}')

                cosmo = AbacusSummit(cosmo_idx)

                for i, los in enumerate(['x', 'y', 'z']):
                    data_positions = get_hod_positions(hod_fn, los=los)
                    tpcf_los = compute_tpcf(data_positions, edges, boxsize, nthreads=128, gpu=False, los=los).normalize()
                    if i == 0:
                        tpcf = tpcf_los
                    else:
                        tpcf += tpcf_los

                # save the results
                output_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/xi_rppi/cosmo+hod/z0.5/yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed}/'
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                output_fn = Path(output_dir) / f'xi_rppi_hod{hod_idx:03}.npy'
                tpcf.save(output_fn)