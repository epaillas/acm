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

def get_distorted_positions(positions, q_perp, q_para, los='z'):
    """Given a set of comoving galaxy positions in cartesian
    coordinates, return the positions distorted by the
    Alcock-Pacynski effect"""
    positions_ap = np.copy(positions)
    factor_x = q_para if los == 'x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp
    positions_ap[:, 0] /= factor_x
    positions_ap[:, 1] /= factor_y
    positions_ap[:, 2] /= factor_z
    return positions_ap

def get_distorted_box(boxsize, q_perp, q_para, los='z'):
    """Distort the dimensions of a cubic box with the
    Alcock-Pacynski effect"""
    factor_x = q_para if los == 'x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp
    boxsize_ap = [boxsize/factor_x, boxsize/factor_y, boxsize/factor_z]
    return boxsize_ap 


def compute_tpcf(data_positions, edges, boxsize, nthreads=4, gpu=True, los='z'):
    return TwoPointCorrelationFunction(
        'smu', edges=edges, data_positions1=data_positions,
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
    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)

    fid_cosmo = AbacusSummit(0)

    for cosmo_idx in cosmos:
        for phase_idx in phases:
            for hod_idx in hods:
                for seed in seeds:
                    hod_dir = f'/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z0.5/yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed}/'
                    hod_fn = Path(hod_dir) / f'hod{hod_idx:03}.fits'
                    if not hod_fn.exists():
                        continue
                    logger.info(f'Reading {hod_fn}')

                    cosmo = AbacusSummit(cosmo_idx)

                    # calculate distortion parameters
                    q_perp = cosmo.comoving_angular_distance(redshift) / fid_cosmo.comoving_angular_distance(redshift)
                    q_para = fid_cosmo.efunc(redshift) / cosmo.efunc(redshift)
                    q = q_perp**(2/3) * q_para**(1/3)
                    print(f'q_perp = {q_perp:.3f}')
                    print(f'q_para = {q_para:.3f}')
                    print(f'q = {q:.3f}')

                    for i, los in enumerate(['x', 'y', 'z']):
                        data_positions = get_hod_positions(hod_fn, los=los)

                        data_positions_ap = get_distorted_positions(positions=data_positions, los=los,
                                                                q_perp=q_perp, q_para=q_para)
                        boxsize_ap = np.array(get_distorted_box(boxsize=boxsize, q_perp=q_perp, q_para=q_para,
                                                                los=los))

                        tpcf_los = compute_tpcf(data_positions_ap, edges, boxsize_ap, nthreads=4, gpu=True, los=los).normalize()
                        if i == 0:
                            tpcf = tpcf_los
                        else:
                            tpcf += tpcf_los

                    # save the results
                    output_dir = f'/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/raw/tpcf/ap/c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed}/'
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_fn = Path(output_dir) / f'tpcf_hod{hod_idx:03}.npy'
                    tpcf.save(output_fn)
