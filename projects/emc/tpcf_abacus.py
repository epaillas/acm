import yaml
import numpy as np
import argparse
import fitsio
from pathlib import Path
from abacusnbody.hod.abacus_hod import AbacusHOD
from pathlib import Path
from pypower import setup_logging
from pycorr import TwoPointCorrelationFunction
from cosmoprimo.fiducial import AbacusSummit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def get_hod_positions(hod_idx, los='z'):
    data_dir = f'/pscratch/sd/e/epaillas/emc/hods/z{redshift}/yuan23_prior3/acm/c000_ph000'
    data_fn = Path(data_dir) / f'hod{hod_idx:03}.fits'
    data = fitsio.read(data_fn)
    if los == 'x':
        return np.c_[data['X_RSD'], data['Y'], data['Z']]
    elif los == 'y':
        return np.c_[data['X'], data['Y_RSD'], data['Z']]
    elif los == 'z':
        return np.c_[data['X'], data['Y'], data['Z_RSD']]


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
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    start_hod = args.start_hod
    n_hod = args.n_hod
    start_cosmo = args.start_cosmo
    n_cosmo = args.n_cosmo
    start_phase = args.start_phase
    n_phase = args.n_phase

    setup_logging(level='INFO')
    boxsize = 2000
    redshift = 0.5
    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)

    for cosmo in range(start_cosmo, start_cosmo + n_cosmo):
        for phase in range(start_phase, start_phase + n_phase):

            fig, ax = plt.subplots()
            for hod in range(start_hod, start_hod + n_hod):
                print(f'c{cosmo:03} ph{phase:03} hod{hod}')

                for i, los in enumerate(['x', 'y', 'z']):
                    data_positions = get_hod_positions(hod, los=los)
                    tpcf_los = compute_tpcf(data_positions, edges, boxsize, nthreads=4, gpu=True, los=los).normalize()
                    if i == 0:
                        tpcf = tpcf_los
                    else:
                        tpcf += tpcf_los

                # save the results
                output_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/test/tpcf/z0.5/{hod_prior}_prior/c{cosmo:03}_ph{phase:03}'
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                output_fn = Path(output_dir) / f'tpcf_hod{hod:03}.npy'
                tpcf.save(output_fn)

            plt.show()

