import fitsio
from pathlib import Path
import argparse
import numpy as np
from acm.estimators.galaxy_clustering import DensitySplit
from acm import setup_logging
from cosmoprimo.fiducial import AbacusSummit


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


sedges = np.arange(0, 201, 1)
muedges = np.linspace(-1, 1, 241)
edges = (sedges, muedges)

cosmo = AbacusSummit(0)
redshift = 0.5

# setup_logging()
ds = DensitySplit(boxsize=2000, boxcenter=2000/2, cellsize=4.0)

# load the HODs
for hod in range(start_hod, start_hod + n_hod):
    print('Processing HOD', hod)
    hod_dir = '/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior/c000_ph000'
    hod_fn = Path(hod_dir) / f'hod{hod:03}.fits'

    for i, los in enumerate(['x', 'y', 'z']):
        hod_positions = get_hod_positions(hod_fn, los=los)

        ds.assign_data(positions=hod_positions, wrap=True, clear_previous=True)
        ds.set_density_contrast(smoothing_radius=10, save_wisdom=True)
        ds.set_quantiles(nquantiles=5, query_method='randoms')

        ccf_los = ds.quantile_data_correlation(hod_positions, edges=edges, los=los, nthreads=4, gpu=True)
        acf_los = ds.quantile_correlation(edges=edges, los=los, nthreads=4, gpu=True)

        if i == 0:
            ccf = [ccf.normalize() for ccf in ccf_los]
            acf = [acf.normalize() for acf in acf_los]
        else:
            for q in range(5):
                ccf[q] += ccf_los[q].normalize()
                acf[q] += acf_los[q].normalize()

    save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/density_split/quantile_data_correlation/z0.5/yuan23_prior/c000_ph000/seed0'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / f'quantile_data_correlation_hod{hod:03}.npy'
    np.save(save_fn, ccf)

    save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/density_split/quantile_correlation/z0.5/yuan23_prior/c000_ph000/seed0'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / f'quantile_correlation_hod{hod:03}.npy'
    np.save(save_fn, acf)