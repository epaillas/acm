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
    pos += boxsize / 2
    if los == 'x':
        pos[:, 0] += hod['VX'] / (hubble * scale_factor)
    elif los == 'y':
        pos[:, 1] += hod['VY'] / (hubble * scale_factor)
    elif los == 'z':
        pos[:, 2] += hod['VZ'] / (hubble * scale_factor)
    return pos


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
ds = DensitySplit(boxsize=boxsize, boxcenter=boxsize/2, cellsize=4.0)

for phase in phases:
    hod_dir = '/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior/small'
    hod_fn = Path(hod_dir) / f'ph{phase:03}_hod{hod:03}.fits'
    if not hod_fn.exists():
        print(f'{hod_fn} not found')
        continue

    hod_positions = get_hod_positions(hod_fn, los=los)

    ds.assign_data(positions=hod_positions, wrap=True, clear_previous=True)
    ds.set_density_contrast(smoothing_radius=10, save_wisdom=True)
    ds.set_quantiles(nquantiles=5, query_method='randoms')

    ccf = ds.quantile_data_correlation(hod_positions, edges=edges, los=los, wrap=True, nthreads=256, gpu=False)
    acf = ds.quantile_correlation(edges=edges, los=los, wrap=True, nthreads=256, gpu=False)

    save_dir = '/pscratch/sd/e/epaillas/emc/covariance_sets/density_split/quantile_data_correlation/z0.5/yuan23_prior/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / f'quantile_data_correlation_ph{phase:03}_hod{hod:03}.npy'
    print(f'Saving {save_fn}')
    np.save(save_fn, ccf)

    save_dir = '/pscratch/sd/e/epaillas/emc/covariance_sets/density_split/quantile_correlation/z0.5/yuan23_prior/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / f'quantile_correlation_ph{phase:03}_hod{hod:03}.npy'
    print(f'Saving {save_fn}')
    np.save(save_fn, acf)