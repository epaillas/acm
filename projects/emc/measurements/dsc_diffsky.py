from pathlib import Path
from acm.estimators.galaxy_clustering.density_split import DensitySplit
from acm.utils import setup_logging
from cosmoprimo.cosmology import Cosmology
import numpy as np
from astropy.table import Table
import logging

logger = logging.getLogger('densitysplit_diffsky')
setup_logging()

def read_lrg(filename, apply_rsd=True, los='z',):
    data = Table.read(filename)
    pos = data['pos']
    if apply_rsd:
        vel = data['vel']
        pos_rsd = (pos + vel / (hubble * scale_factor)) % boxsize
        los_dict = {'x': 0, 'y': 1, 'z': 2}
        pos[:, los_dict[los]] = pos_rsd[:, los_dict[los]]
    is_lrg = data["diffsky_isLRG"].astype(bool)
    return pos[is_lrg]


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

for phase_idx in phases:
    # read simulation
    data_dir = '/global/cfs/cdirs/desicollab/users/gbeltzmo/C3EMC/UNIT'
    data_fn = Path(data_dir) / f'galsampled_diffsky_mock_{adict[redshift]}_fixedAmp_{phase_idx:03}_{galsample}_v{version}.hdf5'
    data_positions = read_lrg(data_fn, apply_rsd=True, los='z')

    # calculate density splits
    smoothing_radius = 10
    cellsize = 4.0
    nquantiles = 5
    ds = DensitySplit(boxsize=boxsize, boxcenter=boxsize/2, cellsize=cellsize)
    ds.assign_data(positions=data_positions, wrap=True)
    ds.set_density_contrast(smoothing_radius=smoothing_radius)
    ds.set_quantiles(nquantiles=nquantiles, query_method='randoms')

    # calculate clustering
    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    ccf = ds.quantile_data_correlation(data_positions, edges=(sedges, muedges), los=los, nthreads=128, gpu=False)
    acf = ds.quantile_correlation(edges=(sedges, muedges), los=los, nthreads=128, gpu=False)

    # save the results
    output_dir = f'/pscratch/sd/e/epaillas/emc/data_vectors/diffsky/dsc_conf/z{redshift}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_fn = Path(output_dir) / f'quantile_data_correlation_galsampled_diffsky_mock_{adict[redshift]}_fixedAmp_{phase_idx:03}_{galsample}_v{version}.npy'
    np.save(output_fn, ccf)

    output_dir = f'/pscratch/sd/e/epaillas/emc/data_vectors/diffsky/dsc_conf/z{redshift}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_fn = Path(output_dir) / f'quantile_correlation_galsampled_diffsky_mock_{adict[redshift]}_fixedAmp_{phase_idx:03}_{galsample}_v{version}.npy'
    np.save(output_fn, acf)
        
