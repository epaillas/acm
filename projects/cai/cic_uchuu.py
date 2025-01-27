from pathlib import Path
from acm.estimators.galaxy_clustering.cic import CountsInCells
from acm.utils import setup_logging
import numpy as np
from cosmoprimo.fiducial import DESI
from pyrecon.utils import sky_to_cartesian
import fitsio
import matplotlib.pyplot as plt
import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def read_uchuu_data(complete=False):
    data_dir = '/pscratch/sd/e/efdez/Uchuu/Y1/'
    if complete: 
        data_fn = Path(data_dir) / 'LRG_complete_clustering.dat.fits'
        return fitsio.read(data_fn)
    else:
        data_fn = Path(data_dir) / 'uchuu-desi-y3_v2_0p65_mask.txt'
        data = np.genfromtxt(data_fn, usecols=(-3, -2, -1))
        ra = data[:, 0]
        dec = data[:, 1]
        z = data[:, 2]
        return np.c_[ra, dec, z][region_mask]

def read_uchuu_randoms():
    data_dir = '/pscratch/sd/e/efdez/Uchuu/Y1/'
    data_fn = Path(data_dir) / 'random_mask_y3_v2_0p65.txt'
    data = np.genfromtxt(data_fn, usecols=(0, 1, 2))
    ra = data[:, 0]
    dec = data[:, 1]
    z = data[:, 2]
    return np.c_[ra, dec, z]

def read_desi_data():
    data_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y3/LSS/kibo-v1/LSScats/v0.1/PIP'
    data_fn = Path(data_dir) / 'LRG_clustering.dat.fits'
    return fitsio.read(data_fn)

def read_desi_randoms(irand=0):
    data_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y3/LSS/kibo-v1/LSScats/v0.1/PIP'
    data_fn = Path(data_dir) / f'LRG_{irand}_clustering.ran.fits'
    return fitsio.read(data_fn)

def get_clustering_positions_weights(data):
    if cap == 'NGC':
        region_mask = (data[:, 0] > 80) & (data[:, 0] < 300)
    else:
        region_mask = (data[:, 0] < 80) | (data[:, 0] > 300)
    zmask = (data[:, 2] > zmin) & (data[:, 2] < zmax)
    mask = region_mask & zmask
    ra = data[mask][:, 0]
    dec = data[mask][:, 1]
    dist = distance(data[mask][:, 2])
    pos = sky_to_cartesian(ra=ra, dec=dec, dist=dist)
    weights = np.ones(len(pos))
    return pos, weights, mask

def get_cic_uchuu(smoothing_radius=5.0):
    data = read_uchuu_data()
    randoms = read_uchuu_randoms()
    data_positions, data_weights, data_mask = get_clustering_positions_weights(data)
    randoms_positions, randoms_weights, randoms_mask = get_clustering_positions_weights(randoms)

    cic = CountsInCells(positions=data_positions, cellsize=5.0, boxpad=1.2)
    cic.assign_data(positions=data_positions, weights=data_weights)
    cic.assign_randoms(positions=randoms_positions, weights=randoms_weights)
    cic.set_density_contrast(smoothing_radius=smoothing_radius)

    nquery = 5 * len(data_positions)
    idx = np.random.choice(len(randoms_positions), size=nquery)
    query_positions = randoms_positions[idx]
    delta_query = cic.sample_pdf(query_positions=query_positions)
    return delta_query

if __name__ == '__main__':
    # set up
    setup_logging()
    zmin, zmax = 0.4, 0.6
    region = 'NGC'
    cosmo = DESI()
    distance = cosmo.comoving_radial_distance
    for smoothing_radius in [5.0, 7.5, 10.0]:
    
        cic_uchuu = get_cic_uchuu(smoothing_radius=smoothing_radius)

        save_fn = Path(save_dir) / f'CIC_LRG_{region}_Uchuu_Y3_z{zmin}-{zmax}_sm{smoothing_radius}.npy'
        np.save(save_fn, cic_uchuu)