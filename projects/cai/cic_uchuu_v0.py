from pathlib import Path
from acm.estimators.galaxy_clustering.cic import CountsInCells
from acm.utils import setup_logging
import numpy as np
from cosmoprimo.fiducial import DESI
from pyrecon.utils import sky_to_cartesian
import fitsio


def read_uchuu_data(complete=False):
    data_dir = '/pscratch/sd/e/efdez/Uchuu/LRG/v0'
    data_fn = Path(data_dir) / 'Uchuu-SHAM_LRG_Y3-v1_0000.fits'
    return fitsio.read(data_fn)

def read_uchuu_randoms(irand=0):
    data_dir = '/pscratch/sd/e/efdez/Uchuu/LRG/v0'
    data_fn = Path(data_dir) / f'Uchuu-SHAM_LRG_Y3-v1_{irand}.ran.fits'
    return fitsio.read(data_fn)

def get_clustering_positions_weights(data, cattype='uchuu'):
    if region == 'NGC':
        region_mask = (data['RA'] > 80) & (data['RA'] < 300)
    else:
        region_mask = (data['RA'] < 80) | (data['RA'] > 300)
    zmask = (data['Z'] > zmin) & (data['Z'] < zmax)
    mask = region_mask & zmask
    ra = data[mask]['RA']
    dec = data[mask]['DEC']
    dist = distance(data[mask]['Z'])
    pos = sky_to_cartesian(ra=ra, dec=dec, dist=dist)
    weights = np.ones(len(pos))
    return pos, weights, mask

def get_cic_uchuu(smoothing_radius=5.0):
    data = read_uchuu_data()
    data_positions, data_weights, data_mask = get_clustering_positions_weights(data)

    cic = CountsInCells(positions=data_positions, cellsize=4.0, boxpad=1.2)
    cic.assign_data(positions=data_positions, weights=data_weights)

    for irand in range(4):
        randoms = read_uchuu_randoms(irand)
        randoms_positions, randoms_weights, randoms_mask = get_clustering_positions_weights(randoms)
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
    zmin, zmax = 0.8, 1.1
    region = 'NGC'
    cosmo = DESI()
    distance = cosmo.comoving_radial_distance
    for smoothing_radius in [5.0, 7.5, 10.0]:
    
        cic_uchuu = get_cic_uchuu(smoothing_radius=smoothing_radius)

        save_dir = '/pscratch/sd/e/epaillas/cic/'
        save_fn = Path(save_dir) / f'CIC_LRG_{region}_Uchuu_Y3v0_z{zmin}-{zmax}_sm{smoothing_radius}.npy'
        np.save(save_fn, cic_uchuu)
