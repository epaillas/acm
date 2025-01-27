from pathlib import Path
from acm.estimators.galaxy_clustering.cic import CountsInCells
from acm.utils import setup_logging
import numpy as np
from cosmoprimo.fiducial import DESI
from pyrecon.utils import sky_to_cartesian
import fitsio


def read_data():
    if mocks == 'AbacusSummit':
        data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/altmtl{mock_idx}/mock{mock_idx}/LSScats'
        data_fn = Path(data_dir) / f'{tracer}_{region}_clustering.dat.fits'
    elif mocks == 'EZmock':
        data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/FFA/mock{mock_idx}/'
        data_fn = Path(data_dir) / f'{tracer}_ffa_{region}_clustering.dat.fits'
    print(f'Reading {data_fn}')
    return fitsio.read(data_fn)

def read_randoms(rnd_idx=0):
    if mocks == 'AbacusSummit':
        data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/altmtl{mock_idx}/mock{mock_idx}/LSScats'
        data_fn = Path(data_dir) / f'{tracer}_{region}_{rnd_idx}_clustering.ran.fits'
    elif mocks == 'EZmock':
        data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/FFA/mock{mock_idx}/'
        data_fn = Path(data_dir) / f'{tracer}_ffa_{region}_{rnd_idx}_clustering.ran.fits'
    return fitsio.read(data_fn)

def get_clustering_positions_weights(data):
    mask = (data['Z'] > zmin) & (data['Z'] < zmax)
    ra = data[mask]['RA']
    dec = data[mask]['DEC']
    dist = distance(data[mask]['Z'])
    pos = sky_to_cartesian(ra=ra, dec=dec, dist=dist)
    weights = data[mask]['WEIGHT']
    return pos, weights, mask

def get_cic(smoothing_radius=5.0):
    data = read_data()
    data_positions, data_weights, data_mask = get_clustering_positions_weights(data)

    cic = CountsInCells(positions=data_positions, cellsize=5.0, boxpad=1.2)
    cic.assign_data(positions=data_positions, weights=data_weights)

    for rnd_idx in range(4):
        randoms = read_desi_randoms(rnd_idx)
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
    tracer = 'LRG'
    zmin, zmax = 0.4, 0.6
    region = 'NGC'
    cosmo = DESI()
    distance = cosmo.comoving_radial_distance

    mock_idx = 1  # EZmock realization

    for smoothing_radius in [5.0, 7.5, 10.0]:
        cic = get_cic(smoothing_radius=smoothing_radius)

        save_fn = Path(save_dir) / f'CIC_{tracer}_{region}_EZmock_z{zmin}-{zmax}_sm{smoothing_radius}.npy'
        np.save(save_fn, cic)
