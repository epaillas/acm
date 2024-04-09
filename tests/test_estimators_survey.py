from pathlib import Path
import fitsio
from cosmoprimo.fiducial import DESI
from pyrecon.utils import sky_to_cartesian, setup_logging
from acm.estimators import WaveletScatteringTransform, VoxelVoids
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_desi(filename, distance, zmin=0.45, zmax=0.6):
    """Read CMASS LSS catalogues."""
    data = fitsio.read(filename)
    mask = (data['Z'] > zmin) & (data['Z'] < zmax)
    ra = data[mask]['RA']
    dec = data[mask]['DEC']
    redshift = data[mask]['Z']
    weights = data[mask]['WEIGHT']
    dist = distance(redshift)
    positions = sky_to_cartesian(dist=dist, ra=ra, dec=dec)
    return positions, weights




def test_wst():
    wst = WaveletScatteringTransform(positions=randoms_positions, cellsize=15.0)
    wst.assign_data(positions=data_positions, weights=data_weights)
    wst.assign_randoms(positions=randoms_positions, weights=randoms_weights)
    wst.set_density_contrast()
    smatavg = wst.run()

def test_voxel():
    voxel = VoxelVoids(positions=randoms_positions, cellsize=5.0, temp_dir='/pscratch/sd/e/epaillas/tmp')
    voxel.assign_data(positions=data_positions, weights=data_weights)
    voxel.assign_randoms(positions=randoms_positions, weights=randoms_weights)
    voxel.set_density_contrast(smoothing_radius=10)
    voxel.find_voids()
    voxel.plot_slice(data_positions=data_positions, save_fn='slice.png')
    # sedges = np.linspace(0, 150, 100)
    # muedges = np.linspace(-1, 1, 241)
    # voxel.void_data_correlation(data_positions=data_positions, randoms_positions=randoms_positions,
    #                             data_weights=data_weights, randoms_weights=randoms_weights,
    #                             edges=(sedges, muedges), los='midpoint', nthreads=256)
    # voxel.plot_void_data_correlation(ells=(0, 2), save_fn='void_data_correlation.png')


if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger('test_wst_survey')

    # define the cosmology
    cosmo = DESI()
    distance = cosmo.comoving_radial_distance
    zmin, zmax = 0.4, 0.5
    version = 'v1.2/blinded'
    tracer = 'LRG'
    region = 'NGC'
    nrandoms = 1

    # read data
    data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{version}'
    data_fn = Path(data_dir) / f'{tracer}_{region}_clustering.dat.fits'
    logger.info(f'Reading {data_fn}')
    data_positions, data_weights = read_desi(distance=distance, filename=data_fn,
                                            zmin=zmin, zmax=zmax)
    # read randoms
    randoms_positions = []
    randoms_weights = []
    for i in range(nrandoms):
        randoms_dir = Path(f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{version}')
        randoms_fn = randoms_dir / f'{tracer}_{region}_{i}_clustering.ran.fits'
        logger.info(f'Reading {randoms_fn}')
        randoms_positions_i, randoms_weights_i = read_desi(distance=distance, filename=randoms_fn,
                                                        zmin=zmin, zmax=zmax)
        randoms_positions.append(randoms_positions_i)
        randoms_weights.append(randoms_weights_i)
    randoms_positions = np.concatenate(randoms_positions)
    randoms_weights = np.concatenate(randoms_weights)

    # test_wst()
    test_voxel()