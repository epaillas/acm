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


def read_desi_data(mock_id=0, region="NGC", fiber_asign=None):
    if fiber_asign is None:
        data_fn = data_dir / f"mock{mock_id}/LRG_complete_{region}_clustering.dat.fits"
    elif fiber_asign == "altmtl":
        data_fn = (
            data_dir
            / f"altmtl{mock_id}/mock{mock_id}/LSScats/LRG_{region}_clustering.dat.fits"
        )
    elif fiber_asign == "ffa":
        data_fn = data_dir / f"mock{mock_id}/LRG_ffa_{region}_clustering.dat.fits"
    print(f'Reading {data_fn}')
    return fitsio.read(data_fn)


def read_desi_randoms(mock_id=0, irand=0, region="NGC", fiber_asign=None):
    if fiber_asign is None:
        data_fn = (
            data_dir
            / f"mock{mock_id}/LRG_complete_{region}_{irand}_clustering.ran.fits"
        )
    elif fiber_asign == "altmtl":
        data_fn = (
            data_dir
            / f"altmtl{mock_id}/mock{mock_id}/LSScats/LRG_{region}_{irand}_clustering.ran.fits"
        )
    elif fiber_asign == "ffa":
        data_fn = (
            data_dir / f"mock{mock_id}/LRG_ffa_{region}_{irand}_clustering.ran.fits"
        )
    return fitsio.read(data_fn)

def get_clustering_positions_weights(data, cattype='uchuu'):
    mask = (data['Z'] > zmin) & (data['Z'] < zmax)
    ra = data[mask]['RA']
    dec = data[mask]['DEC']
    dist = distance(data[mask]['Z'])
    pos = sky_to_cartesian(ra=ra, dec=dec, dist=dist)
    weights = data[mask]['WEIGHT']
    # weights = np.ones(len(pos))
    return pos, weights, mask

def get_cic_data(smoothing_radius=5.0, fiber_asign=None):
    data = read_desi_data(fiber_asign=fiber_asign)
    data_positions, data_weights, data_mask = get_clustering_positions_weights(data)

    cic = CountsInCells(positions=data_positions, cellsize=4.0, boxpad=1.2)
    cic.assign_data(positions=data_positions, weights=data_weights)

    for irand in range(18):
        randoms = read_desi_randoms(irand, fiber_asign=fiber_asign)
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
    zmin, zmax = 0.4, 0.6
    region = 'NGC'
    cosmo = DESI()
    distance = cosmo.comoving_radial_distance
    smoothing_radius = 10
    data_dir = Path(
        "/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/"
    )
    
    for fiber_asign in ['altmtl', 'ffa', None]:
        cic_data = get_cic_data(smoothing_radius=smoothing_radius, fiber_asign=fiber_asign)

        save_dir = '/pscratch/sd/e/epaillas/dsc-desi/fibers/'
        fa = 'complete' if fiber_asign is None else fiber_asign
        save_fn = Path(save_dir) / f'CIC_LRG_{fa}_{region}_z{zmin}-{zmax}_sm{smoothing_radius}.npy'
        np.save(save_fn, cic_data)
