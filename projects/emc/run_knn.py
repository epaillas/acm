from pathlib import Path
import pandas as pd
from acm.estimators.galaxy_clustering import KthNearestNeighbor
from acm.utils import setup_logging
from pypower import mpi
import numpy as np

setup_logging()

def get_data(data_dir, cosmology, phase, hod, boxsize):
    # read some random galaxy catalog
    data_dir = data_dir / f'hods/cosmo+hod/z0.5/yuan23_prior/c{str(cosmology).zfill(3)}_ph{str(phase).zfill(3)}/seed0/'
    data_fn = data_dir / f'hod{str(hod).zfill(3)}.fits'
    data = np.load(data_fn, allow_pickle=True).item()
    x = data['x'] % boxsize
    y = data['y'] % boxsize
    z_rsd = data['z_rsd'] % boxsize
    data_positions = np.c_[x, y, z_rsd]
    return data_positions

def get_randoms(n_randoms=10000, boxsize=2000.):
    # generate some query points
    xrand = np.random.random(n_randoms)*boxsize
    yrand = np.random.random(n_randoms)*boxsize
    zrand = np.random.random(n_randoms)*boxsize
    pos_rand = np.vstack((xrand, yrand, zrand)).T
    return pos_rand


if __name__ == '__main__':
    # define ks, rs, pis 
    n_hods = 1000
    boxsize = 2000.
    ks = [1, 2,3,4,5,6,7,8,9]
    rp_start, rp_end = 0.63, 63
    pi_start, pi_end = 0.5, 32
    rps = np.logspace(np.log10(rp_start), np.log10(rp_end), 8)
    pis = np.logspace(np.log10(pi_start), np.log10(pi_end), 5)
    knn = KthNearestNeighbor(boxsize=boxsize, boxcenter=boxsize/2, cellsize=5.0)
    data_dir = Path(f'/pscratch/sd/e/epaillas/emc/')


    cosmologies = list(range(4)) + list(range(100, 182))
    lhc_x, lhc_y = [], []
    for cosmology in cosmologies:
        params = pd.read_csv(data_dir / f'cosmo+hod_params/AbacusSummit_c{str(cosmology).zfill(3)}.csv')
        lhc_x.append(params.values[:n_hods])
        for hod in range(n_hods):
            print(f'Doing hod = {hod}')
            data_positions = get_data(data_dir, cosmology, 0, hod, boxsize)
            pos_rand = get_randoms(n_randoms=10000, boxsize=boxsize)
            lhc_y.append(knn.run_knn(rps, pis, data_positions, pos_rand, kneighbors = ks))

    # read cosmo + hod paramters
    lhc_y = np.array(lhc_y)
    print('lhc_y = ', lhc_y.shape)
    lhc_x = np.concatenate(lhc_x)
    print('lhc_x = ', lhc_x.shape)

    rps_c = 0.5*(rps[1:]+rps[:-1])
    pis_c = 0.5*(pis[1:]+pis[:-1])
    cout = {
        'rps': rps_c, 
        'pis': pis_c,  
        'lhc_x': lhc_x, 
        'lhc_y': lhc_y, 
        'lhc_x_names': lhc_x_names,
    }
    np.save('/pscratch/sd/c/cuesta/emc_data/knn/knn_lhc.npy', cout)
