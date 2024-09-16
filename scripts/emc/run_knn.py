from pathlib import Path
import pandas as pd
from acm.estimators.galaxy_clustering import KthNearestNeighbor
from acm.utils import setup_logging
from pypower import mpi
import numpy as np

setup_logging()

def get_data(cosmology, phase, hod, boxsize):
    # read some random galaxy catalog
    data_dir = f'/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior2/c{str(cosmology).zfill(3)}_ph{str(phase).zfill(3)}/'
    data_fn = Path(data_dir) / f'hod{str(hod).zfill(3)}.npy'
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
    rps = np.logspace(-0.2, 1.8, 20)
    pis = np.logspace(-0.3, 1.5, 15)
    knn = KthNearestNeighbor(boxsize=boxsize, boxcenter=boxsize/2, cellsize=5.0)

    lhc_y = []
    for hod in range(n_hods):
        print(f'Doing hod = {hod}')
        data_positions = get_data(0, 0, hod, boxsize)
        pos_rand = get_randoms(n_randoms=10000, boxsize=boxsize)
        lhc_y.append(knn.run_knn(rps, pis, data_positions, pos_rand, kneighbors = ks))
    lhc_y = np.array(lhc_y)
    lhc_x = pd.read_csv('/pscratch/sd/e/epaillas/emc/hod_params/yuan23/hod_params_yuan23_c000.csv')
    lhc_x_names = list(lhc_x.columns)
    lhc_x_names = [name.replace(' ', '').replace('#', '') for name in lhc_x_names]
    lhc_x = lhc_x.values[:len(lhc_y),:]

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