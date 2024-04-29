from pathlib import Path
from acm.utils import setup_logging
from pypower import mpi
import numpy as np
import matplotlib.pyplot as plt

setup_logging()

def read_mock_catalog():
    # read some random galaxy catalog
    data_dir = '/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior2/c000_ph000/'
    data_fn = Path(data_dir) / 'hod3177.npy'
    data = np.load(data_fn, allow_pickle=True).item()
    boxsize = 2000.0
    x = data['x']
    y = data['y']
    z_rsd = data['z_rsd']
    data_positions = np.c_[x, y, z_rsd]
    return data_positions, boxsize

def test_density_split():
    from estimators.galaxy_clustering.density_split import DensitySplit
    data_positions, boxsize = read_mock_catalog()
    ds = DensitySplit(boxsize=boxsize, boxcenter=boxsize/2, cellsize=5.0)
    ds.assign_data(positions=data_positions)
    ds.set_density_contrast(smoothing_radius=10)
    ds.set_quantiles(nquantiles=3, query_method='randoms')
    sedges = np.linspace(0, 150, 100)
    muedges = np.linspace(-1, 1, 241)
    ds.quantile_data_correlation(data_positions, edges=(sedges, muedges), los='z', nthreads=256)
    ds.quantile_correlation(edges=(sedges, muedges), los='z', nthreads=256)

    ds.plot_quantiles()
    ds.plot_quantile_data_correlation(ell=0)
    ds.plot_quantile_correlation(ell=0)

def test_wst():
    from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform
    data_positions, boxsize = read_mock_catalog()
    wst = WaveletScatteringTransform(boxsize=boxsize, boxcenter=boxsize/2, cellsize=5.0)
    wst.assign_data(positions=data_positions)
    wst.set_density_contrast()
    wst.run()
    wst.plot_coefficients()

def test_voxel():
    data_positions, boxsize = read_mock_catalog()
    voxel = VoxelVoids(boxsize=boxsize, boxcenter=boxsize/2, cellsize=5.0,
                       temp_dir='/pscratch/sd/e/epaillas/tmp')
    voxel.assign_data(positions=data_positions)
    voxel.set_density_contrast(smoothing_radius=10)
    voxel.find_voids()
    # sedges = np.linspace(0, 150, 100)
    # muedges = np.linspace(-1, 1, 241)
    # voxel.void_data_correlation(data_positions, edges=(sedges, muedges), los='z', nthreads=256)
    # voxel.plot_void_data_correlation(ells=(0, 2))
    # voxel.plot_void_size_distribution()
    voxel.plot_slice(data_positions=data_positions, save_fn='slice.png')

def test_minkowski():
    from acm.estimators.galaxy_clustering import MinkowskiFunctionals
    data_positions, boxsize = read_mock_catalog()
    mf = MinkowskiFunctionals(boxsize=boxsize, boxcenter=boxsize/2, cellsize=5.0)
    mf.assign_data(positions=data_positions)
    mf.set_density_contrast(smoothing_radius=10)
    mf.run()

if __name__ == '__main__':
    # test_density_split()
    # test_wst()
    # test_voxel()
    test_minkowski()
