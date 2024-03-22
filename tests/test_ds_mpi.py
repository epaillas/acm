from pathlib import Path
from acm.estimators import WaveletScatteringTransform, DensitySplit
from acm.utils import setup_logging
from pypower import mpi
import numpy as np

setup_logging()

# set up MPI
mpicomm = mpi.COMM_WORLD
mpiroot = 0

# read some random galaxy catalog
data_dir = '/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior2/c000_ph000/'
data_fn = Path(data_dir) / 'hod3177.npy'
data = np.load(data_fn, allow_pickle=True).item()
boxsize = 2000.0
x = data['x']
y = data['y']
z_rsd = data['z_rsd']
data_positions = np.c_[x, y, z_rsd]

# calculate density splits
ds = DensitySplit(boxsize=boxsize, boxcenter=boxsize/2, cellsize=16.0)
ds.assign_data(positions=data_positions)
ds.set_density_contrast(smoothing_radius=10, query_method='randoms')
ds.set_quantiles(nquantiles=3)

# calculate clustering in Fourier space
edges = np.linspace(0.02, 0.5, 100)
ds.quantile_data_power(data_positions, edges=edges, los='z', nmesh=1024, mpicomm=mpicomm, mpiroot=mpiroot)
ds.quantile_power(edges=edges, los='z', nmesh=1024, mpicomm=mpicomm, mpiroot=mpiroot)

# save figures
if mpicomm.rank == mpiroot:
    ds.plot_quantile_data_power(save_fn='quantile_data_power.png')
    ds.plot_quantile_power(save_fn='quantile_power.png')
