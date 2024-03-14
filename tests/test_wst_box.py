import numpy as np
from pathlib import Path
from acm.estimators import WaveletScatteringTransform
from acm import setup_logging
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

setup_logging()

data_dir = '/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior/c000_ph000'
data_fn = Path(data_dir) / 'hod3177.npy'
data = np.load(data_fn, allow_pickle=True).item()

# read positions
boxsize = 2000.0
x = data['x']
y = data['y']
z_rsd = data['z_rsd']
data_positions = np.c_[x, y, z_rsd]

wst = WaveletScatteringTransform(data_positions=data_positions, boxsize=boxsize,
                                 boxcenter=boxsize/2, nthreads=1, cellsize=16.0,
                                 device='gpu', wrap=True)

wst.get_delta_mesh()
wst.get_wst()