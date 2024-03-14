import numpy as np
from pathlib import Path
from acm.estimators import WaveletScatteringTransform
from acm import setup_logging
import matplotlib.pyplot as plt
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

# initialize the WST grid, using the random positions as a reference
wst = WaveletScatteringTransform(boxsize=boxsize, boxcenter=boxsize/2, cellsize=16.0)

# set up the density contrast
wst.assign_data(positions=data_positions)
wst.set_density_contrast()

# get the WST coefficients
smatavg = wst.run()

# plot the WST coefficients
fig, ax = plt.subplots()
ax.plot(smatavg, ls='-', marker='o', markersize=4, label=r'{\rr AbacusSummit}')
ax.set_xlabel('WST coefficient order')
ax.set_ylabel('WST coefficient')
plt.savefig('WST_coefficients_box.png', dpi=300, bbox_inches='tight')
plt.show()