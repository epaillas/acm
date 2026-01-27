from pathlib import Path
import glob
import fitsio
import numpy as np

from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform
from acm import setup_logging

def get_hod_fns(cosmo=0, phase=0, redshift=0.5, seed=0):
    """
    Get the list of HOD file names for a given cosmology,
    phase, and redshift.
    """
    base_dir = '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/'
    hod_dir = Path(base_dir) / f'c{cosmo:03}_ph{phase:03}/seed{seed}/'
    hod_fns = glob.glob(str(Path(hod_dir) / f'hod*.fits'))
    return sorted(hod_fns)

def get_hod_positions(filename, los='z'):
    """Get redshift-space positions from a HOD file."""
    hod, header = fitsio.read(filename, header=True)
    qpar, qperp = header['Q_PAR'], header['Q_PERP']
    if los == 'x':
        pos = np.c_[hod['X_RSD'], hod['Y_PERP'], hod['Z_PERP']]
        boxsize = np.array([2000/qpar, 2000/qperp, 2000/qperp])
    elif los == 'y':
        pos = np.c_[hod['X_PERP'], hod['Y_RSD'], hod['Z_PERP']]
        boxsize = np.array([2000/qperp, 2000/qpar, 2000/qperp])
    elif los == 'z':
        pos = np.c_[hod['X_PERP'], hod['Y_PERP'], hod['Z_RSD']]
        boxsize = np.array([2000/qperp, 2000/qperp, 2000/qpar])
    return pos, boxsize

def get_box_args(boxsize, cellsize):
    meshsize = (boxsize / cellsize).astype(int)
    return dict(boxsize=boxsize, boxcenter=0.0, meshsize=meshsize)


setup_logging()

hod_fn = get_hod_fns(cosmo=0, phase=0, redshift=0.5)[0]
positions, boxsize = get_hod_positions(hod_fn, los='z')

box_args = get_box_args(boxsize, cellsize=100)

coeffs = []
deltas = []
for backend in ['jaxpower', 'pypower']:
    box_args = get_box_args(boxsize, cellsize=100)
    wst = WaveletScatteringTransform(data_positions=positions, backend=backend, **box_args)
    wst.set_density_contrast()
    coeff = wst.run()
    coeffs.append(coeff)
    deltas.append(wst.delta_query)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 3))
for coeff, backend in zip(coeffs, ['jaxpower', 'pypower']):
    ls = '-' if backend == 'jaxpower' else '--'
    ax.plot(coeff, label=backend, ls=ls)
ax.set_xlabel('bin index')
ax.set_ylabel('WST coefficient')
ax.legend()
plt.savefig('wst_backend_comparison.png', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=(4, 3))
for delta, backend in zip(deltas, ['jaxpower', 'pypower']):
    ls = '-' if backend == 'jaxpower' else '--'
    ax.hist(delta.flatten(), bins=50, density=True, histtype='step', label=backend, ls=ls)
ax.set_xlabel('Density contrast δ')
ax.set_ylabel('PDF')
ax.legend()
plt.savefig('delta_backend_comparison.png', bbox_inches='tight', dpi=300)
