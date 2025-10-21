import os
import fitsio
from pathlib import Path
import numpy as np
import time
import glob


def get_box_args(boxsize, cellsize):
    meshsize = (boxsize / cellsize).astype(int)
    return dict(boxsize=boxsize, boxcenter=0.0, meshsize=meshsize)

def get_hod_fns(cosmo=0, phase=0, hod=0, redshift=0.8):
    """
    Get the list of HOD file names for a given cosmology,
    phase, and redshift.
    """
    base_dir = '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/'
    hod_dir = Path(base_dir) / f'c{cosmo:03}_ph{phase:03}/seed{seed_idx}/'
    hod_fn = hod_dir / f'hod{hod:03}.fits'
    return hod_fn

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


def compute_wst(positions, **attrs):
    """Compute the wavelet scattering transform using the ACM package."""
    from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform
    import warnings
    warnings.filterwarnings("ignore")

    wst = WaveletScatteringTransform(data=positions, **attrs)

    wst.set_density_contrast()

    smatavg = wst.run()
    np.save('/pscratch/sd/e/epaillas/emc/wst_from_acm_mesh.npy', smatavg)

    mesh_georgios = np.load('/pscratch/sd/g/gvalogia/density_200_CIC.npy')
    smatavg_georgios = wst.run(delta_query=mesh_georgios)
    np.save('/pscratch/sd/e/epaillas/emc/wst_from_georgios_mesh.npy', smatavg_georgios)


if __name__ == '__main__':

    from jax import config
    config.update('jax_enable_x64', True)
    from acm import setup_logging

    setup_logging()

    redshift = 0.5
    cosmo_idx = 0
    hod_idx = 8
    phase_idx = 0
    seed_idx = 0

    hod_fn = get_hod_fns(cosmo=cosmo_idx, phase=phase_idx, hod=hod_idx, redshift=redshift)

    hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
    box_args = get_box_args(boxsize, cellsize=10)
    compute_wst(hod_positions, **box_args)

