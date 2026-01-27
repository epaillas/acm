from pathlib import Path
import glob
import fitsio
import numpy as np

from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform
from acm.estimators.galaxy_clustering.density_split import DensitySplit
from acm import setup_logging

def get_hod_fns(cosmo=1, phase=0, redshift=0.5, seed=0):
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

def test_wst():
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
    ax.set_xlabel('Density contrast')
    ax.set_ylabel('PDF')
    ax.legend()
    plt.savefig('delta_backend_comparison.png', bbox_inches='tight', dpi=300)

def test_density_split():
    box_args = get_box_args(boxsize, cellsize=10)
    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)
    poles = []
    deltas = []
    for backend in ['jaxpower', 'pypower']:
        ds = DensitySplit(data_positions=positions, backend=backend, **box_args)
        ds.set_density_contrast(smoothing_radius=10)
        ds.set_quantiles(nquantiles=5, query_method='randoms')
        deltas.append(ds.delta_query)
        ds.plot_quantiles(save_fn=f'ds_quantiles_{backend}.png')

        ccf = ds.quantile_data_correlation(positions, edges=edges, los='z', nthreads=4, gpu=True)
        ds.plot_quantile_data_correlation(save_fn=f'ds_ccf_{backend}.png')
        acf = ds.quantile_correlation(edges=edges, los='z', nthreads=4, gpu=True)
        ds.plot_quantile_correlation(save_fn=f'ds_acf_{backend}.png')

        poles_quantiles = []
        for i in range(5):
            s, multipoles = acf[i](ells=[0, 2], return_sep=True)
            poles_quantiles.append(multipoles)
        poles.append(poles_quantiles)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3))
    for i in range(5):
        for pole, backend in zip(poles, ['jaxpower', 'pypower']):
            ls = '-' if backend == 'jaxpower' else '--'
            ax.plot(s, s ** 2 * pole[i][0], label=f'Q{i+1} {backend}', ls=ls)
    ax.set_xlabel('s [Mpc/h]')
    ax.set_ylabel(r'$\xi_0(s)$')
    ax.legend()
    plt.savefig('ds_backend_comparison_xi0.png', bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(figsize=(4, 3))
    for delta, backend in zip(deltas, ['jaxpower', 'pypower']):
        ls = '-' if backend == 'jaxpower' else '--'
        ax.hist(delta.flatten(), bins=50, density=True, histtype='step', label=backend, ls=ls)
    ax.set_xlabel('Density contrast')
    ax.set_ylabel('PDF')
    ax.legend()
    plt.savefig('delta_backend_comparison.png', bbox_inches='tight', dpi=300)



setup_logging()

hod_fn = get_hod_fns(cosmo=1, phase=0, redshift=0.5)[0]
positions, boxsize = get_hod_positions(hod_fn, los='z')

test_density_split()
test_wst()

