import os
import fitsio
from pathlib import Path
import numpy as np
import time
import glob


def get_cli_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--n_seed", type=int, default=1)
    parser.add_argument('--todo_stats', nargs='+', default=['spectrum'])

    args = parser.parse_args()
    return args

def get_hod_fns(cosmo=0, phase=0, redshift=0.8):
    """
    Get the list of HOD file names for a given cosmology,
    phase, and redshift.
    """
    base_dir = '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/'
    hod_dir = Path(base_dir) / f'c{cosmo:03}_ph{phase:03}/seed{seed_idx}/'
    hod_fns = glob.glob(str(Path(hod_dir) / f'hod*.fits'))
    return hod_fns

def get_hod_positions(filename, los='z'):
    """Get redshift-space positions from a HOD file."""
    hod, header = fitsio.read(filename, header=True)
    qpar, qperp = header['Q_PAR'], header['Q_PERP']
    if los == 'x':
        pos = np.c_[hod['X_RSD'], hod['Y_PERP'], hod['Z_PERP']]
        boxsize = np.array([2000/qpar, 2000/qperp, 2000/qperp])
        return pos, boxsize
    elif los == 'y':
        pos = np.c_[hod['X_PERP'], hod['Y_RSD'], hod['Z_PERP']]
        boxsize = np.array([2000/qperp, 2000/qpar, 2000/qperp])
        return pos, boxsize
    elif los == 'z':
        pos = np.c_[hod['X_PERP'], hod['Y_PERP'], hod['Z_RSD']]
        boxsize = np.array([2000/qperp, 2000/qperp, 2000/qpar])
        return pos, boxsize

def compute_density_split(output_fn, positions, smoothing_radius=10, ells=(0, 2, 4), los='z', **attrs):
    """Compute density-split statistics using the ACM package."""
    from acm.estimators.galaxy_clustering.density_split import DensitySplit

    ds = DensitySplit(data=positions, **attrs)

    ds.set_density_contrast(smoothing_radius=smoothing_radius)
    ds.set_quantiles(nquantiles=5, query_method='randoms')

    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)

    ccf = ds.quantile_data_correlation(positions, edges=edges, los=los, nthreads=4, gpu=True)
    acf = ds.quantile_correlation(edges=edges, los=los, nthreads=4, gpu=True)

    np.save(output_fn['xiqg'], ccf)
    np.save(output_fn['xiqq'], acf)
    # fig = ds.plot_quantiles()
    # fig.savefig('ds_quantiles.png', dpi=300, bbox_inches='tight')

def compute_density_split2(output_fn, positions, smoothing_radius=10, ells=(0, 2, 4), los='z', **attrs):
    """Compute density-split statistics using the ACM package."""
    from acm.estimators.galaxy_clustering.density_split import CatalogMeshDensitySplit

    ds = CatalogMeshDensitySplit(data_positions=positions, position_type='pos', **attrs)

    ds.set_density_contrast(smoothing_radius=smoothing_radius)
    ds.set_quantiles(nquantiles=5, query_method='randoms')

    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)

    ccf = ds.quantile_data_correlation(positions, edges=edges, los=los, nthreads=4, gpu=True)
    acf = ds.quantile_correlation(edges=edges, los=los, nthreads=4, gpu=True)

    np.save(output_fn['xiqg'], ccf)
    np.save(output_fn['xiqq'], acf)
    # fig = ds.plot_quantiles()
    # fig.savefig('ds_quantiles2.png', dpi=300, bbox_inches='tight')

def plot_validation():
    """Plot validation of jaxpower port against pypower for density-split xi."""

    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    data_dir = '/pscratch/sd/e/epaillas/emc/debug/abacus/raw_measurements/density_split/c000_ph000/seed0/'
    data_fn = Path(data_dir) / 'dsc_xiqq_poles_c000_hod827.npy'
    data_jaxpower = np.load(data_fn, allow_pickle=True)

    data_dir = '/pscratch/sd/e/epaillas/emc/debug/abacus/raw_measurements/density_split/c000_ph000/seed0/'
    data_fn = Path(data_dir) / 'pypower_dsc_xiqq_poles_c000_hod827.npy'
    data_pypower = np.load(data_fn, allow_pickle=True)

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    for quantile in [0, 1, 3, 4]:
        s, poles_jaxpower = data_jaxpower[quantile][::4](ells=(0, 2), return_sep=True)
        s, poles_pypower = data_pypower[quantile][::4](ells=(0, 2), return_sep=True)

        for ell in [0, 2]:
            ax[ell // 2].plot(s, s ** 2 * poles_jaxpower[ell // 2])
            ax[ell // 2].plot(s, s ** 2 * poles_pypower[ell // 2], ls='--')

    ax[0].plot(np.nan, np.nan, 'k-', label='jaxpower')
    ax[0].plot(np.nan, np.nan, 'k--', label='pypower')

    ax[0].set_ylabel(r'$s^2 \xi_0(s)\, [h^{-2}{\rm Mpc}^2]$')
    ax[1].set_ylabel(r'$s^2 \xi_2(s)\, [h^{-2}{\rm Mpc}^2]$')
    ax[0].set_xlabel(r'$s\, [h^{-1}{\rm Mpc}]$')
    ax[1].set_xlabel(r'$s\, [h^{-1}{\rm Mpc}]$')
    ax[0].legend()
    plt.tight_layout()
    plt.savefig('jaxpower_port_validation.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    args = get_cli_args()

    is_distributed = any(td in ['spectrum'] for td in args.todo_stats)
    if is_distributed:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
        import jax
        jax.distributed.initialize()
    from jax import config
    config.update('jax_enable_x64', True)
    from jaxpower.mesh import create_sharding_mesh
    from acm import setup_logging

    setup_logging()

    phases = list(range(args.start_phase, args.start_phase + args.n_phase))
    cosmos = list(range(args.start_cosmo, args.start_cosmo + args.n_cosmo))
    seeds = list(range(args.start_seed, args.start_seed + args.n_seed))

    redshift = 0.5
    init = None

    for cosmo_idx in cosmos:
        for phase_idx in phases:
            for seed_idx in seeds:
                hod_fns = get_hod_fns(cosmo=cosmo_idx, phase=phase_idx, redshift=redshift)

                for hod_fn in hod_fns[args.start_hod : args.start_hod +args.n_hod]:
                    hod_idx = hod_fn.split('.fits')[0].split('hod')[-1]

                    hod_positions, boxsize = get_hod_positions(hod_fn, los='z')

                    save_dir = '/pscratch/sd/e/epaillas/emc/debug/abacus/raw_measurements/density_split/'
                    save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    output_fn = {
                        'xiqg': Path(save_dir) / f'pypower_dsc_xiqg_poles_c{cosmo_idx:03}_hod{hod_idx:03}.npy',
                        'xiqq': Path(save_dir) / f'pypower_dsc_xiqq_poles_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                    }
                    box_args = dict(boxsize=boxsize, boxcenter=0.0, nmesh=512)
                    compute_density_split2(output_fn, hod_positions, smoothing_radius=10, **box_args)

                    save_dir = '/pscratch/sd/e/epaillas/emc/debug/abacus/raw_measurements/density_split/'
                    save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    output_fn = {
                        'xiqg': Path(save_dir) / f'dsc_xiqg_poles_c{cosmo_idx:03}_hod{hod_idx:03}.npy',
                        'xiqq': Path(save_dir) / f'dsc_xiqq_poles_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                    }
                    box_args = dict(boxsize=boxsize, boxcenter=0.0, meshsize=512)
                    compute_density_split(output_fn, hod_positions, smoothing_radius=10, **box_args)

    plot_validation()

