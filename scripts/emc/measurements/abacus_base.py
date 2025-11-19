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

def get_box_args(boxsize, cellsize):
    meshsize = (boxsize / cellsize).astype(int)
    return dict(boxsize=boxsize, boxcenter=0.0, meshsize=meshsize)

def get_hod_fns(cosmo=0, phase=0, redshift=0.8):
    """
    Get the list of HOD file names for a given cosmology,
    phase, and redshift.
    """
    base_dir = '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/'
    hod_dir = Path(base_dir) / f'c{cosmo:03}_ph{phase:03}/seed{seed_idx}/'
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

def compute_spectrum(output_fn, positions, ells=(0, 2, 4), los='z', **attrs):
    """Compute the power spectrum of a set of positions using jaxpower."""
    from jaxpower import (MeshAttrs, ParticleField, FKPField, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum, compute_fkp2_shotnoise)
    t0 = time.time()
    mattrs = MeshAttrs(**attrs)
    data = ParticleField(positions, attrs=mattrs, exchange=True, backend='jax')
    mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mean = mesh.mean()
    mesh = mesh - mean
    bin = BinMesh2SpectrumPoles(mesh.attrs, edges={'step': 0.001}, ells=ells)
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los)
    num_shotnoise = compute_fkp2_shotnoise(data, bin=bin)
    spectrum = spectrum.clone(norm=[pole.values('norm') * mean**2 for pole in spectrum], num_shotnoise=num_shotnoise)
    jax.block_until_ready(spectrum)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Power spectrum done in {t1 - t0:.2f} s.')
        print(f'Saving to {output_fn}')
        spectrum.write(output_fn)

def compute_tpcf(output_fn, positions, los='z', **attrs):
    """Compute the two-point correlation function using the ACM package."""
    from pycorr import TwoPointCorrelationFunction

    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)

    xi = TwoPointCorrelationFunction(
        'smu', edges=edges, data_positions1=positions,
        engine='corrfunc', boxsize=attrs['boxsize'], nthreads=4, gpu=True,
        compute_sepsavg=False, position_type='pos', los=los,
    )

    xi.save(output_fn)

def compute_recon_spectrum(output_fn, positions, ells=(0, 2, 4), los='z', **attrs):
    from jaxpower import (MeshAttrs, ParticleField, FKPField, BinMesh2SpectrumPoles,
                          get_mesh_attrs, compute_mesh2_spectrum, compute_fkp2_normalization,
                          compute_box2_normalization, compute_fkp2_shotnoise, generate_uniform_particles)
    from jaxrecon.zeldovich import IterativeFFTReconstruction
    t0 = time.time()
    mattrs = MeshAttrs(**attrs)
    data = ParticleField(positions, attrs=mattrs, exchange=True, backend='jax')
    recon = IterativeFFTReconstruction(data, growth_rate=0.76, bias=2.0, los=los, smoothing_radius=10)
    positions_rec = recon.read_shifted_positions(data.positions)
    randoms = generate_uniform_particles(mattrs, 20 * len(positions), seed=42)
    randoms_positions_rec = recon.read_shifted_positions(randoms.positions)
    print(f'Reconstruction done in {time.time() - t0:.2f} s.')

    t0 = time.time()
    data = ParticleField(positions_rec, attrs=mattrs, exchange=True, backend='jax')
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=ells)
    norm = compute_box2_normalization(data, bin=bin)
    wsum_data1 = data.sum()
    data = FKPField(data, ParticleField(randoms_positions_rec, attrs=mattrs, exchange=True, backend='jax'))
    num_shotnoise = compute_fkp2_shotnoise(data, bin=bin)
    mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mesh = mesh - mesh.mean()
    del data
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
    if jax.process_index() == 0:
        print(f'Reconstructed power spectrum done in {time.time() - t0:.2f}')
        print(f'Saving to {output_fn}')
        spectrum.write(output_fn)

def compute_recon_tpcf(output_fn, positions, los='z', **attrs):
    """Compute the two-point correlation function of reconstructed positions."""
    from jaxpower import (MeshAttrs, ParticleField, generate_uniform_particles)
    from jaxrecon.zeldovich import IterativeFFTReconstruction
    from pycorr import TwoPointCorrelationFunction
    t0 = time.time()
    attrs.update(dict(meshsize=512))
    mattrs = MeshAttrs(**attrs)
    data = ParticleField(positions, attrs=mattrs, exchange=True, backend='jax')
    recon = IterativeFFTReconstruction(data, growth_rate=0.8, bias=2.0, los=los, smoothing_radius=15)
    positions_rec = recon.read_shifted_positions(data.positions)
    randoms = generate_uniform_particles(mattrs, 20 * len(positions), seed=42)
    randoms_positions_rec = recon.read_shifted_positions(randoms.positions)
    print(f'Reconstruction done in {time.time() - t0:.2f} s.')

    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)
    xi = TwoPointCorrelationFunction(
        'smu', edges=edges, data_positions1=positions_rec,
        shifted_positions1=randoms_positions_rec,
        engine='corrfunc', boxsize=attrs['boxsize'], nthreads=4, gpu=True,
        compute_sepsavg=False, position_type='pos', los=los,
    )
    xi.save(output_fn)

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

def compute_wst(output_fn, positions, init=None, **attrs):
    """Compute the wavelet scattering transform using the ACM package."""
    from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform
    import warnings
    warnings.filterwarnings("ignore")

    wst = init if init is not None else WaveletScatteringTransform(data_positions=positions, **attrs)

    wst.set_density_contrast()
    smatavg = wst.run()

    print(f'Saving WST coefficients to {output_fn}')
    np.save(output_fn, smatavg)

def compute_minkowski(output_fn, positions, **attrs):
    from acm.estimators.galaxy_clustering.jaxmf import MinkowskiFunctionals

    thresholds_fn = '/pscratch/sd/e/epaillas/emc/Thresholds_for_MFs_with_Rg5_7_10_15.npy'
    thresholds_all = np.load(thresholds_fn, allow_pickle=True).item()
    smoothing_radii = [5, 7, 10, 15]
    
    mf = MinkowskiFunctionals(data=positions, thres_mask=-5, **attrs)

    mfs3d = {}
    for smoothing_radius in smoothing_radii:
        thresholds = thresholds_all[f"Thresholds_Rg{smoothing_radius}"]
        mf.set_density_contrast(smoothing_radius=smoothing_radius)
        mf3d = mf.run(thresholds=thresholds)
        mfs3d[f'Rg{smoothing_radius}'] = mf3d
        mfs3d[f'thresholds_Rg{smoothing_radius}'] = thresholds

    print(f'Saving {output_fn}')
    np.save(output_fn, mfs3d)

def compute_spherical_voids(output_fn, positions, boxsize, radii=np.arange(24,62,2), cellsize=5, **attrs):
    """Compute the spherical void size function using the ACM package."""
    from VERSUS import SphericalVoids

    sv = SphericalVoids(data_positions=positions, cellsize=cellsize, **attrs)
    sv.run_voidfinding(radii, threads=32)

    n_v = np.vstack([sorted(radii, reverse=True),
                    sv.void_count / np.prod(boxsize)])  # comoving number density of voids

    print(f'Saving spherical VSF to {output_fn}')
    np.save(output_fn, n_v)


if __name__ == '__main__':

    args = get_cli_args()

    is_distributed = any(td in ['spectrum', 'recon_spectrum'] for td in args.todo_stats)
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

    for cosmo_idx in cosmos:
        init = None
        for phase_idx in phases:
            for seed_idx in seeds:
                hod_fns = get_hod_fns(cosmo=cosmo_idx, phase=phase_idx, redshift=redshift)
                if len(hod_fns) == 0:
                    print(f'No HOD files found for c{cosmo_idx:03}_ph{phase_idx:03}_seed{seed_idx}. Skipping.')
                    continue

                for hod_fn in hod_fns[args.start_hod : args.start_hod +args.n_hod]:
                    hod_idx = hod_fn.split('.fits')[0].split('hod')[-1]

                    if 'spectrum' in args.todo_stats:
                        save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/base/spectrum/'
                        save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        output_fn = Path(save_dir) / f'mesh2_spectrum_poles_c{cosmo_idx:03}_hod{hod_idx:03}.h5'
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = dict(boxsize=boxsize, boxcenter=0.0, meshsize=512, los='z', ells=(0, 2, 4))
                        with create_sharding_mesh() as sharding_mesh:
                            compute_spectrum(output_fn, hod_positions, **box_args)

                    if 'recon_spectrum' in args.todo_stats:
                        save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/base/recon_spectrum/'
                        save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        output_fn = Path(save_dir) / f'mesh2_recon_spectrum_poles_c{cosmo_idx:03}_hod{hod_idx:03}.h5'
                        if output_fn.exists():
                            print(f'Skipping {output_fn}, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = dict(boxsize=boxsize, boxcenter=0.0, meshsize=512, los='z', ells=(0, 2, 4))
                        with create_sharding_mesh() as sharding_mesh:
                            compute_recon_spectrum(output_fn, hod_positions, **box_args)

                    if 'tpcf' in args.todo_stats:
                        save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/base/tpcf/'
                        save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        output_fn = Path(save_dir) / f'tpcf_smu_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        if output_fn.exists():
                            print(f'Skipping {output_fn}, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = dict(boxsize=boxsize, boxcenter=0.0)
                        compute_tpcf(output_fn, hod_positions, **box_args)

                    if 'recon_tpcf' in args.todo_stats:
                        save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/base/recon_tpcf/'
                        save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        output_fn = Path(save_dir) / f'recon_tpcf_smu_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = dict(boxsize=boxsize, boxcenter=0.0)
                        compute_recon_tpcf(output_fn, hod_positions, **box_args)

                    if 'density_split' in args.todo_stats:
                        save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/base/density_split/'
                        save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        output_fn = {
                            'xiqg': Path(save_dir) / f'dsc_xiqg_poles_c{cosmo_idx:03}_hod{hod_idx:03}.npy',
                            'xiqq': Path(save_dir) / f'dsc_xiqq_poles_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        }
                        if output_fn['xiqg'].exists() and output_fn['xiqq'].exists():
                            print(f'Skipping {output_fn["xiqg"]} and {output_fn["xiqq"]}, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = get_box_args(boxsize, cellsize=3.9)
                        compute_density_split(output_fn, hod_positions, smoothing_radius=10, **box_args)

                    if 'minkowski' in args.todo_stats:
                        save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/base/minkowski/'
                        save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        output_fn = Path(save_dir) / f'minkowski_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = get_box_args(boxsize, cellsize=3.9)
                        compute_minkowski(output_fn, hod_positions, **box_args)

                    if 'wst' in args.todo_stats:
                        save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/base/wst/'
                        save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        output_fn = Path(save_dir) / f'wst_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = get_box_args(boxsize, cellsize=10)
                        init = compute_wst(output_fn, hod_positions, init=init, **box_args)

                    if 'spherical_voids' in args.todo_stats:
                        save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/base/spherical_voids/'
                        save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        output_fn = Path(save_dir) / f'sv_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        compute_spherical_voids(output_fn, hod_positions, boxsize=boxsize, boxcenter=0.)

