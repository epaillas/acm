import os
import fitsio
from pathlib import Path
import numpy as np
import time
import glob
from acm.utils.catalogs_safety_checks import check_catalog


def get_cli_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--start_phase", type=int, default=3000)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--n_seed", type=int, default=1)
    parser.add_argument('--todo_stats', nargs='+', default=['spectrum'])

    args = parser.parse_args()
    return args

def get_box_args(boxsize, cellsize):
    meshsize = (boxsize / cellsize).astype(int)
    return dict(boxsize=boxsize, boxcenter=0.0, meshsize=meshsize)

def get_hod_fn(phase=0, redshift=0.5):
    """
    Get the list of HOD file names for a given cosmology,
    phase, and redshift.
    """
    base_dir = f'/pscratch/sd/e/epaillas/emc/hods/z{redshift}/yuan23_prior/small/hod466'
    filename = Path(base_dir) / f'ph{phase:03}_hod466.fits'
    return filename

def get_hod_positions(filename, los='z'):
    boxsize = np.array([500.0, 500.0, 500.0])
    hod = fitsio.read(filename)
    pos = np.c_[hod['X'], hod['Y'], hod['Z']]
    hubble = 100 * fid_cosmo.efunc(redshift)
    scale_factor = 1 / (1 + redshift)
    if los == 'x':
        pos[:, 0] += hod['VX'] / (hubble * scale_factor)
    elif los == 'y':
        pos[:, 1] += hod['VY'] / (hubble * scale_factor)
    elif los == 'z':
        pos[:, 2] += hod['VZ'] / (hubble * scale_factor)

    # Periodic wrap is necessary after RSD to keep in [0,L).
    pos = np.mod(pos + boxsize/2, boxsize) - boxsize/2

    # Make sure the catalog has all galaxies are inside expected ranges
    check_catalog(pos, boxsize, check_in_float32=False, center_at_zero=True)
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
        logger.info(f'Power spectrum done in {t1 - t0:.2f} s.')
        logger.info(f'Saving to {output_fn}')
        spectrum.write(output_fn)

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
    logger.info(f'Reconstruction done in {time.time() - t0:.2f} s.')

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
    #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
    if jax.process_index() == 0:
        logger.info(f'Reconstructed power spectrum done in {time.time() - t0:.2f}')
        logger.info(f'Saving to {output_fn}')
        spectrum.write(output_fn)

    # t0 = time.time()
    # data = ParticleField(positions_rec, weights=data.weights, attrs=mattrs, exchange=True, backend='jax')
    # shifted = ParticleField(randoms_positions_rec, weights=randoms.weights, attrs=mattrs, exchange=True, backend='jax')
    # fkp = FKPField(data, shifted, attrs=mattrs)
    # bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=ells)
    # norm, num_shotnoise = compute_fkp2_normalization(fkp, bin=bin), compute_fkp2_shotnoise(fkp, bin=bin)
    # mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    # wsum_data1 = data.sum()
    # del fkp, data, shifted
    # jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    # spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    # mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    # spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
    # jax.block_until_ready(spectrum)
    # if jax.process_index() == 0:
    #     print(f'Reconstructed power spectrum done in {time.time() - t0:.2f}')
    #     spectrum.write(output_fn)

def compute_tpcf(output_fn, positions, los='z', **attrs):
    """Compute the two-point correlation function using the ACM package."""
    from pycorr import TwoPointCorrelationFunction
    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)
    xi = TwoPointCorrelationFunction(
        'smu', edges=edges, data_positions1=positions,
        engine='corrfunc', boxsize=boxsize, nthreads=128, gpu=False,
        compute_sepsavg=False, position_type='pos', los=los,
    )
    xi.save(output_fn)

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
        engine='corrfunc', boxsize=boxsize, nthreads=4, gpu=True,
        compute_sepsavg=False, position_type='pos', los=los,
    )
    xi.save(output_fn)

def compute_density_split(output_fn, positions, smoothing_radius=10, ells=(0, 2, 4), los='z', **attrs):
    """Compute density-split statistics using the ACM package."""
    from acm.estimators.galaxy_clustering.density_split import DensitySplit

    ds = DensitySplit(**attrs)

    ds.assign_data(positions=hod_positions, wrap=True, clear_previous=True)
    ds.set_density_contrast(smoothing_radius=smoothing_radius, save_wisdom=True)
    ds.set_quantiles(nquantiles=5, query_method='randoms')

    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)

    ccf = ds.quantile_data_correlation(hod_positions, edges=edges, los=los, nthreads=4, gpu=True)
    acf = ds.quantile_correlation(edges=edges, los=los, nthreads=4, gpu=True)

    np.save(output_fn['xiqg'], ccf)
    np.save(output_fn['xiqq'], acf)

def compute_wst(output_fn, positions, init=None, **attrs):
    """Compute the wavelet scattering transform using the ACM package."""
    from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform
    import warnings
    warnings.filterwarnings("ignore")

    # wst = init if init is not None else WaveletScatteringTransform(data_positions=positions, **attrs)
    wst = WaveletScatteringTransform(data_positions=positions, init_kymatio=init, **attrs)

    wst.set_density_contrast()
    smatavg = wst.run()

    print(f'Saving WST coefficients to {output_fn}')
    np.save(output_fn, smatavg)
    return wst.S  # Return the kymatio initialization for reuse

def compute_spherical_voids(output_fn, positions, radii=np.arange(22, 48, 2), cellsize=5, recon=False, los='z', **attrs):
    """Compute the spherical void size function using the ACM package."""
    from VERSUS import SphericalVoids
    from pycorr import TwoPointCorrelationFunction

    sv = SphericalVoids(data_positions=positions, cellsize=cellsize,
                        reconstruct='rsd' if recon else None,
                        recon_args={'f': 0.76, 'bias': 2., 'los': los, 'smoothing_radius': 10.},
                        **attrs)
    sv.run_voidfinding(radii, threads=32)

    # position and radius
    print(f"Saving spherical void positions and radii to {output_fn['void']}")
    np.save(output_fn['void'], np.c_[sv.void_position, sv.void_radius])

    # comoving number density of voids
    n_v = np.vstack([sorted(radii, reverse=True),
                    sv.void_count / np.prod(attrs['boxsize'])])
    print(f"Saving spherical VSF to {output_fn['vsf']}")
    np.save(output_fn['vsf'], n_v)


    # correlation functions
    redges = np.hstack([np.arange(0, 5, 1),
                        np.arange(7, 30, 3),
                        np.arange(31, 80, 5),
                        np.arange(81, 150, 8)])
    muedges = np.linspace(-1, 1, 241)
    edges = (redges, muedges)

    # void-galaxy cross correlation
    xivg = TwoPointCorrelationFunction(
        'smu', edges=edges, data_positions1=sv.void_position,
        data_positions2=positions,
        engine='corrfunc', boxsize=attrs['boxsize'], nthreads=32,
        compute_sepsavg=False, position_type='pos', los=los,
    )
    print(f"Saving spherical vg-CCF to {output_fn['xivg']}")
    xivg.save(output_fn['xivg'])

    # void auto correlation
    xivv = TwoPointCorrelationFunction(
        'smu', edges=edges, data_positions1=sv.void_position,
        engine='corrfunc', boxsize=attrs['boxsize'], nthreads=32,
        compute_sepsavg=False, position_type='pos', los=los,
    )
    print(f"Saving spherical vv-ACF to {output_fn['xivv']}")
    xivv.save(output_fn['xivv'])


def compute_dr_knn(output_fn, positions, boxsize, los='z', **attrs):
    """Compute data-random knn CDFs using the ACM package"""
    from acm.estimators.galaxy_clustering.knn import KthNearestNeighbor

    # Force boxsize to be an array of shape (3,)
    if isinstance(boxsize, float):
        boxsize = np.array([boxsize, boxsize, boxsize])
    else:
        assert isinstance(boxsize, np.ndarray), "boxsize should be either float or np.array of floats"
        if boxsize.shape==(1,) or boxsize.shape==():
            boxsize = np.repeat(boxsize, 3)
    assert boxsize.shape==(3,)

    # Convert to single precision
    positions = positions.astype(np.float32)
    boxsize   = boxsize.astype(np.float32)

    # Shift positions to [0,L]^3 box from [-L/2, L/2]^3
    positions += (boxsize/2)

    # And periodic wrap in single precision
    positions = np.mod(positions, boxsize)

    # Generate a query of randoms, 10 times the size of data
    N_randoms = 10 * len(positions)
    randoms = np.random.random((N_randoms, 3)).astype(np.float32)
    randoms[0] *= boxsize[0]
    randoms[1] *= boxsize[1]
    randoms[2] *= boxsize[2]
    randoms = np.mod(randoms, boxsize)

    # Measurement params
    ks  = [1,2,3,4,5,6,7,8,9]
    rps = np.logspace(-0.2, 1.8, 8)
    pis = np.logspace(-0.3, 1.5, 5)

    # Do the measurement
    knn  = KthNearestNeighbor()
    cdfs = knn.run_knn(
             rps,
             pis,
             xgal=positions,
             xrand=randoms,
             kneighbors=ks,
             nthread=32,
             periodic=boxsize,
             leafsize=32
           )

    # Save
    print(f'Saving DR knns in {output_fn}')
    np.save(output_fn, cdfs)

def compute_dd_knn(output_fn, positions, boxsize, los='z', **attrs):
    """Compute data-data knn CDFs using the ACM package"""
    from acm.estimators.galaxy_clustering.knn import KthNearestNeighbor

    # Force boxsize to be an array of shape (3,)
    if isinstance(boxsize, float) or isinstance(boxsize, int):
        boxsize = np.array([boxsize, boxsize, boxsize], dtype=np.float32)
    else:
        assert isinstance(boxsize, np.ndarray), "boxsize should be either float or np.array of floats"
        if boxsize.shape==(1,) or boxsize.shape==():
            boxsize = np.repeat(boxsize, 3)
    assert boxsize.shape==(3,)

    # No need in randoms, positions are used as query
    # Measurement params, k is shifted by 1 compured to dr
    ks  = [2,3,4,5,6,7,8,9,10]
    rps = np.logspace(-0.2, 1.8, 8)
    pis = np.logspace(-0.3, 1.5, 5)

    # Convert to single precision
    positions = positions.astype(np.float32)
    boxsize   = boxsize.astype(np.float32)

    # Shift positions to [0,L/2]^3 box from [-L/2, L/2]^3
    positions += (boxsize/2)

    # And periodic wrap in single precision
    positions = np.mod(positions, boxsize)

    # Do the measurement
    knn  = KthNearestNeighbor()
    cdfs = knn.run_knn(
             rps,
             pis,
             xgal=positions,
             xrand=positions,
             kneighbors=ks,
             nthread=32,
             periodic=boxsize,
             leafsize=32
           )

    # Save
    print(f'Saving DD knns in {output_fn}')
    np.save(output_fn, cdfs)


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
    from cosmoprimo.fiducial import AbacusSummit
    from acm import setup_logging
    import logging

    logger = logging.getLogger(__name__)
    setup_logging()

    phases = list(range(args.start_phase, args.start_phase + args.n_phase))

    fid_cosmo = AbacusSummit(0)
    redshift = 0.5
    wst_init = None

    for phase_idx in phases:
        hod_fn = get_hod_fn(phase=phase_idx, redshift=redshift)
        if not hod_fn.exists():
            logger.info(f'{hod_fn} not found')
            continue

        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')

        if 'spectrum' in args.todo_stats:
            save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/small/spectrum/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            output_fn = Path(save_dir) / f'mesh2_spectrum_poles_ph{phase_idx:03}.h5'
            box_args = dict(boxsize=boxsize, boxcenter=0.0, meshsize=512, los='z', ells=(0, 2, 4))
            with create_sharding_mesh() as sharding_mesh:
                compute_spectrum(output_fn, hod_positions, **box_args)

        if 'recon_spectrum' in args.todo_stats:
            save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/small/recon_spectrum/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            output_fn = Path(save_dir) / f'mesh2_recon_spectrum_poles_ph{phase_idx:03}.h5'
            box_args = dict(boxsize=boxsize, boxcenter=0.0, meshsize=512, los='z', ells=(0, 2, 4))
            with create_sharding_mesh() as sharding_mesh:
                compute_recon_spectrum(output_fn, hod_positions, **box_args)

        if 'dr_knn' in args.todo_stats:
            save_dir = '/pscratch/sd/p/pd2487/knn_measurements/small/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            output_fn = Path(save_dir) / f'dr_knn_ph{phase_idx:03}.npy'
            box_args = dict(boxsize=boxsize, los='z')
            compute_dr_knn(output_fn, hod_positions, **box_args)

        if 'dd_knn' in args.todo_stats:
            save_dir = '/pscratch/sd/p/pd2487/knn_measurements/small/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            output_fn = Path(save_dir) / f'dd_knn_ph{phase_idx:03}.npy'
            box_args = dict(boxsize=boxsize, los='z')
            compute_dd_knn(output_fn, hod_positions, **box_args)            

        # if 'tpcf' in args.todo_stats:
        #     save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/small/tpcf/'
        #     save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
        #     Path(save_dir).mkdir(parents=True, exist_ok=True)
        #     output_fn = Path(save_dir) / f'tpcf_smu_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
        #     box_args = dict(boxsize=boxsize, boxcenter=0.0)
        #     compute_tpcf(output_fn, hod_positions, **box_args)
        if 'wst' in args.todo_stats:
            save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/small/wst/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            output_fn = Path(save_dir) / f'wst_ph{phase_idx:03}.npy'
            if output_fn.exists():
                logger.info(f'Skipping {output_fn}, already exists.')
                continue
            box_args = get_box_args(boxsize, cellsize=10)
            wst_init = compute_wst(output_fn, hod_positions, init=wst_init, **box_args)

        if 'tpcf' in args.todo_stats:
            save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/small/tpcf/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            output_fn = Path(save_dir) / f'tpcf_smu_ph{phase_idx:03}.npy'
            box_args = dict(boxsize=boxsize, boxcenter=0.0)
            compute_tpcf(output_fn, hod_positions, **box_args)

        if 'spherical_voids' in args.todo_stats:
            save_dir = '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/small/spherical_voids/'
            save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            output_fn = {
                'void': Path(save_dir) / f'sv_void_ph{phase_idx:03}.npy',
                'vsf' : Path(save_dir) / f'sv_vsf_ph{phase_idx:03}.npy',
                'xivg': Path(save_dir) / f'sv_xivg_ph{phase_idx:03}.npy',
                'xivv': Path(save_dir) / f'sv_xivv_ph{phase_idx:03}.npy'
            }
            hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
            box_args = dict(boxsize=boxsize, boxcenter=0.0)
            compute_spherical_voids(output_fn, hod_positions, los='z', **box_args)

        if 'recon_spherical_voids' in args.todo_stats:
            save_dir = '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/small/recon_spherical_voids/'
            save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            output_fn = {
                'void': Path(save_dir) / f'sv_recon_void_ph{phase_idx:03}.npy',
                'vsf' : Path(save_dir) / f'sv_recon_vsf_ph{phase_idx:03}.npy',
                'xivg': Path(save_dir) / f'sv_recon_xivg_ph{phase_idx:03}.npy',
                'xivv': Path(save_dir) / f'sv_recon_xivv_ph{phase_idx:03}.npy'
            }
            hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
            box_args = dict(boxsize=boxsize, boxcenter=0.0)
            compute_spherical_voids(output_fn, hod_positions, los='z', recon=True, **box_args)

        # if 'recon_tpcf' in args.todo_stats:
        #     save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/small/recon_tpcf/'
        #     save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
        #     Path(save_dir).mkdir(parents=True, exist_ok=True)
        #     output_fn = Path(save_dir) / f'recon_tpcf_smu_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
        #     box_args = dict(boxsize=boxsize, boxcenter=0.0)
        #     compute_recon_tpcf(output_fn, hod_positions, **box_args)

        # if 'density_split' in args.todo_stats:
        #     save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/small/density_split/'
        #     save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
        #     Path(save_dir).mkdir(parents=True, exist_ok=True)
        #     output_fn = {
        #         'xiqg': Path(save_dir) / f'dsc_xiqg_poles_c{cosmo_idx:03}_hod{hod_idx:03}.npy',
        #         'xiqq': Path(save_dir) / f'dsc_xiqq_poles_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
        #     }
        #     box_args = dict(boxsize=boxsize, boxcenter=0.0, nmesh=512)
        #     compute_density_split(output_fn, hod_positions, smoothing_radius=10, **box_args)

        # if 'wst' in args.todo_stats:
        #     save_dir = '/pscratch/sd/e/epaillas/emc/v1.2/abacus/small/density_split/'
        #     save_dir += f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
        #     Path(save_dir).mkdir(parents=True, exist_ok=True)
        #     output_fn = Path(save_dir) / f'wst_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
        #     box_args = dict(boxsize=boxsize, boxcenter=0.0, nmesh=200)
        #     init = compute_wst(output_fn, hod_positions, init=init, **box_args)

