import os
import fitsio
from pathlib import Path
import cloudpickle as cp
import numpy as np
import time
import glob
import jax
from acm.utils.catalogs_safety_checks import check_catalog
from acm.utils.default import cosmo_list
import gc


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
    parser.add_argument("-s", "--statistics", nargs='+', default=['spectrum'])
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/'
    )
    parser.add_argument(
        '--wst_config',
        type=str,
        default='j5',
        choices=['j4', 'j5', 'j4_alt'],
        help='WST configuration: j4 (J=4,L=4,q=1,sigma=0.8), j5 (J=5,L=3,q=0.8,sigma=0.4), j4_alt (J=4,L=4,q=1,sigma=1.0)'
    )
    parser.add_argument("--version", type=str, default='v1.3', help='Version string for organizing outputs')

    args = parser.parse_args()
    return args

def get_box_args(boxsize, cellsize):
    meshsize = (boxsize / cellsize).astype(int)
    return dict(boxsize=boxsize, boxcenter=0.0, meshsize=meshsize)

def get_save_dir(base_save_dir, stat_name, cosmo_idx, phase_idx, seed_idx, extra_path=None, version='v1.2'):
    """
    Construct the save directory path for a given statistic.
    
    Parameters
    ----------
    base_save_dir : str or Path
        Base directory for saving measurements.
    stat_name : str
        Name of the statistic (e.g., 'wst', 'spectrum', 'spherical_voids').
    cosmo_idx : int
        Cosmology index.
    phase_idx : int
        Phase index.
    seed_idx : int
        Seed index.
    extra_path : str, optional
        Additional path component to insert before cosmo/phase/seed (e.g., for WST configs).
    version : str, optional
        Version string (default: 'v1.2').
    
    Returns
    -------
    Path
        Complete save directory path.
    """
    stat_path = f'{version}/abacus/base/{stat_name}/'
    
    if extra_path is not None:
        save_dir = Path(base_save_dir, stat_path, extra_path, f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/')
    else:
        save_dir = Path(base_save_dir, stat_path, f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/')
    
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

def get_hod_fns(cosmo=0, phase=0, seed=0, redshift=0.8, version='v1.2'):
    """
    Get the list of HOD file names for a given cosmology,
    phase, and redshift.
    """
    base_dir = f'/pscratch/sd/n/ntbfin/emulator/hods/{version}/z0.5/yuan23_prior/'
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

    # Make sure the catalog has all galaxies are inside expected ranges. Note: pos in [-L/2,L/2)
    pos = np.mod(pos + boxsize/2, boxsize) - boxsize/2
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
        print(f'Power spectrum done in {t1 - t0:.2f} s.')
        print(f'Saving to {output_fn}')
        spectrum.write(output_fn)

def compute_spectrum_acm(output_fn, ells=(0, 2, 4), los='z', **attrs):
    """Compute the power spectrum of a set of positions using the ACM package."""
    from acm.estimators.galaxy_clustering.spectrum import PowerSpectrumMultipoles
    hod_positions, boxsize = get_hod_positions(hod_fn, los=los)
    box_args = dict(boxsize=boxsize, boxcenter=0.0, meshsize=512)
    t0 = time.time()
    ps = PowerSpectrumMultipoles(data_positions=hod_positions, **box_args)
    ps.set_density_contrast(resampler='tsc', interlacing=3, compensate=True)
    ps.compute_spectrum(edges={'step': 0.001}, ells=ells, los=los, save_fn=output_fn)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Power spectrum (ACM) done in {t1 - t0:.2f} s.')

def compute_bispectrum(output_fn, positions, basis='scoccimarro', los='z', **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, get_mesh_attrs, compute_mesh3_spectrum, MeshAttrs)
    t0 = time.time()
    mattrs = MeshAttrs(**attrs)
    data = ParticleField(positions, attrs=mattrs, exchange=True, backend='jax')
    del positions
    mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mean = mesh.mean()
    mesh = mesh - mean
    ells = [(0, 0, 0), (0, 0, 2)] if 'sugiyama' in basis else [0, 2]
    if bin is None:
        bin = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01}, basis=basis, ells=ells, buffer_size=30)
    jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'], donate_argnums=[0])
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(data, los=los, bin=bin, **kw)
    mesh = data.paint(**kw, out='real')
    del data
    spectrum = jitted_compute_mesh3_spectrum(mesh, los=los, bin=bin)
    spectrum = spectrum.clone(norm=[pole.values('norm') * mean**3 for pole in spectrum], num_shotnoise=num_shotnoise)
    # spectrum.attrs.update(mesh=dict(mesh.attrs), los=los)
    jax.block_until_ready(spectrum)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Bispectrum done in {t1 - t0:.2f} s.', flush=True)
        print(f'Saving to {output_fn}', flush=True)
        spectrum.write(output_fn)
    return bin

def compute_tpcf_smu(output_fn, positions, los='z', **attrs):
    """Compute the two-point correlation function in s-mu bins using the ACM package."""
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

def compute_tpcf_rppi(output_fn, positions, los='z', **attrs):
    """Compute the two-point correlation function in rp-pi bins using the ACM package."""
    from pycorr import TwoPointCorrelationFunction

    rp_edges = np.logspace(-1, 1.5, num = 19, endpoint = True, base = 10.0)
    pi_edges = np.linspace(-40, 40, 41)
    edges = (rp_edges, pi_edges)
    xi = TwoPointCorrelationFunction(
        'rppi', edges=edges, data_positions1=positions,
        engine='corrfunc', boxsize=attrs['boxsize'], nthreads=128, gpu=False,
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

def compute_recon_tpcf_smu(output_fn, positions, los='z', **attrs):
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

def compute_density_split(output_fn, positions, smoothing_radius=10, ells=(0, 2, 4),
    los='z', do_correlation=False, do_power=True, **attrs):
    """Compute density-split statistics using the ACM package."""
    from acm.estimators.galaxy_clustering.density_split import DensitySplit

    ds = DensitySplit(data_positions=positions, **attrs)

    ds.set_density_contrast(smoothing_radius=smoothing_radius)
    ds.set_quantiles(nquantiles=5, query_method='randoms')

    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)

    if do_correlation:
        ds.quantile_data_correlation(positions, edges=edges, los=los, nthreads=4, gpu=True, save_fn=output_fn['xiqg'])
        ds.quantile_correlation(edges=edges, los=los, nthreads=4, gpu=True, save_fn=output_fn['xiqq'])
    if do_power:
        ds.quantile_data_power(positions, edges={'step': 0.001}, ells=ells, los=los, save_fn=output_fn['pkqg'])
        ds.quantile_power(edges={'step': 0.001}, ells=ells, los=los, save_fn=output_fn['pkqq'])
            

def compute_wst(output_fn, positions, init=None, **attrs):
    """Compute the wavelet scattering transform using the ACM package."""
    from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform
    import warnings
    warnings.filterwarnings("ignore")

    init_dir = Path('/pscratch/sd/e/epaillas/emc/v1.2/abacus/base/wst/init/torch/')
    meshsize_str = '-'.join([f'{int(bs)}' for bs in attrs['meshsize']])
    init_fn = init_dir / f'meshsize{meshsize_str}_J{wst_args["J"]}_L{wst_args["L"]}_sigma{wst_args["sigma"]}.npy'
    if init_fn.exists() and init is None:
        print(f'Loading WST initialization from {init_fn}', flush=True)
        with open(init_fn, 'rb') as f:
            init = cp.load(f)

    wst = WaveletScatteringTransform(data_positions=positions, init_kymatio=init,
                                     backend='pypower', kymatio_backend='torch', **attrs)

    wst.set_density_contrast()
    smatavg = wst.run(save_fn=output_fn)

    if not init_fn.exists():
        # save kymatio initalization to a file
        with open(init_fn, 'wb') as f:
            print(f'Saving WST initialization to {init_fn}')
            cp.dump(wst.S, f)
    return wst.S

def compute_minkowski(output_fn, positions, **attrs):
    from acm.estimators.galaxy_clustering.jaxmf import MinkowskiFunctionals

    thresholds_fn = '/pscratch/sd/e/epaillas/emc/Thresholds_for_MFs_with_Rg5_7_10_15.npy'
    thresholds_all = np.load(thresholds_fn, allow_pickle=True).item()
    smoothing_radii = [5, 7, 10, 15]
    
    mf = MinkowskiFunctionals(data_positions=positions, thres_mask=-5, **attrs)

    mfs3d = {}
    for smoothing_radius in smoothing_radii:
        thresholds = thresholds_all[f"Thresholds_Rg{smoothing_radius}"]
        mf.set_density_contrast(smoothing_radius=smoothing_radius)
        mf3d = mf.run(thresholds=thresholds)
        mfs3d[f'Rg{smoothing_radius}'] = mf3d
        mfs3d[f'thresholds_Rg{smoothing_radius}'] = thresholds

    print(f'Saving {output_fn}')
    np.save(output_fn, mfs3d)


def compute_mst(output_fn, positions, boxsize, Nthpoint=5, sigmaJ=3, split=4, quartiles=10):
    """Computes the MST for the small abacus mocks."""
    from acm.estimators.galaxy_clustering.mst import MinimumSpanningTree

    halfbox = boxsize/2
    
    MST = MinimumSpanningTree(data_positions=positions, meshsize=128, boxsize=boxsize)
    MST.setup(sigmaJ, boxsize, Nthpoint, origin=-halfbox, split=split, iterations=1, quartiles=quartiles)
    mst_dict = MST.get_percolation_statistics(positions)

    print(f'Saving {output_fn}')
    np.savez(
        output_fn,
        mst1pt = mst_dict['mst1pt'],
        mst2pt = mst_dict['mst2pt'], end2pt = mst_dict['end2pt'],
        mst3pt = mst_dict['mst3pt'], end3pt = mst_dict['end3pt'],
        mst4pt = mst_dict['mst4pt'], end4pt = mst_dict['end4pt'],
        mst5pt = mst_dict['mst5pt'], end5pt = mst_dict['end5pt'],
    )


def compute_spherical_voids(output_fn, positions, radii=np.arange(20, 50, 2), cellsize=5, recon=False, los='z', **attrs):
    """Compute the spherical void size function using the ACM package."""
    from VERSUS import SphericalVoids
    from pycorr import TwoPointCorrelationFunction

    sv = SphericalVoids(data_positions=positions, cellsize=cellsize, 
                        reconstruct='rsd' if recon else None, 
                        recon_args={'f': 0.76, 'bias': 2., 'los': los, 'smoothing_radius': 10.},
                        use_wisdom=True,
                        **attrs)
    sv.run_voidfinding(radii, void_overlap=True, void_resizing=False, threads=32)

    # remove initial radius bin as this contains voids at all larger radii
    mask = sv.radius < sv.input_radii[0]
    void_rad = sv.radius[mask]
    void_pos = sv.position[mask]
    
    # position and radius
    print(f"Saving spherical void positions and radii to {output_fn['void']}")
    np.save(output_fn['void'], np.c_[void_pos, void_rad])

    # comoving number density of voids
    n_v = np.vstack([sv.input_radii[1:],
                    sv.counts[1:] / np.prod(attrs['boxsize'])])  
    print(f"Saving spherical VSF to {output_fn['vsf']}")
    np.save(output_fn['vsf'], n_v)

    # void-galaxy cross correlation
    muedges = np.linspace(-1, 1, 241)
    redges = np.hstack([np.arange(3, 80, 4), np.arange(83, 150, 7)])
    xivg = TwoPointCorrelationFunction(
        'smu', edges=(redges, muedges), 
        data_positions1=void_pos, data_positions2=positions,
        engine='corrfunc', boxsize=attrs['boxsize'], nthreads=32,
        compute_sepsavg=False, position_type='pos', los=los,
    )   
    print(f"Saving spherical vg-CCF to {output_fn['xivg']}")
    xivg.save(output_fn['xivg'])

    # void auto correlation
    redges = np.hstack([35, np.arange(40, 80, 2), np.arange(81, 150, 8)])
    xivv = TwoPointCorrelationFunction(
        'smu', edges=(redges, muedges), 
        data_positions1=void_pos,
        engine='corrfunc', boxsize=attrs['boxsize'], nthreads=32,
        compute_sepsavg=False, position_type='pos', los=los,
    )   
    print(f"Saving spherical vv-ACF to {output_fn['xivv']}")
    xivv.save(output_fn['xivv'])

'''
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

    # Perform all operations in double
    positions = positions.astype(np.float32)
    boxsize   = boxsize.astype(np.float32)

    # Shift positions to [0,L]^3 box from [-L/2, L/2]^3
    positions += (boxsize/2)

    # Periodic wrap after conversions to single precision
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
'''

def compute_dd_knn(output_fn, positions, boxsize, los='z', **attrs):
    """Compute data-data knn CDFs using the ACM package"""
    from acm.estimators.galaxy_clustering.knn import KthNearestNeighbor

    # Force boxsize to be an array of shape (3,)
    if isinstance(boxsize, float):
        boxsize = np.array([boxsize, boxsize, boxsize])
    else:
        assert isinstance(boxsize, np.ndarray), "boxsize should be either float or np.array of floats"
        if boxsize.shape==(1,) or boxsize.shape==():
            boxsize = np.repeat(boxsize, 3)
    assert boxsize.shape==(3,)

    positions = positions.astype(np.float32, copy=True)
    if los == 'x':
        # Swap X and Z (X becomes the new Z)
        positions[:, [0, 2]] = positions[:, [2, 0]]
        # Swap box dimensions too if non-cubic
        boxsize[[0, 2]] = boxsize[[2, 0]]
    elif los == 'y':
        # Swap Y and Z
        positions[:, [1, 2]] = positions[:, [2, 1]]
        boxsize[[1, 2]] = boxsize[[2, 1]]

    # No need in randoms, positions are used as query
    # Measurement params, k is shifted by 1 compured to dr
    ks = np.array([1,2,3,4,5,6,7,8,9]) + 1          # +1 for DD purposes. Reusing ks from original papers
    rps = [np.logspace(np.log10(0.5), np.log10(21.21), 9) for _ in range(len(ks))]  # Max distance scale of this analysis: 30Mpc/h
    pis = [np.logspace(np.log10(0.5), np.log10(21.21), 6) for _ in range(len(ks))]  # same bins for all k's, all elems in list are the same

    # Convert to single precision
    positions = positions.astype(np.float32)
    boxsize   = boxsize.astype(np.float32)

    # Shift positions to [0,L/2]^3 box from [-L/2, L/2]^3
    positions += (boxsize/2)

    # And periodic wrap in single precision
    positions = np.mod(positions, boxsize)

    # Swap axes of the box AND boxsize if want non-z LOS
    if los=='x':
        positions[:, [0, 2]] = positions[:, [2, 0]]
        boxsize[[0, 2]] = boxsize[[2, 0]]
    elif los == 'y':
        positions[:, [1, 2]] = positions[:, [2, 1]]
        boxsize[[1, 2]] = boxsize[[2, 1]]

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

def compute_dt_voids(output_fn, positions, **attrs):
    """Compute the Delaunay Triangulation void size function using the ACM package."""
    from acm.estimators.galaxy_clustering.pydive import DTVoids

    dtv = DTVoids()
    # dtv.run_voidfinding(threads=32)

    # n_v = np.vstack([dtv.void_radii,
    #                 dtv.void_count / np.prod(attrs['boxsize'])])  # comoving number density of voids

    # print(f'Saving DT VSF to {output_fn}')
    # np.save(output_fn, n_v)

def compute_jaxel_voids(output_fn, positions, **attrs):
    """Compute the voxel void size function using the ACM package."""
    from acm.estimators.galaxy_clustering.jaxel_voids import JaxelVoids

    jaxel = JaxelVoids(data_positions=positions, **attrs)
    jaxel.set_density_contrast(smoothing_radius=10)
    jaxel.find_voids()

    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)

    jaxel.void_data_correlation(positions, edges=edges, los='z', nthreads=4, gpu=True, save_fn=output_fn)


if __name__ == '__main__':

    args = get_cli_args()

    is_distributed = any(td in ['spectrum', 'recon_spectrum', 'bispectrum'] for td in args.statistics)
    if is_distributed:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
        import jax
        jax.distributed.initialize()
    from jax import config
    config.update('jax_enable_x64', True)
    from jaxpower.mesh import create_sharding_mesh
    from acm import setup_logging
    import logging

    logger = logging.getLogger(__name__)
    setup_logging()

    phases = list(range(args.start_phase, args.start_phase + args.n_phase))
    seeds = list(range(args.start_seed, args.start_seed + args.n_seed))

    redshift = 0.5
    jitted_compute_mesh3_spectrum = None
    wst_init = None

    for cosmo_idx in cosmo_list[args.start_cosmo : args.start_cosmo + args.n_cosmo]:
        bspec_bin = None
        for phase_idx in phases:
            for seed_idx in seeds:
                hod_fns = get_hod_fns(cosmo=cosmo_idx, phase=phase_idx, seed=seed_idx, redshift=redshift, version=args.version)
                if len(hod_fns) == 0:
                    logger.info(f'No HOD files found for c{cosmo_idx:03}_ph{phase_idx:03}_seed{seed_idx}. Skipping.')
                    continue

                for hod_fn in hod_fns[args.start_hod : args.start_hod +args.n_hod]:
                    hod_idx = hod_fn.split('.fits')[0].split('hod')[-1]

                    if 'spectrum' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'spectrum', cosmo_idx, phase_idx, seed_idx)
                        output_fn = Path(save_dir) / f'mesh2_spectrum_poles_c{cosmo_idx:03}_hod{hod_idx:03}.h5'
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = dict(boxsize=boxsize, boxcenter=0.0, meshsize=512, los='z', ells=(0, 2, 4))
                        with create_sharding_mesh() as sharding_mesh:
                            compute_spectrum(output_fn, hod_positions, **box_args)

                    if 'recon_spectrum' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'recon_spectrum', cosmo_idx, phase_idx, seed_idx)
                        output_fn = Path(save_dir) / f'mesh2_recon_spectrum_poles_c{cosmo_idx:03}_hod{hod_idx:03}.h5'
                        if output_fn.exists():
                            logger.info(f'Skipping {output_fn}, already exists.')
                            continue
                        with create_sharding_mesh() as sharding_mesh:
                            compute_recon_spectrum(output_fn, hod_positions, **box_args)

                    if 'bispectrum' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'bispectrum', cosmo_idx, phase_idx, seed_idx)
                        output_fn = Path(save_dir) / f'mesh3_spectrum_poles_c{cosmo_idx:03}_hod{hod_idx:03}.h5'
                        if output_fn.exists():
                            logger.info(f'Skipping {output_fn}, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = get_box_args(boxsize, cellsize=10)
                        with create_sharding_mesh() as sharding_mesh:
                            while True:
                                try:
                                    bspec_bin = compute_bispectrum(output_fn, hod_positions, bin=bspec_bin, **box_args)
                                    break
                                except:
                                    logger.info('Bispectrum computation failed, retrying after clearing caches...', flush=True)
                                    jax.clear_caches()
                                    gc.collect()

                    if 'tpcf' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'tpcf', cosmo_idx, phase_idx, seed_idx)
                        output_fn = Path(save_dir) / f'tpcf_smu_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        if output_fn.exists():
                            logger.info(f'Skipping {output_fn}, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = dict(boxsize=boxsize, boxcenter=0.0)
                        compute_tpcf_smu(output_fn, hod_positions, **box_args)

                    if 'tpcf_rppi' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'projected_tpcf', cosmo_idx, phase_idx, seed_idx)
                        output_fn = Path(save_dir) / f'tpcf_rppi_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        if output_fn.exists():
                            logger.info(f'Skipping {output_fn}, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = dict(boxsize=boxsize, boxcenter=0.0)
                        compute_tpcf_rppi(output_fn, hod_positions, **box_args)

                    if 'recon_tpcf_smu' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'recon_tpcf_smu', cosmo_idx, phase_idx, seed_idx)
                        output_fn = Path(save_dir) / f'recon_tpcf_smu_smu_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = dict(boxsize=boxsize, boxcenter=0.0)
                        compute_recon_tpcf_smu(output_fn, hod_positions, **box_args)

                    if 'density_split_correlation' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'density_split', cosmo_idx, phase_idx, seed_idx, version=args.version)
                        output_fn = {
                            'xiqg': Path(save_dir) / f'dsc_xiqg_poles_c{cosmo_idx:03}_hod{hod_idx:03}.npy',
                            'xiqq': Path(save_dir) / f'dsc_xiqq_poles_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        }
                        if output_fn['xiqg'].exists() and output_fn['xiqq'].exists():
                            logger.info(f'Skipping {output_fn["xiqg"]} and {output_fn["xiqq"]}, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = get_box_args(boxsize, cellsize=3.9)
                        compute_density_split(output_fn, hod_positions, smoothing_radius=10,
                            do_correlation=True, do_power=False, **box_args)

                    if 'density_split_power' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'density_split_power', cosmo_idx, phase_idx, seed_idx, version=args.version)
                        output_fn = {
                            'pkqg': Path(save_dir) / f'dsc_pkqg_poles_c{cosmo_idx:03}_hod{hod_idx:03}.h5',
                            'pkqq': Path(save_dir) / f'dsc_pkqq_poles_c{cosmo_idx:03}_hod{hod_idx:03}.h5',
                        }
                        if output_fn['pkqg'].exists() and output_fn['pkqq'].exists():
                            logger.info(f'Skipping {output_fn["pkqg"]} and {output_fn["pkqq"]}, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = get_box_args(boxsize, cellsize=3.9)
                        compute_density_split(output_fn, hod_positions, smoothing_radius=10,
                            do_correlation=False, do_power=True, **box_args)

                    if 'minkowski' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'minkowski', cosmo_idx, phase_idx, seed_idx, version=args.version)
                        output_fn = Path(save_dir) / f'minkowski_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        if output_fn.exists():
                            logger.info(f'Skipping {output_fn}, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = get_box_args(boxsize, cellsize=3.9)
                        compute_minkowski(output_fn, hod_positions, **box_args)

                    if 'wst' in args.statistics:
                        wst_configs = {
                            'j5': {'J': 5, 'L': 3, 'q': 0.8, 'sigma': 0.4, 'meshsize': 400},
                            'j4': {'J': 4, 'L': 4, 'q': 1, 'sigma': 0.8, 'meshsize': 360},
                            'j4_alt': {'J': 4, 'L': 4, 'q': 1, 'sigma': 1.0, 'meshsize': 80},
                        }
                        wst_args = wst_configs[args.wst_config].copy()
                        wst_config = f'J{wst_args["J"]}_L{wst_args["L"]}_q{wst_args["q"]}_sigma{wst_args["sigma"]}/'
                        save_dir = get_save_dir(args.save_dir, 'wst', cosmo_idx, phase_idx, seed_idx, extra_path=wst_config, version=args.version)
                        output_fn = Path(save_dir) / f'wst_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        if output_fn.exists():
                            logger.info(f'Skipping {output_fn}, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = dict(boxsize=boxsize, meshsize=np.repeat(wst_args['meshsize'], 3), boxcenter=0.0)
                        wst_args.pop('meshsize')
                        wst_init = compute_wst(output_fn, hod_positions, init=wst_init, **box_args, **wst_args)

                    if 'spherical_voids' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'spherical_voids', cosmo_idx, phase_idx, seed_idx, version=args.version)
                        label = f'c{cosmo_idx:03}_hod{hod_idx:03}' 
                        output_fn = {
                            'void': Path(save_dir) / f'sv_void_{label}.npy',
                            'vsf' : Path(save_dir) / f'sv_vsf_{label}.npy',
                            'xivg': Path(save_dir) / f'sv_xivg_{label}.npy',
                            'xivv': Path(save_dir) / f'sv_xivv_{label}.npy'
                        }
                        if output_fn['void'].exists() and output_fn['vsf'].exists() and output_fn['xivg'].exists() and output_fn['xivv'].exists():
                            logger.info(f'Skipping sv_*_{label}.npy, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = dict(boxsize=boxsize, boxcenter=0.0)
                        compute_spherical_voids(output_fn, hod_positions, los='z', **box_args)

                    if 'recon_spherical_voids' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'recon_spherical_voids', cosmo_idx, phase_idx, seed_idx, version=args.version)
                        label = f'c{cosmo_idx:03}_hod{hod_idx:03}' 
                        output_fn = {
                            'void': Path(save_dir) / f'sv_recon_void_{label}.npy',
                            'vsf' : Path(save_dir) / f'sv_recon_vsf_{label}.npy',
                            'xivg': Path(save_dir) / f'sv_recon_xivg_{label}.npy',
                            'xivv': Path(save_dir) / f'sv_recon_xivv_{label}.npy'
                        }
                        if output_fn['void'].exists() and output_fn['vsf'].exists() and output_fn['xivg'].exists() and output_fn['xivv'].exists():
                            logger.info(f'Skipping sv_recon_*_{label}.npy, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = dict(boxsize=boxsize, boxcenter=0.0)
                        compute_spherical_voids(output_fn, hod_positions, los='z', recon=True, **box_args)
                    
                    #if 'dr_knn' in args.statistics:
                    #    save_dir = get_save_dir(args.save_dir, 'dr_knn', cosmo_idx, phase_idx, seed_idx, version=args.version)
                    #    output_fn = Path(save_dir) / f'dr_knn_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                    #    hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                    #    compute_dr_knn(output_fn, hod_positions, boxsize, los='z')
                    
                    if 'dd_knn' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'dd_knn', cosmo_idx, phase_idx, seed_idx, version=args.version)
                        output_fn = Path(save_dir) / f'dd_knn_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        compute_dd_knn(output_fn, hod_positions, boxsize, los='z')

                    if 'dt_voids' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'dt_voids', cosmo_idx, phase_idx, seed_idx, version=args.version)
                        output_fn = Path(save_dir) / f'dt_voids_c{cosmo_idx:03}_hod{hod_idx:03}.npy'
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        compute_dt_voids(output_fn, hod_positions)

                    if 'jaxel_voids' in args.statistics:
                        save_dir = get_save_dir(args.save_dir, 'jaxel_voids', cosmo_idx, phase_idx, seed_idx)
                        output_fn = Path(save_dir) / f'jaxel_voids_c{cosmo_idx:03}_hod{hod_idx:03}.h5'
                        if output_fn.exists():
                            logger.info(f'Skipping {output_fn}, already exists.')
                            continue
                        hod_positions, boxsize = get_hod_positions(hod_fn, los='z')
                        box_args = get_box_args(boxsize, cellsize=3.9)
                        compute_jaxel_voids(output_fn, hod_positions, **box_args)


        if is_distributed:
            jax.clear_caches()
