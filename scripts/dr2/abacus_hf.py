"""
Script to measure power spectrum and bispectrum from the DR2
Abacus high-fidelity mocks. Some functions are borrowed from
https://github.com/adematti/jax-power/blob/main/scripts/abacus_hf.py
"""
from mockfactory import Catalog, sky_to_cartesian, setup_logging
import fitsio
from pathlib import Path
import numpy as np
import cloudpickle as cp
import time
import os


def get_cli_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument('--start_cosmo', type=int, default=0)
    parser.add_argument('--n_cosmo', type=int, default=1) 
    parser.add_argument('--todo_stats', nargs='+', default=['spectrum'])
    parser.add_argument('--tracer', type=str, default='LRG')
    parser.add_argument('--region', type=str, default='NGC')
    parser.add_argument('--zrange', nargs=2, type=float, default=[0.4, 0.6])
    parser.add_argument('--n_randoms', type=int, default=1)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/pscratch/sd/a/acasella/acm/dr2/measurements/'
    )


    args = parser.parse_args()
    return args


def select_region(ra, dec, region=None):
    # print('select', region)
    if region in [None, 'ALL', 'GCcomb']:
        return np.ones_like(ra, dtype='?')
    mask_ngc = (ra > 100 - dec)
    mask_ngc &= (ra < 280 + dec)
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.)
    if region == 'NGC':
        return mask_ngc
    if region == 'SGC':
        return ~mask_ngc
    if region == 'N':
        return mask_n
    if region == 'S':
        return mask_s
    if region == 'SNGC':
        return mask_ngc & mask_s
    if region == 'SSGC':
        return (~mask_ngc) & mask_s
    if footprint is None: load_footprint()
    north, south, des = footprint.get_imaging_surveys()
    mask_des = des[hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)]
    if region == 'DES':
        return mask_des
    if region == 'SnoDES':
        return mask_s & (~mask_des)
    if region == 'SSGCnoDES':
        return (~mask_ngc) & mask_s & (~mask_des)
    raise ValueError('unknown region {}'.format(region))

def get_proposal_boxsize(tracer):
    if 'BGS' in tracer:
        return 4000.
    if 'LRG' in tracer:
        return 7000.
    if 'LRG+ELG' in tracer:
        return 9000.
    if 'ELG' in tracer:
        return 9000.
    if 'QSO' in tracer:
        return 10000.
    raise NotImplementedError(f'tracer {tracer} is unknown')

def get_clustering_rdzw(*fns, zrange=None, region=None, tracer=None, **kwargs):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
            catalog.get(catalog.columns())  # Faster to read all columns at once
            for name in ['WEIGHT', 'WEIGHT_FKP']:
                if name not in catalog: catalog[name] = catalog.ones()
            if tracer is not None and 'Z' not in catalog:
                catalog['Z'] = catalog[f'Z_{tracer}']
            catalog = catalog[['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP']]
            if zrange is not None:
                mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] <= zrange[1])
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog['RA'], catalog['DEC'], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)

    rdzw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        weight = catalog['WEIGHT'] * catalog['WEIGHT_FKP']
        rdzw.append([catalog['RA'], catalog['DEC'], catalog['Z'], weight])
    return [np.concatenate([arrays[i] for arrays in rdzw], axis=0) for i in range(4)]

def get_clustering_positions_weights(*fns, **kwargs):
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    fiducial = TabulatedDESI()  # faster than DESI/class (which takes ~30 s for 10 random catalogs)
    ra, dec, z, weights = get_clustering_rdzw(*fns, **kwargs)
    weights = np.asarray(weights, dtype='f8')
    dist = fiducial.comoving_radial_distance(z)
    positions = sky_to_cartesian(dist, ra, dec, dtype='f8')
    return positions, weights
 
def get_data_fn(tracer='LRG', region='NGC', cosmo_idx=0, phase_idx=0, hod_idx=0, **kwargs):
    mock_dir = f'/pscratch/sd/e/epaillas/acm/dr2/hods/cutsky/v0.0/c000_ph{phase_idx:03}'
    return Path(mock_dir) / f'{tracer}_{region}_hod{hod_idx:03}.dat.fits'

# def get_randoms_fn(tracer='LRG', region='NGC', rand_idx=0, **kwargs):
#     mock_dir = '/pscratch/sd/e/epaillas/acm/dr2/hods/cutsky/v0.0'
#     return Path(mock_dir) / f'{tracer}_{region}.ran.fits'

def get_randoms_fn(tracer='LRG', region='NGC', rand_idx=0, **kwargs):
    mock_dir = '/pscratch/sd/e/epaillas/acm/dr2/hods/cutsky/v0.0'
    return Path(mock_dir) / f'{tracer}_{region}_{rand_idx}.ran.fits'

def compute_spectrum(save_fn, get_data, get_randoms, ells=(0, 2, 4), los='firstpoint', **attrs):
    import jax
    from jaxpower import (ParticleField, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum)
    t0 = time.time()
    data, randoms = get_data(), get_randoms()
    attrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    data = ParticleField(*data, attrs=attrs, exchange=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=attrs, exchange=True, backend='jax')
    fkp = FKPField(data, randoms)
    bin = BinMesh2SpectrumPoles(attrs, edges={'step': 0.001}, ells=ells)
    norm, num_shotnoise = compute_fkp2_normalization(fkp, bin=bin), compute_fkp2_shotnoise(fkp, bin=bin)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    wsum_data1 = data.sum()
    del fkp, data, randoms
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    # spectrum.attrs.update(mesh=dict(mesh.attrs), los=los, wsum_data1=wsum_data1)
    jax.block_until_ready(spectrum)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Done in {t1 - t0:.2f}', flush=True)
    spectrum.write(save_fn)

def compute_density_split(save_fn, get_data, get_randoms, smoothing_radius=10, ells=(0, 2, 4), los='z', **attrs):
    """Compute density-split statistics using the ACM package."""
    from acm.estimators.galaxy_clustering.density_split import DensitySplit
    from jaxpower import get_mesh_attrs

    data_positions, data_weights = get_data()
    randoms_positions, randoms_weights = get_randoms()
    randoms_positions_query, randoms_weights_acf = randoms_positions[:2418666], randoms_weights[:2418666]
    randoms_positions_acf, randoms_weights_query = randoms_positions[2418666:], randoms_weights[2418666:]
    print("SIZE RANDOMS: ", len(randoms_positions))
    print("SIZE RANDOMS ACF: ", len(randoms_positions_acf))
    print("SIZE RANDOMS QUERY: ", len(randoms_positions_query))

    ds = DensitySplit(data_positions=data_positions, randoms_positions=randoms_positions, **attrs)

    ds.set_density_contrast(smoothing_radius=smoothing_radius)
    ds.set_quantiles(query_positions=randoms_positions_query, nquantiles=5)

    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)

    # ccf = ds.quantile_data_correlation(
    #     data_positions=data_positions,
    #     randoms_positions=randoms_positions,
    #     data_weights=data_weights,
    #     randoms_weights=randoms_weights, estimator="landyszalay",
    #     edges=edges, los=los, nthreads=4, gpu=True)

    acf = ds.quantile_correlation(
        randoms_positions=randoms_positions_acf, estimator="landyszalay",
        edges=edges, los=los, nthreads=4, gpu=True)

    # np.save(save_fn['xiqg'], ccf)
    np.save(save_fn['xiqq'], acf)
    
    # ds.plot_quantiles(save_fn='quantiles.png')
    # ds.plot_quantile_data_correlation(save_fn='xi_qg.png')
    # ds.plot_quantile_correlation(save_fn='xi_qq.png')


def compute_minkowski(save_fn, get_data, get_randoms, smoothing_radius=10, ells=(0, 2, 4), los='z', **attrs):
    """Compute density-split statistics using the ACM package."""
    from acm.estimators.galaxy_clustering.jaxmf import MinkowskiFunctionals
    from jaxpower import get_mesh_attrs

    data_positions, data_weights = get_data()
    randoms_positions, randoms_weights = get_randoms()

    mf = MinkowskiFunctionals(data_positions=data_positions, randoms_positions=randoms_positions, thres_mask = -5, **attrs)
    mf.set_density_contrast(smoothing_radius=smoothing_radius)

    print("starting mf")
    MFs = mf.run(thresholds = np.linspace(-1,5,num=81,dtype=np.float32))

    np.save(save_fn, MFs)

def compute_wst(save_fn, get_data, get_randoms, init=None, smoothing_radius=10, **attrs):
    from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform
    import warnings
    warnings.filterwarnings("ignore")

    data_positions, data_weights = get_data()
    randoms_positions, randoms_weights = get_randoms()

    init_dir = Path('/pscratch/sd/e/epaillas/emc/v1.2/abacus/base/wst/init/')
    meshsize_str = '-'.join([f'{int(bs)}' for bs in attrs['meshsize']])
    init_fn = init_dir / f'meshsize{meshsize_str}_J{attrs["J"]}_L{attrs["L"]}_sigma{attrs["sigma"]}.npy'

    # init_dir = Path('/pscratch/sd/a/acasella/acm/dr2/measurements/wst/')
    # init_fn = init_dir / f'c{cosmo_idx:03}_cellsize{attrs['cellsize']}_J{attrs['J']}_L{attrs['L']}_sigma{attrs['sigma']}.npy'

    if init_fn.exists() and init is None:
        print(f'Loading WST initialization from {init_fn}')
        with open(init_fn, 'rb') as f:
            init = cp.load(f)

    print("starting initialization")
    wst = WaveletScatteringTransform(data_positions=data_positions, data_weights=data_weights, randoms_positions=randoms_positions,
        randoms_weights=randoms_weights, init_kymatio=init, backend='pypower', kymatio_backend='jax', **attrs) 

    print("starting density contrast")
    #Build density contrast
    wst.set_density_contrast(smoothing_radius=smoothing_radius)

    #Run WST
    smatavg = wst.run()
    np.save(save_fn, smatavg)
    #wst.plot_coefficients(save_fn=save_fn)
    if not init_fn.exists():
        with open(init_fn, 'wb') as f:
            print(f'Saving WST initialization to {init_fn}')
            cp.dump(wst.S, f)
    return wst.S

def compute_voxel_voids(save_fn, save_dir, get_data, get_randoms, smoothing_radius=10, **attrs):
    from acm.estimators.galaxy_clustering.voxel_voids import VoxelVoids
    print(type(get_data), get_data)
    data_positions, data_weights = get_data()
    randoms_positions, randoms_weights = get_randoms()
    vv = VoxelVoids(temp_dir = save_dir, data_positions=data_positions, data_weights=data_weights, randoms_positions=randoms_positions,
        randoms_weights=randoms_weights, **attrs)
    print('setting density contrast')
    vv.set_density_contrast(smoothing_radius=smoothing_radius, ran_min=0.01)
    print("Finding voids...")
    vv.find_voids()
    print("finding voxel positions")
    vv.voxel_position()
    print("finding void-data correlation")
    corr = vv.void_data_correlation(data_positions=data_positions, data_weights=data_weights, randoms_positions=randoms_positions,
        randoms_weights=randoms_weights)
    vv.plot_void_size_distribution(save_fn=save_fn['void_distr'])
    vv.plot_void_data_correlation(save_fn=save_fn['void_data_corr_plot'])
    vv.plot_slice(save_fn=save_fn['density_slice'])

    np.save(save_fn['void_data_corr'], corr)

def compute_min_spanning_tree(save_fn, get_data, get_randoms, boxsize, sigmaJ=3, smoothing_radius=10, **attrs):
    from acm.estimators.galaxy_clustering.mst import MinimumSpanningTree

    data_positions, data_weights = get_data()
    randoms_positions, randoms_weights = get_randoms()

    mst = MinimumSpanningTree(data_positions=data_positions, data_weights=data_weights, randoms_positions=randoms_positions,
        randoms_weights=randoms_weights, meshsize=128, boxsize=boxsize)
    mst.set_density_contrast(smoothing_radius=smoothing_radius)

    mst.setup(
        sigmaJ=sigmaJ,
        boxsize=boxsize,
        Nthpoint=5,      
        origin=0.0,
        split=1,      
        iterations=1,      
        quartiles=10
    )

    mstdict = mst.get_percolation_statistics(data_pos=data_positions)
    mst.plot_percolation_statistics(mstdict, fname=save_fn)

if __name__ == '__main__':
    args = get_cli_args()
    setup_logging()

    is_distributed = any(td in ['spectrum', 'recon_spectrum', 'density_split', 'minimum_spanning_tree'] for td in args.todo_stats)
    if is_distributed:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
        import jax
        jax.distributed.initialize()
    from jax import config
    config.update('jax_enable_x64', True)
    from jaxpower.mesh import create_sharding_mesh

    tracer = args.tracer
    region = args.region
    zmin, zmax = args.zrange
    nrandoms = args.n_randoms  # number of random catalogs per phase
    cosmos = list(range(args.start_cosmo, args.start_cosmo + args.n_cosmo))
    phases = list(range(args.start_phase, args.start_phase + args.n_phase))

    catalog_args = dict(tracer=tracer, region=region, zrange=(zmin, zmax))
    wst_init = None

    for cosmo_idx in cosmos:
        for phase_idx in phases:

            data_fn = get_data_fn(cosmo_idx=cosmo_idx, phase_idx=phase_idx, **catalog_args)
            all_randoms_fn = [get_randoms_fn(phase_idx=phase_idx, rand_idx=i, **catalog_args) for i in range(nrandoms)]
            print("ALL RANDOMS FN: ", all_randoms_fn)

            get_data = lambda: get_clustering_positions_weights(data_fn, **catalog_args)
            get_randoms = lambda: get_clustering_positions_weights(*all_randoms_fn, **catalog_args)

            if 'spectrum' in args.todo_stats:
                save_dir = Path(args.save_dir) / 'spectrum' / f'c{cosmo_idx:03}_ph{phase_idx:03}'
                save_dir.mkdir(parents=True, exist_ok=True)
                cutsky_args = dict(cellsize=10.0, ells=(0, 2, 4))
                with create_sharding_mesh() as sharding_mesh:
                    save_fn = Path(save_dir) / f'mesh2_poles_{tracer}_{region}_z{zmin}-{zmax}.h5'
                    compute_spectrum(save_fn, get_data, get_randoms, **cutsky_args)

            if 'density_split' in args.todo_stats:
                save_dir = Path(args.save_dir) / 'density_split' / f'c{cosmo_idx:03}_ph{phase_idx:03}'
                save_dir.mkdir(parents=True, exist_ok=True)
                # save_fn = {
                #     'xiqg': Path(save_dir) / f'dsc_xiqg_poles_{tracer}_{region}_z{zmin}-{zmax}_nat.npy',
                #     'xiqq': Path(save_dir) / f'dsc_xiqq_poles_{tracer}_{region}_z{zmin}-{zmax}_nat.npy'
                # }
                save_fn = {
                    'xiqq': Path(save_dir) / f'dsc_xiqq_poles_{tracer}_{region}_z{zmin}-{zmax}_diff_acf_randoms_switch.npy'
                }
                # if save_fn['xiqg'].exists() and save_fn['xiqq'].exists():
                #     print(f'Skipping {save_fn["xiqg"]} and {save_fn["xiqq"]}, already exists.')
                #     continue
                cutsky_args = dict(cellsize=5.0, boxpad=1.2, check=True)
                with create_sharding_mesh() as sharding_mesh:
                    compute_density_split(save_fn, get_data, get_randoms, smoothing_radius=10, **cutsky_args)

            if 'minkowski' in args.todo_stats:
                save_dir = Path(args.save_dir) / 'minkowski' / f'c{cosmo_idx:03}_ph{phase_idx:03}'
                save_dir.mkdir(parents=True, exist_ok=True)
                cutsky_args = dict(cellsize=5.0, boxpad=1.2, check=True)
                with create_sharding_mesh() as sharding_mesh:
                    save_fn = Path(save_dir) / f'MFs_{tracer}_{region}_z{zmin}-{zmax}.npy'
                    compute_minkowski(save_fn, get_data, get_randoms, smoothing_radius=10, **cutsky_args) 

            if 'wst' in args.todo_stats:
                save_dir = Path(args.save_dir) / 'wst' / f'c{cosmo_idx:03}_ph{phase_idx:03}'
                save_dir.mkdir(parents=True, exist_ok=True)
                cutsky_args = dict(meshsize=np.repeat(360,3), J=4, L=4, sigma=0.8)
                with create_sharding_mesh() as sharding_mesh:
                    save_fn = Path(save_dir) / f'wst_{tracer}_{region}_z{zmin}-{zmax}_jax.npy'
                    wst_init = compute_wst(save_fn, get_data, get_randoms, init=wst_init, smoothing_radius=10, **cutsky_args)

            if 'voxel_voids' in args.todo_stats:
                save_dir = Path(args.save_dir) / 'voxel_voids' / f'c{cosmo_idx:03}_ph{phase_idx:03}'
                save_dir.mkdir(parents=True, exist_ok=True)
                cutsky_args = dict(cellsize=10.0)
                with create_sharding_mesh() as sharding_mesh:
                    #save_fn = Path(save_dir) / f'vv_{tracer}_{region}_z{zmin}-{zmax}.h5'
                    save_fn = {
                    'void_data_corr': Path(save_dir) / f'void_data_corr_{tracer}_{region}_z{zmin}-{zmax}.npy',
                    'void_distr': Path(save_dir) / f'void_distr_{tracer}_{region}_z{zmin}-{zmax}.png',
                    'void_data_corr_plot': Path(save_dir) / f'void_data_corr_{tracer}_{region}_z{zmin}-{zmax}.png',
                    'density_slice': Path(save_dir) / f'void_density_slice_{tracer}_{region}_z{zmin}-{zmax}.png'
                    }
                    compute_voxel_voids(save_fn, save_dir, get_data, get_randoms, smoothing_radius=10, **cutsky_args)

            if 'mst' in args.todo_stats:
                save_dir = Path(args.save_dir) / 'minimum_spanning_tree' / f'c{cosmo_idx:03}_ph{phase_idx:03}'
                save_dir.mkdir(parents=True, exist_ok=True)
                cutsky_args = dict(cellsize=10.0)
                with create_sharding_mesh() as sharding_mesh:
                    save_fn = Path(save_dir) / f'mst_{tracer}_{region}_z{zmin}-{zmax}.png'
                    boxsize = get_proposal_boxsize(tracer)
                    compute_min_spanning_tree(save_fn, get_data, get_randoms, boxsize, smoothing_radius=10, **cutsky_args)