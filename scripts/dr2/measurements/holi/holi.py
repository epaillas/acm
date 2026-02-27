"""
Script to measure clustering statistics from the DR2 Holi mocks
(altmtl catalogs).

Some functions are borrowed from
https://github.com/adematti/jax-power/blob/main/scripts/abacus_hf.py
"""
from cosmoprimo.fiducial import TabulatedDESI
from mockfactory import Catalog, sky_to_cartesian, setup_logging
from collections.abc import Callable
import functools
from pathlib import Path
import numpy as np
import time
import os
import cloudpickle as cp

from jax import config
config.update('jax_enable_x64', True)


def get_cli_args():
    """Parse command-line arguments for Holi clustering measurements."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--statistics', nargs='+', default=['spectrum'])
    parser.add_argument("--start_phase", type=int, default=201)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument('--tracer', type=str, default='LRG')
    parser.add_argument('--region', type=str, default='NGC')
    parser.add_argument('--zrange', nargs=2, type=float, default=[0.4, 0.6])
    parser.add_argument('--n_randoms', type=int, default=19)
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/global/cfs/cdirs/desicollab/mocks/cai/LSS/DA2/mocks/holi_v1/'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default= '/pscratch/sd/a/acasella/acm/dr2/measurements/holi'
    )

    args = parser.parse_args()
    return args


def default_mpicomm(func: Callable):
    """Wrapper to provide a default MPI communicator."""
    @functools.wraps(func)
    def wrapper(*args, mpicomm=None, **kwargs):
        if mpicomm is None:
            from mpi4py import MPI
            mpicomm = MPI.COMM_WORLD
        return func(*args, mpicomm=mpicomm, **kwargs)

    return wrapper


@default_mpicomm
def _read_catalog(fn, mpicomm=None):
    """Wrapper around :meth:`Catalog.read` to read catalog(s)."""
    one_fn = fn[0] if isinstance(fn, (tuple, list)) else fn
    kw = {}
    if str(one_fn).endswith('.h5'): kw['group'] = 'LSS'
    catalog = Catalog.read(fn, mpicomm=mpicomm, **kw)
    if str(one_fn).endswith('.fits'): catalog.get(catalog.columns())  # Faster to read all columns at once
    return catalog

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


def get_clustering_rdzw(*fns, zrange=None, region=None, tracer=None, **kwargs):
    """Read one or more catalogs and return concatenated RA/DEC/Z/weight arrays.

    Weights are defined as ``WEIGHT * WEIGHT_FKP``. Optional redshift and
    region selections are applied before concatenation.
    """
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            print(fn)
            catalog = _read_catalog(fn, mpicomm=MPI.COMM_SELF)
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
    """Convert input catalogs to Cartesian positions and combined weights."""
    fiducial = TabulatedDESI()
    ra, dec, z, weights = get_clustering_rdzw(*fns, **kwargs)
    weights = np.asarray(weights, dtype='f8')
    dist = fiducial.comoving_radial_distance(z)
    positions = sky_to_cartesian(dist, ra, dec, dtype='f8')
    return positions, weights


def get_data_fn(tracer='LRG', region='NGC', phase_idx=0, base_dir='', **kwargs):
    """Build the Holi altmtl data-catalog path for a given phase."""
    mock_dir = Path(base_dir) / f'altmtl{phase_idx}' / 'loa-v1' / f'mock{phase_idx}' / 'LSScats'
    return mock_dir / f'{tracer}_{region}_clustering.dat.h5'


def get_randoms_fn(tracer='LRG', region='NGC', phase_idx=0, rand_idx=0, base_dir='', **kwargs):
    """Build the Holi altmtl random-catalog path for a given phase/index."""
    mock_dir = Path(base_dir) / f'altmtl{phase_idx}' / 'loa-v1' / f'mock{phase_idx}' / 'LSScats'
    return mock_dir / f'{tracer}_{region}_{rand_idx}_clustering.ran.h5'


def compute_spectrum(save_fn, get_data, get_randoms, ells=(0, 2, 4), los='firstpoint', **attrs):
    """Compute the power spectrum of a set of positions using the ACM package."""
    from acm.estimators.galaxy_clustering.spectrum import PowerSpectrumMultipoles
    data_positions, data_weights = get_data()
    randoms_positions, randoms_weights = get_randoms()
    ps = PowerSpectrumMultipoles(
        data_positions=data_positions,
        randoms_positions=randoms_positions,
        data_weights=data_weights,
        randoms_weights=randoms_weights,
        **attrs
    )
    ps.compute_spectrum(edges={'step': 0.001}, ells=ells, los=los, save_fn=save_fn)
    return ps


def compute_density_split(save_fn, get_data, get_randoms, smoothing_radius=10, ells=(0, 2, 4), los='z', **attrs):
    """Compute density-split statistics using the ACM package."""
    from acm.estimators.galaxy_clustering.density_split import DensitySplit

    data_positions, data_weights = get_data()
    randoms_positions, randoms_weights = get_randoms()

    ds = DensitySplit(data_positions=data_positions, data_weights=data_weights, randoms_positions=randoms_positions, randoms_weights=randoms_weights, **attrs)
    ds.set_density_contrast(smoothing_radius=smoothing_radius)
    ds.set_quantiles(query_positions=randoms_positions, nquantiles=5)

    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)

    ccf = ds.quantile_data_correlation(
        data_positions=data_positions,
        randoms_positions=randoms_positions,
        data_weights=data_weights,
        randoms_weights=randoms_weights,
        edges=edges,
        los=los,
        nthreads=4,
        gpu=True,
    )

    acf = ds.quantile_correlation(
        randoms_positions=randoms_positions,
        edges=edges,
        los=los,
        nthreads=4,
        gpu=True,
    )

    np.save(save_fn['xiqg'], ccf)
    np.save(save_fn['xiqq'], acf)


def compute_minkowski(save_fn, get_data, get_randoms, smoothing_radius=10, **attrs):
    """Compute density-split statistics using the ACM package."""
    from acm.estimators.galaxy_clustering.jaxmf import MinkowskiFunctionals
    from jaxpower import get_mesh_attrs

    data_positions, data_weights = get_data()
    randoms_positions, randoms_weights = get_randoms()

    mf = MinkowskiFunctionals(data_positions=data_positions, data_weights=data_weights, randoms_positions=randoms_positions, randoms_weights=randoms_weights, thres_mask = -5, **attrs)
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
    if not init_fn.exists():
        with open(init_fn, 'wb') as f:
            print(f'Saving WST initialization to {init_fn}')
            cp.dump(wst.S, f)
    return wst.S

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

    tracer = args.tracer
    region = args.region
    zmin, zmax = args.zrange
    nrandoms = args.n_randoms
    phases = list(range(args.start_phase, args.start_phase + args.n_phase))

    catalog_args = dict(
        tracer=tracer,
        region=region,
        zrange=(zmin, zmax),
        base_dir=args.base_dir,
    )
    wst_init = None

    for phase_idx in phases:
        data_fn = get_data_fn(phase_idx=phase_idx, **catalog_args)
        if not data_fn.exists():
            print(f'Skipping phase {phase_idx}: missing data catalog {data_fn}')
            continue
        all_randoms_fn = [
            get_randoms_fn(phase_idx=phase_idx, rand_idx=i, **catalog_args)
            for i in range(nrandoms)
        ]

        get_data = lambda: get_clustering_positions_weights(data_fn, **catalog_args)
        get_randoms = lambda: get_clustering_positions_weights(*all_randoms_fn, **catalog_args)

        if 'spectrum' in args.statistics:
            save_dir = Path(args.save_dir) / 'spectrum' / f'ph{phase_idx:03}'
            save_dir.mkdir(parents=True, exist_ok=True)
            cutsky_args = dict(cellsize=10.0, ells=(0, 2, 4))
            save_fn = Path(save_dir) / f'mesh2_poles_{tracer}_{region}_z{zmin}-{zmax}.h5'
            compute_spectrum(save_fn, get_data, get_randoms, **cutsky_args)

        if 'density_split' in args.statistics:
            save_dir = Path(args.save_dir) / 'density_split' / f'ph{phase_idx:03}'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_fn = {
                'xiqg': Path(save_dir) / f'dsc_xiqg_poles_{tracer}_{region}_z{zmin}-{zmax}.npy',
                'xiqq': Path(save_dir) / f'dsc_xiqq_poles_{tracer}_{region}_z{zmin}-{zmax}.npy',
            }
            if save_fn['xiqg'].exists() and save_fn['xiqq'].exists():
                print(f'Skipping {save_fn["xiqg"]} and {save_fn["xiqq"]}, already exists.')
                continue
            cutsky_args = dict(cellsize=5.0, boxpad=1.2, check=True)
            compute_density_split(save_fn, get_data, get_randoms, smoothing_radius=10, **cutsky_args)

        if 'minkowski' in args.statistics:
            save_dir = Path(args.save_dir) / 'minkowski' / f'ph{phase_idx:03}'
            save_dir.mkdir(parents=True, exist_ok=True)
            cutsky_args = dict(cellsize=10.0, boxpad=1.2, check=True)
            save_fn = Path(save_dir) / f'MFs_{tracer}_{region}_z{zmin}-{zmax}.npy'
            compute_minkowski(save_fn, get_data, get_randoms, smoothing_radius=10, **cutsky_args) 

        if 'wst' in args.statistics:
            save_dir = Path(args.save_dir) / 'wst' / f'ph{phase_idx:03}'
            save_dir.mkdir(parents=True, exist_ok=True)
            cutsky_args = dict(meshsize=np.repeat(360,3), J=4, L=4, sigma=0.8)
            save_fn = Path(save_dir) / f'wst_{tracer}_{region}_z{zmin}-{zmax}_jax.npy'
            wst_init = compute_wst(save_fn, get_data, get_randoms, init=wst_init, smoothing_radius=10, **cutsky_args)

        if 'mst' in args.statistics:
            save_dir = Path(args.save_dir) / 'minimum_spanning_tree' / f'ph{phase_idx:03}'
            save_dir.mkdir(parents=True, exist_ok=True)
            cutsky_args = dict(cellsize=10.0)
            save_fn = Path(save_dir) / f'mst_{tracer}_{region}_z{zmin}-{zmax}.png'
            boxsize = get_proposal_boxsize(tracer)
            compute_min_spanning_tree(save_fn, get_data, get_randoms, boxsize, smoothing_radius=10, **cutsky_args)