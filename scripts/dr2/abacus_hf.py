"""
Script to measure power spectrum and bispectrum from the DR2
Abacus high-fidelity mocks. Some functions are borrowed from
https://github.com/adematti/jax-power/blob/main/scripts/abacus_hf.py
"""
from mockfactory import Catalog, sky_to_cartesian, setup_logging
import fitsio
from pathlib import Path
import numpy as np
import time
import os


def get_cli_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument('--todo_stats', nargs='+', default=['spectrum'])

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

def get_data_fn(tracer='LRG', region='NGC', phase_idx=0, complete=False, **kwargs):
    if complete:
        mock_dir = '/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/SecondGenMocks/'
        mock_dir += f'AbacusSummit_v4_1/mock{phase_idx}'
        return Path(mock_dir) / f'{tracer}_complete_{region}_clustering.dat.fits'
    mock_dir = '/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/SecondGenMocks/'
    mock_dir += f'AbacusSummit_v4_1/altmtl{phase_idx}/kibo-v1/mock{phase_idx}/LSScats'
    return Path(mock_dir) / f'{tracer}_{region}_clustering.dat.fits'

def get_randoms_fn(tracer='LRG', region='NGC', phase_idx=0, rand_idx=0, complete=False, **kwargs):
    if complete:
        mock_dir = '/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/SecondGenMocks/'
        mock_dir += f'AbacusSummit_v4_1/mock{phase_idx}'
        return Path(mock_dir) / f'{tracer}_complete_{region}_{rand_idx}_clustering.ran.fits'
    mock_dir = '/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/SecondGenMocks/'
    mock_dir += f'AbacusSummit_v4_1/altmtl{phase_idx}/kibo-v1/mock{phase_idx}/LSScats'
    return Path(mock_dir) / f'{tracer}_{region}_{rand_idx}_clustering.ran.fits'

def compute_spectrum(output_fn, get_data, get_randoms, ells=(0, 2, 4), los='firstpoint', **attrs):
    import jax
    from jaxpower import (ParticleField, FKPField, compute_fkp2_spectrum_normalization, compute_fkp2_spectrum_shotnoise, BinMesh2Spectrum, get_mesh_attrs, compute_mesh2_spectrum)
    t0 = time.time()
    data, randoms = get_data(), get_randoms()
    attrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    data = ParticleField(*data, attrs=attrs, exchange=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=attrs, exchange=True, backend='jax')
    fkp = FKPField(data, randoms)
    norm, num_shotnoise = compute_fkp2_spectrum_normalization(fkp), compute_fkp2_spectrum_shotnoise(fkp)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    wsum_data1 = data.sum()
    del fkp, data, randoms
    bin = BinMesh2Spectrum(mesh.attrs, edges={'step': 0.001}, ells=ells)
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    spectrum.attrs.update(mesh=dict(mesh.attrs), los=los, wsum_data1=wsum_data1)
    jax.block_until_ready(spectrum)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Done in {t1 - t0:.2f}')
    spectrum.save(output_fn)

def compute_density_split(output_fn, get_data, get_randoms, smoothing_radius=10, ells=(0, 2, 4), los='z', **attrs):
    """Compute density-split statistics using the ACM package."""
    from acm.estimators.galaxy_clustering.density_split import DensitySplit
    from jaxpower import get_mesh_attrs

    data_positions, data_weights = get_data()
    randoms_positions, randoms_weights = get_randoms()

    ds = DensitySplit(data_positions=data_positions, randoms_positions=randoms_positions, **attrs)

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
        edges=edges, los=los, nthreads=4, gpu=True)

    acf = ds.quantile_correlation(
        randoms_positions=randoms_positions,
        edges=edges, los=los, nthreads=4, gpu=True)

    np.save(output_fn['xiqg'], ccf)
    np.save(output_fn['xiqq'], acf)
    
    ds.plot_quantiles(save_fn='quantiles.png')
    ds.plot_quantile_data_correlation(save_fn='xi_qg.png')
    ds.plot_quantile_correlation(save_fn='xi_qq.png')



if __name__ == '__main__':
    args = get_cli_args()
    setup_logging()

    is_distributed = any(td in ['spectrum', 'recon_spectrum', 'density_split'] for td in args.todo_stats)
    if is_distributed:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
        import jax
        jax.distributed.initialize()
    from jax import config
    config.update('jax_enable_x64', True)
    from jaxpower.mesh import create_sharding_mesh

    tracer = 'LRG'
    region = 'NGC'
    zmin, zmax = 0.4, 0.6
    nrandoms = 3  # number of random catalogs per phase
    phases = list(range(args.start_phase, args.start_phase + args.n_phase))

    catalog_args = dict(tracer=tracer, region=region, zrange=(zmin, zmax), complete=False)

    for phase_idx in phases:

        data_fn = get_data_fn(phase_idx=phase_idx, **catalog_args)
        all_randoms_fn = [get_randoms_fn(phase_idx=phase_idx, rand_idx=i, **catalog_args) for i in range(nrandoms)]

        get_data = lambda: get_clustering_positions_weights(data_fn, **catalog_args)
        get_randoms = lambda: get_clustering_positions_weights(*all_randoms_fn, **catalog_args)

        if 'spectrum' in args.todo_stats:
            cutsky_args = dict(cellsize=10.0, boxsize=get_proposal_boxsize(catalog_args['tracer']), ells=(0, 2, 4))
            with create_sharding_mesh() as sharding_mesh:
                output_dir = '/global/cfs/cdirs/desicollab/users/epaillas/y3-growth/'
                output_fn = Path(output_dir) / f'spectrum_QSO_NGC_z{zmin}-{zmax}_abacus_hf_ph{phase_idx:03}.npy'
                compute_spectrum(output_fn, get_data, get_randoms, **cutsky_args)

        if 'density_split' in args.todo_stats:
            save_dir = '/pscratch/sd/e/epaillas/acm/dr2/'
            save_dir += f'test/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            output_fn = {
                'xiqg': Path(save_dir) / f'dsc_xiqg_poles_{tracer}_{region}_z{zmin}-{zmax}_ph{phase_idx:03}.npy',
                'xiqq': Path(save_dir) / f'dsc_xiqq_poles_{tracer}_{region}_z{zmin}-{zmax}_ph{phase_idx:03}.npy'
            }
            if output_fn['xiqg'].exists() and output_fn['xiqq'].exists():
                print(f'Skipping {output_fn["xiqg"]} and {output_fn["xiqq"]}, already exists.')
                continue
            cutsky_args = dict(cellsize=5.0, boxpad=1.2, check=True)
            with create_sharding_mesh() as sharding_mesh:
                compute_density_split(output_fn, get_data, get_randoms, smoothing_radius=10, **cutsky_args)