from pyrecon import IterativeFFTReconstruction, mpi, setup_logging
from pyrecon.utils import DistanceToRedshift, sky_to_cartesian, cartesian_to_sky
import mpytools as mpy
from cosmoprimo.fiducial import DESI
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import fitsio
from pathlib import Path
import argparse


def read_data():
    if mocks == 'AbacusSummit':
        data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/altmtl{mock_idx}/mock{mock_idx}/LSScats'
        data_fn = Path(data_dir) / f'{tracer}_{region}_clustering.dat.fits'
    elif mocks == 'EZmock':
        data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/FFA/mock{mock_idx}/'
        data_fn = Path(data_dir) / f'{tracer}_ffa_{region}_clustering.dat.fits'
    print(f'Reading {data_fn}')
    return fitsio.read(data_fn)

def read_randoms(rnd_idx=0):
    if mocks == 'AbacusSummit':
        data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/altmtl{mock_idx}/mock{mock_idx}/LSScats'
        data_fn = Path(data_dir) / f'{tracer}_{region}_{rnd_idx}_clustering.ran.fits'
    elif mocks == 'EZmock':
        data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/FFA/mock{mock_idx}/'
        data_fn = Path(data_dir) / f'{tracer}_ffa_{region}_{rnd_idx}_clustering.ran.fits'
    return fitsio.read(data_fn)

def get_clustering_positions_weights(data):
    mask = (data['Z'] > zmin) & (data['Z'] < zmax)
    ra = data[mask]['RA']
    dec = data[mask]['DEC']
    dist = distance(data[mask]['Z'])
    pos = sky_to_cartesian(ra=ra, dec=dec, dist=dist)
    weights = data[mask]['WEIGHT']
    return pos, weights, mask

def bias_evolution(z, tracer='QSO'):
    """
    Bias model fitted from DR1 unblinded data (the formula from Laurent et al. 2016 (1705.04718))
    """
    if tracer == 'QSO':
        alpha = 0.237
        beta = 2.328
    elif tracer == 'LRG':
        alpha = 0.209
        beta = 2.790
    elif tracer == 'ELG_LOPnotqso':
        alpha = 0.153 
        beta = 1.541
    else:
        raise NotImplementedError(f'{tracer} not implemented.')
    return alpha * ((1+z)**2 - 6.565) + beta

def interpolate_f_bias(cosmo, tracer, zdependent=False):
    P0 = {'LRG': 8.9e3, 'QSO': 5.0e3}[tracer]
    if zdependent:
        z = np.linspace(0.0, 5.0, 10000)
        growth_rate = cosmo.growth_rate(z)
        bias = bias_evolution(z, tracer)
        distance = cosmo.comoving_radial_distance
        f_at_dist = InterpolatedUnivariateSpline(distance(z), growth_rate, k=3)
        bias_at_dist = InterpolatedUnivariateSpline(distance(z), bias, k=3)
        f_at_dist, bias_at_dist
    else:
        f_at_dist = {'LRG': 0.834, 'QSO': 0.928}[tracer]
        bias_at_dist = {'LRG': 2.0, 'QSO': 2.1}[tracer]
    return f_at_dist, bias_at_dist, P0

def interpolate_nbar():
    if mocks == 'AbacusSummit':
        nz_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/altmtl{mock_idx}/mock{mock_idx}/LSScats'
        nz_fn = Path(nz_dir) / f'{tracer}_nz.txt'
    elif mocks == 'EZmock':
        nz_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/FFA/mock{mock_idx}/'
        nz_fn = Path(nz_dir) / f'{tracer}_ffa_nz.txt'
    data = np.genfromtxt(nz_fn)
    zmid = data[:, 0]
    nz = data[:, 3]
    n_at_dist = InterpolatedUnivariateSpline(distance(zmid), nz, k=3, ext=1)
    return n_at_dist

def run_recon(recon_weights=False, fmesh=False, bmesh=False):
    f, bias, P0 = interpolate_f_bias(cosmo, tracer, zdependent=False)
    f_at_dist, bias_at_dist, P0 = interpolate_f_bias(cosmo, tracer, zdependent=True)
    nbar_at_dist =  interpolate_nbar()
    f = f_at_dist if fmesh else f
    bias = bias_at_dist if bmesh else bias
    if mpicomm.rank == 0:
        data = read_data()
        data_positions, data_weights, data_mask = get_clustering_positions_weights(data)
        data = data[data_mask]
    else:
        data_positions, data_weights = None, None
    recon = IterativeFFTReconstruction(f=f, bias=bias, positions=data_positions,
                                       los='local', cellsize=5.0, boxpad=1.2,
                                       position_type='pos', dtype='f8', mpicomm=mpicomm,
                                       mpiroot=0)
    recon.assign_data(data_positions, data_weights)
    for i in range(18):
        if mpicomm.rank == 0:
            randoms = read_randoms(i)
            random_positions, random_weights, randoms_mask = get_clustering_positions_weights(randoms)
        else:
            random_positions, random_weights = None, None
        recon.assign_randoms(random_positions, random_weights)
    recon.set_density_contrast(smoothing_radius=smoothing_radius,
                               kw_weights={'P0': P0, 'nbar': nbar_at_dist} if recon_weights else None)
    recon.run()
    if mpicomm.rank == 0:
        print('Reading shifted positions...')
    data_positions_recon = recon.read_shifted_positions(data_positions)
    if mpicomm.rank == 0:
        dist, ra, dec = cartesian_to_sky(data_positions_recon)
        data['RA'], data['DEC'], data['Z'] = ra, dec, d2r(dist)
        if recon_weights and fmesh and bmesh:
            output_dir = f'/pscratch/sd/e/epaillas/recon_weights/v2/f_bias_weights/mocks/{mocks}/{mock_idx}'
        elif fmesh and bmesh:
            output_dir = f'/pscratch/sd/e/epaillas/recon_weights/v2/f_bias/mocks/{mocks}/{mock_idx}'
        elif recon_weights:
            output_dir = f'/pscratch/sd/e/epaillas/recon_weights/v2/weights/mocks/{mocks}/{mock_idx}'
        elif fmesh:
            output_dir = f'/pscratch/sd/e/epaillas/recon_weights/v2/f/mocks/{mocks}/{mock_idx}'
        elif bmesh:
            output_dir = f'/pscratch/sd/e/epaillas/recon_weights/v2/bias/mocks/{mocks}/{mock_idx}'
        else:
            output_dir = f'/pscratch/sd/e/epaillas/recon_weights/v2/vanilla/mocks/{mocks}/{mock_idx}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if mocks == 'EZmock':
            output_fn = Path(output_dir) / f'{tracer}_ffa_{region}_clustering.IFTrecsym.dat.fits'
        else:
            output_fn = Path(output_dir) / f'{tracer}_{region}_clustering.IFTrecsym.dat.fits'
        fitsio.write(output_fn, data, clobber=True)
    for i in range(4):
        if mpicomm.rank == 0:
            randoms = read_randoms(i)
            randoms_positions, randoms_weights, randoms_mask = get_clustering_positions_weights(randoms)
            randoms = randoms[randoms_mask]
        else:
            randoms_positions, randoms_weights = None, None
        randoms_positions_recon = recon.read_shifted_positions(randoms_positions)
        if mpicomm.rank == 0:
            dist, ra, dec = cartesian_to_sky(randoms_positions_recon)
            randoms['RA'], randoms['DEC'], randoms['Z'] = ra, dec, d2r(dist)
            if mocks == 'EZmock':
                output_fn = Path(output_dir) / f'{tracer}_ffa_{region}_{i}_clustering.IFTrecsym.ran.fits'
            else:
                output_fn = Path(output_dir) / f'{tracer}_{region}_{i}_clustering.IFTrecsym.ran.fits'
            fitsio.write(output_fn, randoms, clobber=True)


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--start_idx', type=int, default=1)
    args.add_argument('--end_idx', type=int, default=100)
    args.add_argument('--mocks', type=str, default='EZmock')
    args.add_argument('--tracer', type=str, default='QSO')
    args.add_argument('--regions', type=str, nargs='+', default=['NGC', 'SGC'])
    args.add_argument('--zmin', type=float, default=0.8)
    args.add_argument('--zmax', type=float, default=2.1)
    args.add_argument('--recon_weights', action='store_true', default=False)
    args.add_argument('--fmesh', action='store_true', default=False)
    args.add_argument('--bmesh', action='store_true', default=False)

    args = args.parse_args()

    setup_logging()
    mpicomm = mpy.COMM_WORLD

    mocks = args.mocks
    tracer = args.tracer
    regions = args.regions
    zmin, zmax = args.zmin, args.zmax
    smoothing_radius = 30 if tracer == 'QSO' else 15

    cosmo = DESI()
    distance = cosmo.comoving_radial_distance
    d2r = DistanceToRedshift(distance)

    mock_idxs = list(range(args.start_idx, args.end_idx))

    for mock_idx in mock_idxs:
        if mock_idx in [11, 25]: continue
        for region in regions:
            run_recon(recon_weights=args.recon_weights, fmesh=args.fmesh, bmesh=args.bmesh)