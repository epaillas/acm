from pathlib import Path
import fitsio
from cosmoprimo.fiducial import DESI
from pyrecon.utils import sky_to_cartesian, setup_logging
from acm.estimators import WaveletScatteringTransform
import logging
import argparse
import numpy as np


def read_desi(filename, distance, zmin=0.45, zmax=0.6, is_random=False, weight_type=None):
    """Read CMASS LSS catalogues."""
    data = fitsio.read(filename)
    mask = (data['Z'] > zmin) & (data['Z'] < zmax)
    ra = data[mask]['RA']
    dec = data[mask]['DEC']
    redshift = data[mask]['Z']
    weights = data[mask]['WEIGHT']
    dist = distance(redshift)
    positions = sky_to_cartesian(dist=dist, ra=ra, dec=dec)
    return positions, weights


setup_logging()
logger = logging.getLogger('WST_DESI')
parser = argparse.ArgumentParser()
parser.add_argument('--tracer', type=str, default='LRG')
parser.add_argument('--version', type=str, default='v1.2/blinded')
parser.add_argument("--zmin", type=float, default=0.4)
parser.add_argument("--zmax", type=float, default=0.6)
parser.add_argument("--smoothing_radius", type=float, default=10)
parser.add_argument("--region", type=str, default='NGC')
parser.add_argument("--weight_type", type=str, default='')
parser.add_argument("--nthreads", type=int, default=1)
parser.add_argument("--nrandoms", type=int, default=1)
args = parser.parse_args()

# define the cosmology
cosmo = DESI()
distance = cosmo.comoving_radial_distance

# read data
data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{args.version}'
data_fn = Path(data_dir) / f'{args.tracer}_{args.region}_clustering.dat.fits'
logger.info(f'Reading {data_fn}')
data_positions, data_weights = read_desi(distance=distance, filename=data_fn,
    zmin=args.zmin, zmax=args.zmax, is_random=False, weight_type=args.weight_type)

# read randoms
randoms_positions = []
randoms_weights = []
for i in range(args.nrandoms):
    randoms_dir = Path('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.1/blinded/')
    randoms_fn = randoms_dir / f'{args.tracer}_{args.region}_{i}_clustering.ran.fits'
    logger.info(f'Reading {randoms_fn}')
    randoms_positions_i, randoms_weights_i = read_desi(distance=distance, filename=randoms_fn,
        zmin=args.zmin, zmax=args.zmax, is_random=True, weight_type=args.weight_type)
    randoms_positions.append(randoms_positions_i)
    randoms_weights.append(randoms_weights_i)
randoms_positions = np.concatenate(randoms_positions)
randoms_weights = np.concatenate(randoms_weights)

wst = WaveletScatteringTransform(data_positions=data_positions, data_weights=data_weights,
    randoms_positions=randoms_positions, randoms_weights=randoms_weights, nthreads=args.nthreads,
    cellsize=15.0)

# compute the overdensity
wst.get_delta_mesh(smoothing_radius=10)
wst.get_wst()