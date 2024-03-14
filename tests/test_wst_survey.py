from pathlib import Path
import fitsio
from cosmoprimo.fiducial import DESI
from pyrecon.utils import sky_to_cartesian, setup_logging
from acm.estimators import WaveletScatteringTransform
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_desi(filename, distance, zmin=0.45, zmax=0.6):
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
logger = logging.getLogger('test_wst_survey')
parser = argparse.ArgumentParser()
parser.add_argument('--tracer', type=str, default='LRG')
parser.add_argument('--version', type=str, default='v1.2/blinded')
parser.add_argument("--zmin", type=float, default=0.4)
parser.add_argument("--zmax", type=float, default=0.6)
parser.add_argument("--smoothing_radius", type=float, default=10)
parser.add_argument("--region", type=str, default='NGC')
parser.add_argument("--weight_type", type=str, default='')
parser.add_argument("--nthreads", type=int, default=1)
parser.add_argument("--nrandoms", type=int, default=5)
args = parser.parse_args()

# define the cosmology
cosmo = DESI()
distance = cosmo.comoving_radial_distance

# read data
data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{args.version}'
data_fn = Path(data_dir) / f'{args.tracer}_{args.region}_clustering.dat.fits'
logger.info(f'Reading {data_fn}')
data_positions, data_weights = read_desi(distance=distance, filename=data_fn,
                                         zmin=args.zmin, zmax=args.zmax)

# read randoms
randoms_positions = []
randoms_weights = []
for i in range(args.nrandoms):
    randoms_dir = Path('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.1/blinded/')
    randoms_fn = randoms_dir / f'{args.tracer}_{args.region}_{i}_clustering.ran.fits'
    logger.info(f'Reading {randoms_fn}')
    randoms_positions_i, randoms_weights_i = read_desi(distance=distance, filename=randoms_fn,
                                                       zmin=args.zmin, zmax=args.zmax)
    randoms_positions.append(randoms_positions_i)
    randoms_weights.append(randoms_weights_i)
randoms_positions = np.concatenate(randoms_positions)
randoms_weights = np.concatenate(randoms_weights)

# initialize the WST grid, using the random positions as a reference
wst = WaveletScatteringTransform(positions=randoms_positions, cellsize=15.0)

# set up the density contrast
wst.assign_data(positions=data_positions, weights=data_weights)
wst.assign_randoms(positions=randoms_positions, weights=randoms_weights)
wst.set_density_contrast()

# get the WST coefficients
smatavg = wst.run()

# plot the WST coefficients
fig, ax = plt.subplots()
ax.plot(smatavg, ls='-', marker='o', markersize=4, label=r'Blinded {\tt LRG1}')
ax.set_xlabel('WST coefficient order')
ax.set_ylabel('WST coefficient')
plt.savefig('WST_coefficients_survey.png', dpi=300, bbox_inches='tight')
plt.show()