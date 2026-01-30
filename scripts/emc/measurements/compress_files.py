import acm.observables.emc as emc
from acm.utils.paths import lookup_registry_path
from acm import setup_logging
import argparse


parser = argparse.ArgumentParser(description='Compress EMC measurement files.')
parser.add_argument('-s', '--statistic', type=str, help='Statistic to compress.', default='GalaxyPowerSpectrumMultipoles')
parser.add_argument('--n_hod', type=int, default=500, help='Number of HOD realizations to use for compression.')
parser.add_argument('--add_covariance', action='store_true', help='Whether to add covariance to the compressed data.')

args = parser.parse_args()
statistic = args.statistic
n_hod = args.n_hod
add_covariance = args.add_covariance

setup_logging()

paths = lookup_registry_path('projects.yaml', 'emc')

paths['measurements_dir'] = '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/'
paths['data_dir'] = '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/compressed/'

observable = getattr(emc, statistic)
observable.compress_data(paths=paths, save_to=paths['data_dir'], add_covariance=add_covariance, n_hod=n_hod)