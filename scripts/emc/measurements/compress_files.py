import acm.observables.emc as emc
from acm import setup_logging
import argparse


parser = argparse.ArgumentParser(description='Compress EMC measurement files.')
parser.add_argument('--statistic', type=str, help='Statistic to compress.', default='GalaxyPowerSpectrumMultipoles')
parser.add_argument('--n_hod', type=int, default=500, help='Number of HOD realizations to use for compression.')
parser.add_argument('--add_covariance', action='store_true', help='Whether to add covariance to the compressed data.')

args = parser.parse_args()
statistic = args.statistic
n_hod = args.n_hod
add_covariance = args.add_covariance

setup_logging()

paths = {
    'data_dir': '/pscratch/sd/e/epaillas/emc/v1.2/abacus/compressed/',
    'measurements_dir': '/pscratch/sd/e/epaillas/emc/v1.2/abacus/',
    'param_dir': '/pscratch/sd/e/epaillas/emc/cosmo+hod_params/',
}

observable = getattr(emc, statistic)(paths=paths)
observable.compress_data(save_to=paths['data_dir'], add_covariance=add_covariance, n_hod=n_hod)
