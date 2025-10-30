import acm.observables.emc as emc
from acm import setup_logging
import argparse


parser = argparse.ArgumentParser(description='Compress EMC measurement files.')
parser.add_argument('--stats', nargs='+', default=['spectrum'], help='List of statistics to compress.')
args = parser.parse_args()
todo_stats = args.stats

setup_logging()

paths = {
    'data_dir': '/pscratch/sd/e/epaillas/emc/v1.2/abacus/compressed/',
    'measurements_dir': '/pscratch/sd/e/epaillas/emc/v1.2/abacus/',
    'param_dir': '/pscratch/sd/e/epaillas/emc/cosmo+hod_params/',
}


for stat in todo_stats:
    if stat == 'spectrum':
        observable = emc.GalaxyPowerSpectrumMultipoles(paths=paths)
        observable.compress_data(save_to=paths['data_dir'], add_covariance=True, n_hod=500)

    if stat == 'bispectrum':
        observable = emc.GalaxyBispectrumMultipoles(paths=paths)
        observable.compress_data(save_to=paths['data_dir'], add_covariance=True, n_hod=30)

    if stat == 'recon_spectrum':
        observable = emc.ReconstructedGalaxyPowerSpectrumMultipoles(paths=paths)
        observable.compress_data(save_to=paths['data_dir'], add_covariance=True, n_hod=500)

    if stat == 'minkowski':
        observable = emc.MinkowskiFunctionals(paths=paths)
        observable.compress_data(save_to=paths['data_dir'], add_covariance=True, n_hod=500)

    if stat == 'projected_tpcf':
        observable = emc.ProjectedGalaxyCorrelationFunction(paths=paths)
        observable.compress_data(save_to=paths['data_dir'], add_covariance=True, n_hod=500)
