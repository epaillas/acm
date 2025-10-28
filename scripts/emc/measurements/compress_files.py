import acm.observables.emc as emc
from acm import setup_logging

setup_logging()

todo_stats = [
    # 'spectrum',
    # 'bispectrum',
    # 'recon_spectrum',
    # 'minkowski',
    'projected_tpcf',
]

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
        observable.compress_data(save_to=paths['data_dir'], add_covariance=True, n_hod=9)
        # observable.compress_covariance(save_to=paths['data_dir'])

    if stat == 'recon_spectrum':
        observable = emc.ReconstructedGalaxyPowerSpectrumMultipoles(paths=paths)
        observable.compress_data(save_to=paths['data_dir'], add_covariance=True, n_hod=500)

    if stat == 'minkowski':
        observable = emc.MinkowskiFunctionals(paths=paths)
        observable.compress_data(save_to=paths['data_dir'], add_covariance=True, n_hod=500)

    if stat == 'projected_tpcf':
        observable = emc.ProjectedGalaxyCorrelationFunction(paths=paths)
        observable.compress_data(save_to=paths['data_dir'], add_covariance=True, n_hod=200)