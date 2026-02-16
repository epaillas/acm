import acm.observables.emc as emc
from acm import setup_logging
import argparse
from pathlib import Path


def plot_model(observable_name, cosmo_idx=0, hod_idx=0, multipole=0):
    """
    Plot the model prediction for a given observable and cosmology/HOD index.
    """
    # paths = {
    #     'data_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/compressed/',
    #     'measurements_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/',
    #     'param_dir': None
    # }
    observable = getattr(emc, observable_name, None)(
        # paths=paths,
        select_filters={'cosmo_idx': cosmo_idx, 'hod_idx': hod_idx},
        numpy_output=True,
        squeeze_output=True,
    )
    save_fn = Path(save_dir) / f'{observable.stat_name}_model.png'
    model_params = observable.x
    observable.plot_observable(model_params=model_params, save_fn=save_fn)

def plot_emulator_residuals(observable_name):
    """
    Plot the emulator residuals for a given observable.
    """
    # paths = {
    #     'data_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/compressed/',
    #     'measurements_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/',
    #     'param_dir': None
    # }
    observable = getattr(emc, observable_name, None)(
        # paths=paths,
        select_filters={},
    )
    save_fn = Path(save_dir) / f'{observable.stat_name}_emulator_residuals.png'
    observable.plot_emulator_residuals(save_fn)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compress EMC measurement files.')
    parser.add_argument(
        '-s', '--statistics', nargs='+',
        default=[
            'ProjectedGalaxyCorrelationFunction',
            'GalaxyPowerSpectrumMultipoles',
            'ReconstructedGalaxyPowerSpectrumMultipoles',
            'GalaxyBispectrumMultipoles',
            'DensitySplitGalaxyCorrelationFunctionMultipoles',
            'MinkowskiFunctionals',
        ],
        help='List of statistics to compress.'
    )
    parser.add_argument('--save_dir', type=str, default='fig/',)
    args = parser.parse_args()
    todo_stats = args.statistics
    save_dir = args.save_dir

    setup_logging()

    for stat in todo_stats:
        plot_model(stat)
        plot_emulator_residuals(stat)