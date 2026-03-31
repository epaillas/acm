from acm import setup_logging
from acm.utils.modules import get_class_from_module
from pathlib import Path
import argparse


LEGACY_TO_ALIAS = {
    'GalaxyCorrelationFunctionMultipoles': 'tpcf',
    'ProjectedGalaxyCorrelationFunction': 'projected_tpcf',
    'GalaxyPowerSpectrumMultipoles': 'spectrum',
    'GalaxyBispectrumMultipoles': 'bispectrum',
    'ReconstructedGalaxyPowerSpectrumMultipoles': 'recon_spectrum',
    'DensitySplitQuantileGalaxyCorrelationFunctionMultipoles': 'ds_xiqg',
    'DensitySplitQuantileCorrelationFunctionMultipoles': 'ds_xiqq',
    'DensitySplitGalaxyCorrelationFunctionMultipoles': 'ds_xiqg',
    'MinkowskiFunctionals': 'minkowski',
    'GalaxyOverdensityPDF': 'pdf',
    'WaveletScatteringTransform': 'wst',
}


def resolve_statistic_name(statistic: str) -> str:
    return LEGACY_TO_ALIAS.get(statistic, statistic)


def plot_training_set(observable_name):
    # paths = {
    #     'data_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/compressed/',
    #     'measurements_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/',
    #     'param_dir': None
    # }
    observable_name = resolve_statistic_name(observable_name)
    observable = get_class_from_module(args.module, observable_name)(numpy_output=True)
    save_fn = Path(args.save_dir) / f'{observable.stat_name}_training_set.png'
    observable.plot_training_set(save_fn=save_fn)

def plot_covariance_set(observable_name):
    # paths = {
    #     'data_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/compressed/',
    #     'measurements_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/',
    #     'param_dir': None
    # }
    observable_name = resolve_statistic_name(observable_name)
    observable = get_class_from_module(args.module, observable_name)(select_filters={}, numpy_output=True)
    save_fn = Path(args.save_dir) / f'{observable.stat_name}_covariance_set.png'
    observable.plot_covariance_set(save_fn=save_fn)


if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser(
        description='Generate validation figures for EMC measurements.'
    )
    parser.add_argument(
        '--module',
        type=str,
        default='acm.observables.emc',
        help='Module to load the observable classes from.'
    )
    parser.add_argument(
        '-s', '--statistics',
        nargs='+',
        default=['ds_xiqg'],
        help='List of statistics to generate validation figures for.'
    )
    parser.add_argument(
        '--save_dir',
        type=str, default='fig/',
        help='Directory to save the figures.'
    )
    parser.add_argument(
        '-f', '--figures',
        nargs='+',
        default=['training_set', 'covariance_set'],
        help='Figures to generate.'
    )

    args = parser.parse_args()

    for stat in args.statistics:
        if 'training_set' in args.figures:
            plot_training_set(stat)
        if 'covariance_set' in args.figures:
            plot_covariance_set(stat)
