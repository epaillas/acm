import acm.observables.emc as emc
from acm import setup_logging
from pathlib import Path
import argparse


def plot_training_set(observable_name):
    paths = {
        'data_dir': '/pscratch/sd/e/epaillas/emc/v1.2/abacus/compressed/',
        'measurements_dir': '/pscratch/sd/e/epaillas/emc/v1.2/abacus/',
    }
    observable = getattr(emc, observable_name, None)(
        paths=paths, select_filters={}, numpy_output=True,
    )
    save_fn = Path(args.save_dir) / f'{observable.stat_name}_training_set.png'
    observable.plot_training_set(save_fn=save_fn)


if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser(
        description='Generate validation figures for EMC measurements.'
    )
    parser.add_argument(
        '-s', '--statistics',
        nargs='+',
        default=['DensitySplitGalaxyCorrelationFunctionMultipoles'],
        help='List of statistics to generate validation figures for.'
    )
    parser.add_argument(
        '--save_dir',
        type=str, default='fig/',
        help='Directory to save the figures.'
    )

    args = parser.parse_args()

    for stat in args.statistics:
        plot_training_set(stat)
