from acm import setup_logging
from acm.utils.modules import get_class_from_module
import argparse
import inspect
from pathlib import Path

def get_observable_paths(observable_name, root_dir):
    observable_cls = get_class_from_module(args.module, observable_name)
    stat_name = inspect.signature(observable_cls.__init__).parameters['stat_name'].default
    root_dir = Path(root_dir)
    return {
        'data_dir': root_dir / 'emc/measurements/v1.3/abacus/compressed',
        'measurements_dir': root_dir / 'emc/measurements/v1.3/abacus',
        'param_dir': None,
        'model_dir': root_dir / 'emc/models/v1.3/best',
    }


def plot_model(observable_name, cosmo_idx=0, hod_idx=0, multipole=0):
    """
    Plot the model prediction for a given observable and cosmology/HOD index.
    """
    observable = get_class_from_module(args.module, observable_name)(
        select_filters={'cosmo_idx': cosmo_idx, 'hod_idx': hod_idx},
        numpy_output=True,
        squeeze_output=True,
        paths=get_observable_paths(observable_name, args.root_dir),
    )
    save_fn = Path(save_dir) / f'{observable.stat_name}_model.png'
    model_params = observable.x
    observable.plot_observable(model_params=model_params, save_fn=save_fn)

def plot_emulator_residuals(observable_name):
    """
    Plot the emulator residuals for a given observable.
    """
    observable = get_class_from_module(args.module, observable_name)(
        select_filters={},
        paths=get_observable_paths(observable_name, args.root_dir),
    )
    save_fn = Path(save_dir) / f'{observable.stat_name}_emulator_residuals.png'
    observable.plot_emulator_residuals(save_fn)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compress EMC measurement files.')
    parser.add_argument('--module', type=str, default='acm.observables.emc', help='Module to load the observable classes from.')
    parser.add_argument(
        '-s', '--statistics', nargs='+',
        default=[
            'projected_tpcf',
            'spectrum',
            'recon_spectrum',
            'bispectrum',
            'ds_xiqg',
            'minkowski',
        ],
        help='List of statistics to compress.'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default='/global/cfs/cdirs/desicollab/users/epaillas/acm/',
        help='Base directory for default EMC input and output paths.',
    )
    parser.add_argument('--save_dir', type=str, default='fig/',)
    args = parser.parse_args()
    todo_stats = args.statistics
    save_dir = args.save_dir

    setup_logging()

    for stat in todo_stats:
        plot_model(stat)
        plot_emulator_residuals(stat)
