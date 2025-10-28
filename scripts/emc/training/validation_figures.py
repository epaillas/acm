import acm.observables.emc as emc
from acm import setup_logging


def plot_model(observable_name, cosmo_idx=0, hod_idx=0, multipole=0):
    """
    Plot the model prediction for a given observable and cosmology/HOD index.
    """
    paths = {
        'data_dir': '/pscratch/sd/e/epaillas/emc/v1.2/abacus/compressed/',
        'measurements_dir': '/pscratch/sd/e/epaillas/emc/v1.2/abacus/',
    }
    observable = getattr(emc, observable_name, None)(
        paths=paths, numpy_output=True,
        select_filters={'cosmo_idx': cosmo_idx, 'hod_idx': hod_idx},
    )
    save_fn = f'fig/{observable.stat_name}_model.png'
    model_params = observable.x[0]
    observable.plot_observable(model_params, save_fn)

def plot_emulator_residuals(observable_name):
    """
    Plot the emulator residuals for a given observable.
    """
    paths = {
        'data_dir': '/pscratch/sd/e/epaillas/emc/v1.2/abacus/compressed/',
        'measurements_dir': '/pscratch/sd/e/epaillas/emc/v1.2/abacus/',
    }
    observable = getattr(emc, observable_name, None)(
        paths=paths, select_filters={},
    )
    save_fn = f'fig/{observable.stat_name}_emulator_residuals.png'
    observable.plot_emulator_residuals(save_fn)


if __name__ == "__main__":

    setup_logging()

    todo_stats = [
        'ProjectedGalaxyCorrelationFunction',
        'GalaxyPowerSpectrumMultipoles',
        'GalaxyBispectrumMultipoles',
        'ReconstructedGalaxyPowerSpectrumMultipoles',
        'MinkowskiFunctionals',
    ]

    for stat in todo_stats:
        plot_model(stat)
        plot_emulator_residuals(stat)