from sunbird.inference.pocomc import PocoMCSampler
from sunbird.inference import priors as sunbird_priors
from sunbird.cosmology.model_params import get_model_params
from sunbird.inference.samples import Chain

import acm.observables.emc as emc
from acm.observables import CombinedObservable
from acm.utils.covariance import get_covariance_correction
from acm import setup_logging

from cosmoprimo import fiducial

from pathlib import Path
import numpy as np
import argparse
import logging


class_names = {
    'wp': 'ProjectedGalaxyCorrelationFunction',
    'pk': 'GalaxyPowerSpectrumMultipoles',
    'bk': 'GalaxyBispectrumMultipoles',
    'recon_pk': 'ReconstructedGalaxyPowerSpectrumMultipoles',
    'minkowski': 'MinkowskiFunctionals',
    'ds_xiqg': 'DensitySplitQuantileGalaxyCorrelationFunctionMultipoles',
    'ds_xiqq': 'DensitySplitQuantileCorrelationFunctionMultipoles',
    'pdf': 'GalaxyOverdensityPDF',
}


def get_priors(cosmo=True, hod=True):
    """
    Return a dictionary of prior distributions, hard limits (ranges),
    and labels for cosmological and HOD parameters in a format
    that is readable by PocoMCSampler.
    """
    stats_module = 'scipy.stats'
    priors, ranges, labels = {}, {}, {}
    if cosmo:
        priors.update(sunbird_priors.AbacusSummit(stats_module).priors)
        ranges.update(sunbird_priors.AbacusSummit(stats_module).ranges)
        labels.update(sunbird_priors.AbacusSummit(stats_module).labels)
    if hod:
        priors.update(sunbird_priors.Yuan23(stats_module).priors)
        ranges.update(sunbird_priors.Yuan23(stats_module).ranges)
        labels.update(sunbird_priors.Yuan23(stats_module).labels)
    return priors, ranges, labels


def get_fixed_params(cosmo_model, hod_model):
    """
    Return a list of fixed parameter names based on the cosmological and HOD models.
    This function checks which parameters are free in the specified models.
    """
    free = []
    # cosmology
    if 'base' in cosmo_model:
        free += ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s']
    if 'w0' in cosmo_model:
        free += ['w0_fld']
    if 'wa' in cosmo_model:
        free += ['wa_fld']
    if 'Nur' in cosmo_model:
        free += ['N_ur']
    if 'nrun' in cosmo_model:
        free += ['nrun']
    if 'fixed-ns' in cosmo_model:
        free.remove('n_s')
    # HOD
    if 'base' in hod_model:
        free += ['logM_cut', 'logM_1', 'sigma', 'alpha', 'kappa']
    if 'AB' in hod_model:
        free += ['B_cen', 'B_sat']
    if "CB" in hod_model:
        free += ["A_cen", "A_sat"]
    if 'VB' in hod_model:
        free += ['alpha_c', 'alpha_s']
    if '_s' in hod_model or '-s' in hod_model:
        free += ['s']
    fixed = [par for par in priors.keys() if par not in free]
    return fixed

def get_filters(observable_name):
    """
    Get the select and slice coordinates for the observable.
    This function returns dictionaries that specify which coordinates to select
    and which to slice for the given observable.
    """
    select_filters = {'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx}
    slice_filters = {}
    """Get the select and slice coordinates for the observable."""
    if observable_name == 'GalaxyCorrelationFunctionMultipoles':
        select_filters.update({'multipoles': [0, 2]})
        slice_filters.update({'s': [0.0, 150]})
    elif observable_name == 'GalaxyPowerSpectrumMultipoles':
        select_filters.update({'multipoles': [0, 2, 4]})
    elif observable_name == 'GalaxyBispectrumMultipoles':
        select_filters.update({'multipoles': [0, 2]})
        slice_filters.update({'k': [0.0, 0.7]})
    elif observable_name == 'ReconstructedGalaxyPowerSpectrumMultipoles':
        select_filters.update({'multipoles': [0, 2, 4]})
        slice_filters.update({'k': [0.0, 0.7]})
    elif observable_name == 'DensitySplitPowerSpectrumMultipoles':
        select_filters.update({'statistics': ['quantile_data_power']})
    return select_filters, slice_filters

def get_observable(stat_names):
    """Get the observable class from a list of stat_name."""
    paths = {
        'data_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/compressed/',
        'measurements_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/',
        'model_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/models/v1.2/best/',
        'param_dir': None
    }
    select_filters = {'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx}
    observables = []
    for stat_name in stat_names:
        observable_name = class_names[stat_name]
        select_indices = selected_bins[stat_name]
        obs = getattr(emc, observable_name)(
            paths=paths, numpy_output=True,
            squeeze_output=True,
            select_filters=select_filters,
            select_indices=select_indices,
        )
        observables.append(obs)
    return obs if len(observables) == 1 else CombinedObservable(observables)

def get_data_model_cov(observable):
    """
    This function loads the data, covariance matrix, and model prediction
    from the observable instance.
    """
    # load the data
    data_x = observable.x
    data_x_names = observable.x_names
    data_y = observable.y
    logger.info(f'Loaded data_x with shape: {data_x.shape}')
    logger.info(f'Loaded data_y with shape {data_y.shape}')

    # load the covariance matrix, including emulator error and Percival correction
    cov = observable.get_covariance_matrix(volume_factor=64)
    logger.info(f'Loaded covariance matrix with shape: {cov.shape}')
    if args.add_cov_emu:
        cov += observable.get_emulator_covariance_matrix(
            method=args.cov_emu_method,
            diag=args.cov_emu_diag,
        )

    cov *= get_covariance_correction(
        n_s=len(observable.covariance_y),
        n_d=len(cov),
        n_theta=len(data_x_names) - len(fixed_param_names),
        method=args.cov_correction,
    )

    # load the model
    model = observable.get_model_prediction

    return data_x, data_x_names, data_y, cov, model

def fit_abacus(observable):
    """
    Fit the AbacusSummit data using the PocoMCSampler.
    This function loads the data, covariance matrix, and model,
    prepares the precision matrix, and samples the posterior distribution.
    It also saves the results, including plots and chain data.
    """
    statistics = observable.stat_name

    data_x, data_x_names, data_y, cov, model = get_data_model_cov(observable)

    precision_matrix = np.linalg.inv(cov)

    # a dictionary containing the values of the parameters we want to fix
    fixed_params = {key: data_x[data_x_names.index(key)] for key in fixed_param_names}

    # a 'markers' dictionary containing the true values of the parameters
    markers = {key: data_x[data_x_names.index(key)] for key in data_x_names if key not in fixed_params}
    cosmo = fiducial.AbacusSummit(args.cosmo_idx)
    markers.update({'Omega_m': cosmo['Omega_m'], 'h': cosmo['h']})

    # sample the posterior
    sampler = PocoMCSampler(
        observation=data_y,
        precision_matrix=precision_matrix,
        theory_model=model,
        fixed_parameters=fixed_params,
        priors=priors,
        ranges=ranges,
        labels=labels,
        ellipsoid=True,
        markers=markers,
    )
    sampler(vectorize=True, n_total=4096)

    return sampler

def save_and_plot(sampler, observable):
    """
    Save and plot the results of the sampler.
    This function saves the chain data, plots the triangle and trace plots,
    and plots the best-fit model against the data.
    """
    statistics = "+".join(observable.stat_name)
    if args.identifier is not None: statistics += f'_{args.identifier}'
    save_dir = Path(args.save_dir) / f'c{args.cosmo_idx:03}_hod{args.hod_idx:03}/cosmo-{cosmo_model}_hod-{hod_model}/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    """Save the chain data and plots to the specified directory."""
    sampler.save_chain(save_fn=save_dir / f'chain_greedy.npy', metadata={'markers': sampler.markers, 'zeff': 0.5})
    sampler.save_table(save_fn=save_dir / f'chain_greedy_stats.txt')
    chain = Chain.load(save_dir / f'chain_greedy.npy')
    chain.plot_triangle(save_fn=save_dir / f'chain_greedy_triangle.pdf', thin=128,
                        markers=sampler.markers, title_limit=1)
    chain.plot_trace(save_fn=save_dir / f'chain_greedy_trace.pdf', thin=128)
    observable.plot_observable(model_params=chain.bestfit, save_fn=save_dir / f'chain_greedy_bestfit.pdf')


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy_fn", type=Path, default=Path("/global/u1/e/epaillas/code/acm/scripts/emc/fisher/selected_bins.npy"))
    parser.add_argument("--cosmo_idx", type=int, default=0)
    parser.add_argument("--hod_idx", type=int, default=0)
    parser.add_argument('--add_cov_emu', action='store_true', help='Whether to add emulator covariance or not.')
    parser.add_argument('--cosmo_model', type=str, default='base', help='Cosmological model to use.')
    parser.add_argument('--hod_model', type=str, default='base-VB-AB-CB-s', help='HOD model to use.')
    parser.add_argument('--identifier', type=str, default=None, help='Identifier for the run.')
    parser.add_argument('--cov_emu_method', type=str, default='median', help='Method to compute the emulator covariance.')
    parser.add_argument('--cov_emu_diag', action='store_true', help='Whether to use only the diagonal of the emulator covariance.')
    parser.add_argument('--cov_correction', type=str, default='percival', help='Covariance correction method to use.')
    parser.add_argument('--save_dir', type=str, default='/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/fits/abacus/jan23')

    args = parser.parse_args()
    setup_logging()

    cosmo_model = args.cosmo_model
    hod_model = args.hod_model
    identifier = args.identifier

    logger.info(f'Running inference for cosmo model: {cosmo_model}, hod model: {hod_model}')

    priors, ranges, labels = get_priors(cosmo=True, hod=True)
    fixed_param_names = get_fixed_params(cosmo_model, hod_model)

    # load selected bins from the greedy search
    selected_bins = np.load(args.greedy_fn, allow_pickle=True).item()
    logger.info(f'Loading greedy bins from: {args.greedy_fn}')

    statistics = [key for key in selected_bins.keys() if len(selected_bins[key]) > 1]
    observable = get_observable(statistics)
    sampler = fit_abacus(observable)
    save_and_plot(sampler, observable)
