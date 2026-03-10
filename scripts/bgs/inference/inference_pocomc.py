"""
PocoMC inference script for BGS observables.

Usage:
    python inference_pocomc.py --config path/to/config.yaml [other options]

    python inference_pocomc.py --slice_map "{'tpcf': {'s': [1, 200]}, 'ds_xiqg': {'s': [1, 200]}, 'ds_xiqq': {'s': [1, 200]}}"
"""

import sys
import yaml
import logging
import argparse
from pathlib import Path

import numpy as np
from sunbird.inference.pocomc import PocoMCSampler
from sunbird.inference.priors import get_priors, get_fixed_params, Bouchard25

from acm import setup_logging
from acm.utils.covariance import get_covariance_correction, check_covariance_matrix

from utils import get_observable, save_and_plot
from bgs_mocks import get_mock_data

logger = logging.getLogger(__file__.split('/')[-1])

# Define select filters map for different observables
select_filters_map = {
    'tpcf': dict(
        multipoles = [0, 2],    
    ),
    'ds_xiqg': dict(
        multipoles = [0, 2],
        quantiles = [0, 1, 3, 4],
    ),
    'ds_xigg': dict(
        multipoles = [0, 2],
        quantiles = [0, 1, 3, 4],
    ),
}

def get_observable_model(observable) -> tuple:
    """
    Load model for the given observable.

    Parameters
    ----------
    observable : CombinedObservable
        The observable object containing data and methods to get covariance and model predictions.

    Returns
    -------
    model : callable
        The theory model prediction function.
    x_names : list
        Names of the input data parameters.
    """
    x_names = observable.x_names
    model = observable.get_model_prediction # This is a method passed to the sampler !
    return model, x_names

def get_observable_covariance(
    observable, 
    *args, # See PEP 3102
    add_cov_emu: bool = False, 
    cov_emu_method: str = 'median', 
    cov_emu_diag: bool = False, 
    cov_correction: str = 'percival', 
    fixed_parameters: list[str] = None,
) -> np.ndarray:
    """
    Load covariance matrix for the given observable, adds emulator covariance if specified,
    and applies covariance correction if specified.

    Parameters
    ----------
    observable : CombinedObservable
        The observable object containing data and methods to get covariance and model predictions.
    *args :
        Additional covariances to add to the total covariance matrix.
    add_cov_emu : bool, optional
        Whether to add emulator covariance to the total covariance matrix. Defaults to False.
    cov_emu_method : str, optional
        Method to compute the emulator covariance. Defaults to 'median'.
    cov_emu_diag : bool, optional
        Whether to use only the diagonal of the emulator covariance. Defaults to False.
    cov_correction : str, optional
        Covariance correction method to use. Defaults to 'percival'.
    fixed_parameters : list[str], optional
        List of fixed parameter names for covariance correction. Defaults to None.

    Returns
    -------
    cov : np.ndarray
        The total covariance matrix.
    """
    cov = observable.get_covariance_matrix(volume_factor=64)
    logger.info(f'Loaded covariance matrix with shape: {cov.shape}')
    
    if add_cov_emu:
        cov_emu = observable.get_emulator_covariance_matrix(method=cov_emu_method, diag=cov_emu_diag)
        logger.info(f'Loaded emulator covariance matrix with shape: {cov_emu.shape}')
        cov += cov_emu
    
    if args: 
        for additional_cov in args:
            cov += additional_cov
            
    if cov_correction is not None:
        correction = get_covariance_correction(
            method = cov_correction, 
            n_s = len(observable.covariance_y),
            n_d = len(cov),
            n_theta = len(observable.x_names) - len(fixed_parameters),
        )
        logger.info(f'Applying covariance correction: {correction}')
        cov *= correction
    
    logger.info(f'Final covariance matrix tests (warnings will follow if any fail):')
    check_covariance_matrix(cov, name='Total covariance matrix')
    
    return cov

def get_observable_data(observable) -> tuple: 
    """
    Load observable data (x, y) from the given observable.

    Parameters
    ----------
    observable : CombinedObservable
        The observable object containing data.

    Returns
    -------
    x : np.ndarray
        The input data (e.g., cosmo and hod parameters).
    y : np.ndarray
        The observed data (e.g., clustering measurements).
    """
    x = observable.x
    y = observable.y
    logger.info(f'Loaded observable x with shape: {x.shape}')
    logger.info(f'Loaded observable y with shape: {y.shape}')
    return x, y

def fit_pocomc(
    observable, 
    priors: dict, 
    ranges: dict, 
    labels: dict, 
    fixed_param_names: list[str],
    fit_type: str = 'validation',
    add_cov_emu: bool = False,
    cov_emu_method: str = 'median',
    cov_emu_diag: bool = False,
    cov_correction: str = 'percival',
) -> PocoMCSampler:
    """
    Fit PocoMC sampler to the given observable data and model.
    
    Parameters
    ----------
    observable : CombinedObservable
        The observable object containing data and methods to get covariance and model predictions.
    priors : dict
        A dictionary of all parameter priors.
    ranges : dict
        A dictionary of parameter ranges.
    labels : dict
        A dictionary of parameter labels.
    fixed_param_names : list[str]
        A list of names of parameters to be held fixed during sampling.
    fit_type : str, optional
        Type of mock to fit: 'validation' or 'secondgen'. Defaults to 'validation'.
    add_cov_emu : bool, optional
        Whether to add emulator covariance to the total covariance matrix. Defaults to False.
    cov_emu_method : str, optional
        Method to compute the emulator covariance. Defaults to 'median'.
    cov_emu_diag : bool, optional
        Whether to use only the diagonal of the emulator covariance. Defaults to False.
    cov_correction : str, optional
        Covariance correction method to use. Defaults to 'percival'.
    
    Returns
    -------
    sampler : PocoMCSampler
        The configured PocoMCSampler instance after sampling.
    """
    theory_model, x_names = get_observable_model(observable)
    
    covariances = []
    if fit_type == 'validation':
        x, observation = get_observable_data(observable)
    elif fit_type == 'secondgen':
        x, observation, extra_cov = get_mock_data(observable, mock_type='SecondGen')
        covariances.append(extra_cov)
    elif fit_type == 'uchuu':
        x, observation = get_mock_data(observable, mock_type='Uchuu', add_covariance=False)
    else:
        raise ValueError(f'Unknown fit_type: {fit_type}')
    
    cov = get_observable_covariance(
        observable, 
        *covariances,
        add_cov_emu=add_cov_emu, 
        cov_emu_method=cov_emu_method, 
        cov_emu_diag=cov_emu_diag,
        cov_correction=cov_correction,
        fixed_parameters=fixed_param_names,
    )
    
    # Ensure prior keys are in the same order as data_names
    priors = {key: priors[key] for key in x_names} # FIXME: unordered objects should not require ordering
    fixed_parameters = {key: x[x_names.index(key)] for key in fixed_param_names}
    markers = {key: x[x_names.index(key)] for key in x_names if key not in fixed_param_names} # True value markers
    
    precision_matrix = np.linalg.inv(cov)
    
    sampler = PocoMCSampler(
        observation = observation,
        precision_matrix = precision_matrix,
        theory_model = theory_model,
        fixed_parameters = fixed_parameters,
        priors = priors, # prior keys order must match x_names order 
        ranges = ranges, # ranges and labels order is not important
        labels = labels,
        ellipsoid = True,
        markers = markers,
    )
    sampler(vectorize=True, n_total=4096)
    
    return sampler


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to a configuration file (YAML format). Command line arguments override config file settings.')
    parser.add_argument('--dump_config', action='store_true', help='Dump the current configuration and exit.')
    parser.add_argument('--module', type=str, default='acm.observables.bgs', help='Base module path for observables')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory where the observable compressed data is stored.')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory where the emulatir is stored.')
    parser.add_argument('--statistics', nargs='+', help='List of statistics to use, e.g. GalaxyPowerSpectrumMultipoles')
    parser.add_argument('--slice_map', type=str, default=None, help='Expression of slice_filters_map to use (evaluated as a dictionary). Defaults as None')
    parser.add_argument('-t', '--fit_type', type=str, default='validation', help='Type of mock to fit: validation or secondgen.')
    parser.add_argument('-ce', '--add_cov_emu', action='store_true', help='Whether to add emulator covariance or not.')
    parser.add_argument('-cm', '--cosmo_model', type=str, default='base', help='Cosmological model to use.')
    parser.add_argument('-hm', '--hod_model', type=str, default='base-VB-AB-CB-s', help='HOD model to use.')
    parser.add_argument('-id', '--identifier', type=str, default=None, help='Identifier for the run.')
    parser.add_argument('-d', '--cov_emu_diag', action='store_true', help='Whether to use only the diagonal of the emulator covariance.')
    parser.add_argument('--cov_emu_method', type=str, default='median', help='Method to compute the emulator covariance.')
    parser.add_argument('--cov_correction', type=str, default='percival', help='Covariance correction method to use.')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save results.')
    parser.add_argument('--cosmo_idx', type=int, default=0, help='Index of the cosmology to fit, if fit_type is validation.')
    parser.add_argument('--hod_idx', type=int, default=0, help='Index of the HOD to fit, if fit_type is validation.')
    parser.add_argument('--obs_kwargs', type=str, default=None, help='Mapping of additional keyword arguments to pass at observable load. Will override all other passed arguments if provided')
    
    args = parser.parse_args()
    
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)
            parser.set_defaults(**config_args)
        args = parser.parse_args() # Re-parse arguments with config defaults
    if args.dump_config:
        parser.print_help(sys.stdout)
        print("\nCurrent configuration:")
        print("----------------------")
        tmp_args = vars(args).copy()
        del tmp_args['config']
        del tmp_args['dump_config']
        for arg in tmp_args:
            print(f"{arg}: {getattr(args, arg)}")
        sys.exit(-1)
        
    setup_logging()
    
    module = args.module
    statistics = args.statistics
    cosmo_model = args.cosmo_model
    hod_model = args.hod_model
    identifier = args.identifier
    fit_type = args.fit_type
    
    # Observable paths
    paths = dict(
        data_dir = args.data_dir,
        model_dir = args.model_dir,
    )
    
    logger.info(f'Running inference for cosmo model: {cosmo_model}, hod model: {hod_model}')
    
    if args.slice_map is not None:
        slice_filters_map = eval(args.slice_map)
        logger.info(f'Using slice_filters_map: {slice_filters_map}')
    else:
        slice_filters_map = None
    
    if args.obs_kwargs is not None:
        kwargs_map = eval(args.obs_kwargs)
        logger.info(f'Using kwargs_map for observables: {kwargs_map}')
    else:
        kwargs_map = None
    
    priors, ranges, labels = get_priors(hod_class=Bouchard25)
    fixed_param_names = get_fixed_params(cosmo_model, hod_model, priors)
    
    observable = get_observable(
        statistics, 
        module=module, 
        select_filters_map=select_filters_map,
        slice_filters_map=slice_filters_map,
        kwargs_map=kwargs_map,
        paths=paths,
        numpy_output=True, 
        squeeze_output=True,
        select_filters = {'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx},
    )
    
    sampler = fit_pocomc(
        observable, 
        priors, 
        ranges, 
        labels, 
        fixed_param_names,
        fit_type=fit_type,
        add_cov_emu=args.add_cov_emu,
        cov_emu_method=args.cov_emu_method,
        cov_emu_diag=args.cov_emu_diag,
        cov_correction=args.cov_correction,
    )
    
    if fit_type == 'validation':
        save_dir = Path(args.save_dir) / fit_type / f'c{args.cosmo_idx:03d}_hod{args.hod_idx:03d}' / f'cosmo-{cosmo_model}_hod-{hod_model}'
    else:
        save_dir = Path(args.save_dir) / fit_type / f'cosmo-{cosmo_model}_hod-{hod_model}' # No indices for usual mocks fits
    
    save_and_plot(
        sampler, 
        observable, 
        identifier=identifier,
        save_dir=save_dir,
    )