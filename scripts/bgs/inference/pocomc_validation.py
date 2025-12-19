import logging
import argparse
import numpy as np
from pathlib import Path
from utils import get_fixed_params # Will be moved to sunbird 
from sunbird.inference.pocomc import PocoMCSampler
from sunbird.inference.priors import get_priors, Bouchard25
from acm import setup_logging
from acm.utils.covariance import get_covariance_correction
from utils import get_observable, save_and_plot

logger = logging.getLogger(__name__)

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

# Define functions for inference
def get_data_model_cov(
    observable, 
    add_cov_emu: bool = False, 
    cov_emu_method: str = 'median', 
    cov_emu_diag: bool = False, 
    cov_correction: str = 'percival'
) -> tuple:
    """
    Load data, model, and covariance matrix for the given observable.

    Parameters
    ----------
    observable : CombinedObservable
        The observable object containing data and methods to get covariance and model predictions.
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
    data_x : np.ndarray
        The input data (e.g., cosmo and hod parameters).
    data_x_names : list
        Names of the input data parameters.
    data_y : np.ndarray
        The observed data (e.g., clustering measurements).
    model : callable
        The theory model prediction function.
    cov : np.ndarray
        The total covariance matrix.
    """
    x = observable.x
    x_names = observable.x_names
    y = observable.y

    logger.info(f'Loaded data_x with shape: {x.shape}')
    logger.info(f'Loaded data_y with shape {y.shape}')

    # load the covariance matrix, including emulator error and Percival correction
    cov = observable.get_covariance_matrix(volume_factor=64)
    logger.info(f'Loaded covariance matrix with shape: {cov.shape}')
    
    if add_cov_emu:
        cov_emu = observable.get_emulator_covariance_matrix(method=cov_emu_method, diag=cov_emu_diag)
        logger.info(f'Loaded emulator covariance matrix with shape: {cov_emu.shape}')
        cov += cov_emu
    
    if cov_correction is not None:
        correction = get_covariance_correction(
            method = cov_correction, 
            n_s = len(observable.covariance_y),
            n_d = len(cov),
            n_theta = len(x_names) - len(fixed_param_names),
        )
        logger.info(f'Applying covariance correction: {correction}')
        cov *= correction
        
    # Load the theory model
    model = observable.get_model_prediction # This is a method passed to the sampler !
    
    return x, x_names, y, model, cov

def fit_pocomc(
    observable, 
    priors: dict, 
    ranges: dict, 
    labels: dict, 
    fixed_param_names: list[str],
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
    x, x_names, observation, theory_model, cov = get_data_model_cov(
        observable, 
        add_cov_emu=add_cov_emu, 
        cov_emu_method=cov_emu_method, 
        cov_emu_diag=cov_emu_diag,
        cov_correction=cov_correction,
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
    parser.add_argument('--module', type=str, default='acm.observables.bgs', help='Base module path for observables')
    parser.add_argument('--statistics', nargs='+', help='List of statistics to use, e.g. GalaxyPowerSpectrumMultipoles')
    parser.add_argument("--cosmo_idx", type=int, default=0)
    parser.add_argument("--hod_idx", type=int, default=0)
    parser.add_argument('--add_cov_emu', action='store_true', help='Whether to add emulator covariance or not.')
    parser.add_argument('--cosmo_model', type=str, default='base', help='Cosmological model to use.')
    parser.add_argument('--hod_model', type=str, default='base-VB-AB-CB-s', help='HOD model to use.')
    parser.add_argument('--identifier', type=str, default=None, help='Identifier for the run.')
    parser.add_argument('--cov_emu_method', type=str, default='median', help='Method to compute the emulator covariance.')
    parser.add_argument('--cov_emu_diag', action='store_true', help='Whether to use only the diagonal of the emulator covariance.')
    parser.add_argument('--cov_correction', type=str, default='percival', help='Covariance correction method to use.')
    parser.add_argument('--save_dir', type=str, default='/pscratch/sd/s/sbouchar/acm/bgs/chains/validation/')
    
    args = parser.parse_args()
    setup_logging()
    
    module = args.module
    statistics = args.statistics
    cosmo_model = args.cosmo_model
    hod_model = args.hod_model
    identifier = args.identifier
    
    logger.info(f'Running inference for cosmo model: {cosmo_model}, hod model: {hod_model}')
    
    priors, ranges, labels = get_priors(hod_class=Bouchard25)
    fixed_param_names = get_fixed_params(cosmo_model, hod_model, priors)
    
    observable = get_observable(
        statistics, 
        module=module, 
        select_filters_map=select_filters_map,
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
        add_cov_emu=args.add_cov_emu,
        cov_emu_method=args.cov_emu_method,
        cov_emu_diag=args.cov_emu_diag,
        cov_correction=args.cov_correction,
    )
    
    save_dir = Path(args.save_dir) / f'c{args.cosmo_idx:03d}_h{args.hod_idx:03d}' / f'cosmo-{cosmo_model}_hod-{hod_model}'
    
    save_and_plot(
        sampler, 
        observable, 
        save_dir=save_dir,
    )