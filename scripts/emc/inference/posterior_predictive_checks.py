"""
Posterior Predictive Checks for Cosmological Parameter Inference

This script implements posterior predictive checks to assess goodness-of-fit
from cosmological parameter inference. It uses the posterior samples from
MCMC chains to compute p-values that indicate how well the model fits the data.

Reference: https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html
"""

import numpy as np
import argparse
from pathlib import Path
from sunbird.inference.samples import Chain
import logging


def posterior_predictive_pvalue(data, model_func, cov, theta_samples):
    """
    Calculate posterior predictive p-value.
    
    This function computes a Bayesian p-value by comparing the chi-squared
    statistic of the observed data to that of data replicated from the
    posterior predictive distribution.
    
    Parameters
    ----------
    data : np.ndarray
        Observed data vector of shape (n_features,)
    model_func : callable
        Function that takes theta parameters and returns model prediction
        of shape (n_features,)
    cov : np.ndarray
        Covariance matrix of shape (n_features, n_features)
    theta_samples : np.ndarray
        Posterior samples of parameters, shape (n_samples, n_params)
    
    Returns
    -------
    pval : float
        Posterior predictive p-value. Values close to 0 or 1 indicate
        poor fit, while values around 0.5 indicate good fit.
    T_obs : np.ndarray
        Array of chi-squared statistics for observed data at each theta
    T_rep : np.ndarray
        Array of chi-squared statistics for replicated data at each theta
    """
    Cinv = np.linalg.inv(cov)
    T_obs, T_rep = [], []

    for theta in theta_samples:
        model = model_func(theta)
        d_rep = np.random.multivariate_normal(model, cov)
        T_obs.append((data - model) @ Cinv @ (data - model))
        T_rep.append((d_rep - model) @ Cinv @ (d_rep - model))

    T_obs = np.array(T_obs)
    T_rep = np.array(T_rep)
    pval = np.mean(T_rep > T_obs)
    
    return pval, T_obs, T_rep


def load_data_vector(observable, select_mocks=None, select_coordinates=None, 
                     slice_coordinates=None):
    """
    Load data vector from EMC observable.
    
    Parameters
    ----------
    observable : str
        Name of the observable (e.g., 'spectrum', 'tpcf')
    select_mocks : dict, optional
        Dictionary to select specific mocks (e.g., {'cosmo_idx': 0, 'hod_idx': 30})
    select_coordinates : dict, optional
        Dictionary to select specific coordinates (e.g., {'multipoles': [0, 2]})
    slice_coordinates : dict, optional
        Dictionary to slice coordinates (e.g., {'s': [30, 150]})
    
    Returns
    -------
    data : np.ndarray
        Data vector
    """
    from acm.observables import emc
    
    if observable == 'spectrum':
        emc_dataset = emc.GalaxyPowerSpectrumMultipoles(
            train=True,
            select_mocks=select_mocks or {'cosmo_idx': 0, 'hod_idx': 30},
            select_coordinates=select_coordinates or {'multipoles': [0, 2]},
            slice_coordinates=slice_coordinates or {'k': [0, 0.2]}
        )
    elif observable == 'tpcf':
        emc_dataset = emc.GalaxyCorrelationFunctionMultipoles(
            train=True,
            select_mocks=select_mocks or {'cosmo_idx': 0, 'hod_idx': 30},
            select_coordinates=select_coordinates or {'multipoles': [0, 2]},
            slice_coordinates=slice_coordinates or {'s': [30, 150]}
        )
    else:
        raise ValueError(f"Observable '{observable}' not supported. "
                        "Use 'spectrum' or 'tpcf'.")
    
    data = emc_dataset.lhc_y
    return data, emc_dataset


def load_covariance_matrix(observable_dataset, divide_factor=64):
    """
    Load covariance matrix from EMC observable dataset.
    
    Parameters
    ----------
    observable_dataset : BaseObservableEMC
        EMC observable dataset instance
    divide_factor : int, optional
        Factor to divide the covariance matrix by (to account for
        effective number of realizations). Default is 64.
    
    Returns
    -------
    cov : np.ndarray
        Covariance matrix
    """
    cov = observable_dataset.get_covariance_matrix(divide_factor=divide_factor)
    return cov


def load_emulator_model(observable, model_path=None):
    """
    Load trained emulator model for making predictions.
    
    Parameters
    ----------
    observable : str
        Name of the observable (e.g., 'spectrum', 'tpcf')
    model_path : str, optional
        Path to the model checkpoint. If None, uses default path.
    
    Returns
    -------
    model : sunbird.emulators.FCN
        Trained emulator model
    """
    from sunbird.emulators import FCN
    
    if model_path is None:
        # Use default model paths based on observable
        if observable == 'spectrum':
            model_path = '/pscratch/sd/e/epaillas/emc/v1.2/trained_models/best/spectrum/last-v1.ckpt'
        elif observable == 'tpcf':
            model_path = '/pscratch/sd/e/epaillas/emc/v1.2/trained_models/best/tpcf/last-v1.ckpt'
        else:
            raise ValueError(f"No default model path for observable '{observable}'")
    
    model = FCN.load_from_checkpoint(model_path, map_location='cpu')
    model.eval()
    
    return model


def create_model_function(emulator, observable_dataset):
    """
    Create a model function that takes cosmological/HOD parameters
    and returns model prediction.
    
    Parameters
    ----------
    emulator : sunbird.emulators.FCN
        Trained emulator model
    observable_dataset : BaseObservableEMC
        EMC observable dataset instance to get parameter normalization
    
    Returns
    -------
    model_func : callable
        Function that takes theta parameters and returns model prediction
    """
    import torch
    
    def model_func(theta):
        """
        Evaluate model at given parameters.
        
        Parameters
        ----------
        theta : np.ndarray or dict
            Parameter values. Can be array or dict with parameter names.
        
        Returns
        -------
        prediction : np.ndarray
            Model prediction for the observable
        """
        # Convert theta to torch tensor
        if isinstance(theta, dict):
            # Assume theta is a dict with parameter names as keys
            # Need to order them correctly based on the observable's parameter ordering
            theta_array = np.array([theta[key] for key in sorted(theta.keys())])
        else:
            theta_array = np.array(theta)
        
        theta_tensor = torch.tensor(theta_array, dtype=torch.float32).unsqueeze(0)
        
        # Get prediction from emulator
        with torch.no_grad():
            prediction = emulator(theta_tensor).numpy().squeeze()
        
        return prediction
    
    return model_func


def load_chain_samples(chain_path, n_samples=None, burnin_fraction=0.1, 
                      add_derived=False):
    """
    Load posterior samples from chain file.
    
    Parameters
    ----------
    chain_path : str or Path
        Path to the chain file
    n_samples : int, optional
        Number of samples to use. If None, uses all samples after burnin.
    burnin_fraction : float, optional
        Fraction of samples to discard as burnin. Default is 0.1.
    add_derived : bool, optional
        Whether to add derived parameters. Default is False.
    
    Returns
    -------
    samples : np.ndarray
        Posterior samples of shape (n_samples, n_params)
    param_names : list
        List of parameter names
    """
    chain = Chain.load(chain_path)
    
    # Convert to getdist format to access samples easily
    # Note: Chain.to_getdist can be called as either a class method or instance method
    # depending on the sunbird version. We try class method first (as used in existing code),
    # then fall back to instance method if needed.
    try:
        gd_samples = Chain.to_getdist(chain, add_derived=add_derived)
    except (TypeError, AttributeError):
        gd_samples = chain.to_getdist(add_derived=add_derived)
    
    # Get samples after burnin
    n_total = gd_samples.samples.shape[0]
    n_burnin = int(burnin_fraction * n_total)
    samples = gd_samples.samples[n_burnin:]
    
    # Subsample if requested
    if n_samples is not None and n_samples < len(samples):
        indices = np.random.choice(len(samples), size=n_samples, replace=False)
        samples = samples[indices]
    
    param_names = gd_samples.getParamNames().list()
    
    return samples, param_names


def compute_chi2_at_best_fit(data, model_func, cov, best_fit_params):
    """
    Compute chi-squared and chi-squared per degree of freedom at best fit.
    
    Parameters
    ----------
    data : np.ndarray
        Observed data vector
    model_func : callable
        Model function
    cov : np.ndarray
        Covariance matrix
    best_fit_params : np.ndarray or dict
        Best fit parameter values
    
    Returns
    -------
    chi2 : float
        Chi-squared value at best fit
    dof : int
        Degrees of freedom (n_data - n_params)
    chi2_per_dof : float
        Chi-squared per degree of freedom
    """
    model = model_func(best_fit_params)
    residual = data - model
    Cinv = np.linalg.inv(cov)
    chi2 = residual @ Cinv @ residual
    
    n_data = len(data)
    n_params = len(best_fit_params)
    dof = n_data - n_params
    chi2_per_dof = chi2 / dof
    
    return chi2, dof, chi2_per_dof


def main():
    """
    Main function to run posterior predictive checks.
    """
    parser = argparse.ArgumentParser(
        description='Perform posterior predictive checks for cosmological parameter inference'
    )
    parser.add_argument('--chain', type=str, required=True,
                       help='Path to chain file')
    parser.add_argument('--observable', type=str, default='spectrum',
                       choices=['spectrum', 'tpcf'],
                       help='Observable type (spectrum or tpcf)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to emulator model checkpoint')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of posterior samples to use')
    parser.add_argument('--burnin', type=float, default=0.1,
                       help='Fraction of samples to discard as burnin')
    parser.add_argument('--divide-factor', type=int, default=64,
                       help='Factor to divide covariance matrix by')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set random seed
    np.random.seed(args.seed)
    
    logger.info(f"Loading data for observable: {args.observable}")
    
    # Load data vector
    data, observable_dataset = load_data_vector(args.observable)
    logger.info(f"Data vector shape: {data.shape}")
    
    # Load covariance matrix
    cov = load_covariance_matrix(observable_dataset, divide_factor=args.divide_factor)
    logger.info(f"Covariance matrix shape: {cov.shape}")
    
    # Load emulator model
    logger.info(f"Loading emulator model from: {args.model_path or 'default path'}")
    emulator = load_emulator_model(args.observable, model_path=args.model_path)
    
    # Create model function
    model_func = create_model_function(emulator, observable_dataset)
    
    # Load chain samples
    logger.info(f"Loading chain from: {args.chain}")
    samples, param_names = load_chain_samples(
        args.chain,
        n_samples=args.n_samples,
        burnin_fraction=args.burnin,
        add_derived=False
    )
    logger.info(f"Loaded {len(samples)} samples with parameters: {param_names}")
    
    # Compute posterior predictive p-value
    logger.info("Computing posterior predictive p-value...")
    pval, T_obs, T_rep = posterior_predictive_pvalue(data, model_func, cov, samples)
    
    # Print results
    logger.info("=" * 60)
    logger.info("POSTERIOR PREDICTIVE CHECK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Posterior predictive p-value: {pval:.4f}")
    logger.info(f"Mean T_obs (observed chi-squared): {np.mean(T_obs):.2f}")
    logger.info(f"Mean T_rep (replicated chi-squared): {np.mean(T_rep):.2f}")
    logger.info(f"Std T_obs: {np.std(T_obs):.2f}")
    logger.info(f"Std T_rep: {np.std(T_rep):.2f}")
    
    # Interpretation
    if pval < 0.05 or pval > 0.95:
        logger.warning("WARNING: P-value suggests poor model fit!")
        logger.warning("P-values close to 0 or 1 indicate the model does not fit the data well.")
    else:
        logger.info("Model fit appears reasonable.")
    
    # Compute chi2 at best fit (using mean of posterior as estimate)
    best_fit = np.mean(samples, axis=0)
    chi2, dof, chi2_per_dof = compute_chi2_at_best_fit(data, model_func, cov, best_fit)
    logger.info("=" * 60)
    logger.info(f"Chi-squared at posterior mean: {chi2:.2f}")
    logger.info(f"Degrees of freedom: {dof}")
    logger.info(f"Chi-squared per DOF: {chi2_per_dof:.4f}")
    logger.info("=" * 60)
    
    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'pval': pval,
            'T_obs': T_obs,
            'T_rep': T_rep,
            'chi2_at_best': chi2,
            'dof': dof,
            'chi2_per_dof': chi2_per_dof,
            'n_samples': len(samples),
            'observable': args.observable,
            'chain_path': args.chain,
        }
        
        np.save(output_path, results)
        logger.info(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
