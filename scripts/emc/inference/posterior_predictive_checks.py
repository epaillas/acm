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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def plot_ppc_results(T_obs, T_rep, pval, output_path=None):
    """
    Create visualization of posterior predictive check results.
    
    This function creates a comparison plot showing:
    - Histograms of T_obs and T_rep distributions
    - The p-value
    - Visual indicators of model fit quality
    
    Parameters
    ----------
    T_obs : np.ndarray
        Array of chi-squared statistics for observed data
    T_rep : np.ndarray
        Array of chi-squared statistics for replicated data
    pval : float
        Posterior predictive p-value
    output_path : str or Path, optional
        Path to save the figure. If None, figure is displayed.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    bins = np.linspace(min(T_obs.min(), T_rep.min()), 
                       max(T_obs.max(), T_rep.max()), 50)
    
    ax.hist(T_rep, bins=bins, alpha=0.6, label='T_rep (replicated)', 
            color='steelblue', edgecolor='black', linewidth=0.5)
    ax.hist(T_obs, bins=bins, alpha=0.6, label='T_obs (observed)', 
            color='coral', edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for means
    ax.axvline(np.mean(T_rep), color='steelblue', linestyle='--', 
               linewidth=2, label=f'Mean T_rep = {np.mean(T_rep):.1f}')
    ax.axvline(np.mean(T_obs), color='coral', linestyle='--', 
               linewidth=2, label=f'Mean T_obs = {np.mean(T_obs):.1f}')
    
    # Add p-value text
    fit_status = 'Good fit' if 0.05 < pval < 0.95 else 'Poor fit'
    textstr = f'p-value = {pval:.4f}\n{fit_status}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.72, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Chi-squared statistic', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Posterior Predictive Check: Distribution of Test Statistics', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_ppc_scatter(T_obs, T_rep, pval, output_path=None):
    """
    Create scatter plot comparing T_obs and T_rep.
    
    Parameters
    ----------
    T_obs : np.ndarray
        Array of chi-squared statistics for observed data
    T_rep : np.ndarray
        Array of chi-squared statistics for replicated data
    pval : float
        Posterior predictive p-value
    output_path : str or Path, optional
        Path to save the figure. If None, figure is displayed.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(T_obs, T_rep, alpha=0.5, s=20, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (y=x)
    min_val = min(T_obs.min(), T_rep.min())
    max_val = max(T_obs.max(), T_rep.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
            label='T_rep = T_obs', alpha=0.7)
    
    # Add p-value text
    fit_status = 'Good fit' if 0.05 < pval < 0.95 else 'Poor fit'
    textstr = f'p-value = {pval:.4f}\n{fit_status}\n\nPoints above line:\n{np.sum(T_rep > T_obs)}/{len(T_obs)} ({100*pval:.1f}%)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('T_obs (observed chi-squared)', fontsize=12)
    ax.set_ylabel('T_rep (replicated chi-squared)', fontsize=12)
    ax.set_title('Posterior Predictive Check: T_obs vs T_rep', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_ppc_summary(T_obs, T_rep, pval, chi2_at_best, dof, chi2_per_dof, 
                     output_path=None):
    """
    Create comprehensive summary plot with multiple panels.
    
    Parameters
    ----------
    T_obs : np.ndarray
        Array of chi-squared statistics for observed data
    T_rep : np.ndarray
        Array of chi-squared statistics for replicated data
    pval : float
        Posterior predictive p-value
    chi2_at_best : float
        Chi-squared at best fit
    dof : int
        Degrees of freedom
    chi2_per_dof : float
        Chi-squared per degree of freedom
    output_path : str or Path, optional
        Path to save the figure. If None, figure is displayed.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Histogram comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(min(T_obs.min(), T_rep.min()), 
                       max(T_obs.max(), T_rep.max()), 40)
    ax1.hist(T_rep, bins=bins, alpha=0.6, label='T_rep (replicated)', 
             color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.hist(T_obs, bins=bins, alpha=0.6, label='T_obs (observed)', 
             color='coral', edgecolor='black', linewidth=0.5)
    ax1.axvline(np.mean(T_rep), color='steelblue', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(T_obs), color='coral', linestyle='--', linewidth=2)
    ax1.set_xlabel('Chi-squared statistic', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Distribution of Test Statistics', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Scatter plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(T_obs, T_rep, alpha=0.5, s=20, color='steelblue', 
                edgecolors='black', linewidth=0.5)
    min_val = min(T_obs.min(), T_rep.min())
    max_val = max(T_obs.max(), T_rep.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('T_obs', fontsize=11)
    ax2.set_ylabel('T_rep', fontsize=11)
    ax2.set_title('T_obs vs T_rep', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Panel 3: Cumulative distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sorted_T_obs = np.sort(T_obs)
    sorted_T_rep = np.sort(T_rep)
    ax3.plot(sorted_T_obs, np.arange(1, len(T_obs) + 1) / len(T_obs), 
             label='T_obs', color='coral', linewidth=2)
    ax3.plot(sorted_T_rep, np.arange(1, len(T_rep) + 1) / len(T_rep), 
             label='T_rep', color='steelblue', linewidth=2)
    ax3.set_xlabel('Chi-squared statistic', fontsize=11)
    ax3.set_ylabel('Cumulative probability', fontsize=11)
    ax3.set_title('Cumulative Distribution Functions', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Summary statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    fit_status = 'GOOD FIT ✓' if 0.05 < pval < 0.95 else 'POOR FIT ✗'
    fit_color = 'green' if 0.05 < pval < 0.95 else 'red'
    
    summary_text = f"""
    POSTERIOR PREDICTIVE CHECK SUMMARY
    {'=' * 42}
    
    P-value:              {pval:.4f}
    Model fit status:     {fit_status}
    
    {'─' * 42}
    
    T_obs (observed):
      Mean:               {np.mean(T_obs):.2f}
      Std:                {np.std(T_obs):.2f}
      Min:                {np.min(T_obs):.2f}
      Max:                {np.max(T_obs):.2f}
    
    T_rep (replicated):
      Mean:               {np.mean(T_rep):.2f}
      Std:                {np.std(T_rep):.2f}
      Min:                {np.min(T_rep):.2f}
      Max:                {np.max(T_rep):.2f}
    
    {'─' * 42}
    
    Chi-squared at best fit: {chi2_at_best:.2f}
    Degrees of freedom:      {dof}
    Chi-squared per DOF:     {chi2_per_dof:.4f}
    
    {'─' * 42}
    
    Samples used:            {len(T_obs)}
    Points where T_rep > T_obs: {np.sum(T_rep > T_obs)} ({100*pval:.1f}%)
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', 
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Add colored box for fit status
    status_y = 0.78
    if 0.05 < pval < 0.95:
        ax4.add_patch(plt.Rectangle((0.58, status_y), 0.3, 0.05, 
                                     transform=ax4.transAxes, 
                                     facecolor='lightgreen', alpha=0.5))
    else:
        ax4.add_patch(plt.Rectangle((0.58, status_y), 0.3, 0.05, 
                                     transform=ax4.transAxes, 
                                     facecolor='lightcoral', alpha=0.5))
    
    fig.suptitle('Posterior Predictive Check Results', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


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
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--plot-output', type=str, default=None,
                       help='Directory to save plots (defaults to same dir as output)')
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
    
    # Generate plots if requested
    if args.plot:
        logger.info("Generating visualization plots...")
        
        # Determine plot output directory
        if args.plot_output:
            plot_dir = Path(args.plot_output)
        elif args.output:
            plot_dir = Path(args.output).parent
        else:
            plot_dir = Path('.')
        
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Create base filename for plots
        if args.output:
            base_name = Path(args.output).stem
        else:
            base_name = f"ppc_{args.observable}"
        
        # Generate histogram comparison plot
        histogram_path = plot_dir / f"{base_name}_histogram.png"
        plot_ppc_results(T_obs, T_rep, pval, output_path=histogram_path)
        logger.info(f"Histogram plot saved to: {histogram_path}")
        
        # Generate scatter plot
        scatter_path = plot_dir / f"{base_name}_scatter.png"
        plot_ppc_scatter(T_obs, T_rep, pval, output_path=scatter_path)
        logger.info(f"Scatter plot saved to: {scatter_path}")
        
        # Generate comprehensive summary plot
        summary_path = plot_dir / f"{base_name}_summary.png"
        plot_ppc_summary(T_obs, T_rep, pval, chi2, dof, chi2_per_dof, 
                        output_path=summary_path)
        logger.info(f"Summary plot saved to: {summary_path}")
        
        logger.info("All plots generated successfully!")


if __name__ == '__main__':
    main()
