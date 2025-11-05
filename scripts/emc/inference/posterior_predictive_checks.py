"""
Posterior Predictive Checks for Cosmological Parameter Inference

This module provides functions for performing posterior predictive checks to assess
goodness-of-fit from cosmological parameter inference. It includes the core algorithm
for computing p-values and visualization routines for analyzing results.

The main use case is with synthetic or custom data. See example_synthetic.py for
a complete working example.

Reference: https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html
"""

import numpy as np
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
    
    Examples
    --------
    >>> # Simple linear model example
    >>> x = np.linspace(0, 10, 20)
    >>> X = np.vstack([np.ones(20), x]).T
    >>> theta_true = np.array([1.0, 2.0])
    >>> data = X @ theta_true + np.random.normal(0, 0.5, 20)
    >>> cov = 0.5**2 * np.eye(20)
    >>> def model_func(theta):
    ...     return X @ theta
    >>> theta_samples = theta_true + np.random.randn(100, 2) * 0.1
    >>> pval, T_obs, T_rep = posterior_predictive_pvalue(data, model_func, cov, theta_samples)
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

