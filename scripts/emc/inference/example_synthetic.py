"""
Example script demonstrating posterior predictive checks with synthetic data.

This script shows how to use the posterior_predictive_pvalue function
with synthetic data when actual EMC data or trained models are not available.
It also demonstrates the visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_ppc_summary_inline(T_obs, T_rep, pval, chi2_at_best, dof, chi2_per_dof, 
                            output_path=None):
    """
    Create comprehensive summary plot with multiple panels.
    Inline version for the example that doesn't require external imports.
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


def posterior_predictive_pvalue(data, model_func, cov, theta_samples):
    """
    Calculate posterior predictive p-value.
    
    This is a copy of the function from posterior_predictive_checks.py
    for demonstration purposes without requiring all dependencies.
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
    
    This is a copy of the function from posterior_predictive_checks.py
    for demonstration purposes without requiring all dependencies.
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


def create_synthetic_example():
    """
    Create a synthetic example to demonstrate posterior predictive checks.
    
    This example simulates a simple linear model with Gaussian noise.
    """
    print("=" * 60)
    print("SYNTHETIC EXAMPLE: POSTERIOR PREDICTIVE CHECKS")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define dimensions
    n_features = 20  # Number of data points
    n_params = 3     # Number of model parameters
    n_samples = 500  # Number of posterior samples
    
    print(f"\nDimensions:")
    print(f"  Number of data points: {n_features}")
    print(f"  Number of parameters: {n_params}")
    print(f"  Number of posterior samples: {n_samples}")
    
    # True parameter values
    theta_true = np.array([1.0, 2.0, -0.5])
    print(f"\nTrue parameters: {theta_true}")
    
    # Create design matrix for linear model
    x = np.linspace(0, 10, n_features)
    X = np.vstack([np.ones(n_features), x, x**2]).T
    
    # Generate observed data with noise
    y_true = X @ theta_true
    noise_std = 0.5
    noise = np.random.normal(0, noise_std, n_features)
    data = y_true + noise
    
    # Create covariance matrix (assuming uncorrelated Gaussian noise)
    cov = noise_std**2 * np.eye(n_features)
    
    # Define model function
    def model_func(theta):
        """Linear model: y = theta[0] + theta[1]*x + theta[2]*x^2"""
        return X @ theta
    
    # Generate posterior samples (simulating MCMC output)
    # Add some spread around true values
    posterior_std = np.array([0.1, 0.15, 0.08])
    theta_samples = theta_true + np.random.randn(n_samples, n_params) * posterior_std
    
    print(f"\nPosterior samples:")
    print(f"  Mean: {np.mean(theta_samples, axis=0)}")
    print(f"  Std:  {np.std(theta_samples, axis=0)}")
    
    # Compute posterior predictive p-value
    print("\n" + "=" * 60)
    print("COMPUTING POSTERIOR PREDICTIVE P-VALUE...")
    print("=" * 60)
    
    pval, T_obs, T_rep = posterior_predictive_pvalue(
        data, model_func, cov, theta_samples
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Posterior predictive p-value: {pval:.4f}")
    print(f"  Mean T_obs (observed chi-squared): {np.mean(T_obs):.2f}")
    print(f"  Mean T_rep (replicated chi-squared): {np.mean(T_rep):.2f}")
    print(f"  Std T_obs: {np.std(T_obs):.2f}")
    print(f"  Std T_rep: {np.std(T_rep):.2f}")
    
    # Interpretation
    print("\nInterpretation:")
    if 0.05 < pval < 0.95:
        print("  ✓ P-value is in reasonable range (0.05 - 0.95)")
        print("  ✓ Model appears to fit the data well")
    else:
        print("  ✗ P-value is outside reasonable range")
        print("  ✗ Model may not fit the data well")
    
    # Compute chi2 at best fit
    best_fit = np.mean(theta_samples, axis=0)
    chi2, dof, chi2_per_dof = compute_chi2_at_best_fit(
        data, model_func, cov, best_fit
    )
    
    print("\n" + "=" * 60)
    print("CHI-SQUARED AT BEST FIT (POSTERIOR MEAN)")
    print("=" * 60)
    print(f"  Chi-squared: {chi2:.2f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  Chi-squared per DOF: {chi2_per_dof:.4f}")
    
    if 0.5 < chi2_per_dof < 2.0:
        print("  ✓ Chi-squared per DOF is reasonable")
    else:
        print("  ! Chi-squared per DOF may indicate issues")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    return pval, T_obs, T_rep, chi2, dof, chi2_per_dof


def create_bad_fit_example():
    """
    Create an example with a misspecified model to demonstrate
    how posterior predictive checks detect poor fit.
    """
    print("\n\n")
    print("=" * 60)
    print("EXAMPLE WITH POOR MODEL FIT")
    print("=" * 60)
    
    np.random.seed(123)
    
    n_features = 20
    n_params = 2  # Using only 2 params when true model needs 3
    n_samples = 500
    
    # True model is quadratic, but we'll fit with only linear
    x = np.linspace(0, 10, n_features)
    X_true = np.vstack([np.ones(n_features), x, x**2]).T
    theta_true = np.array([1.0, 2.0, -0.5])
    
    y_true = X_true @ theta_true
    noise_std = 0.5
    noise = np.random.normal(0, noise_std, n_features)
    data = y_true + noise
    
    cov = noise_std**2 * np.eye(n_features)
    
    # Define WRONG model (linear instead of quadratic)
    X_wrong = np.vstack([np.ones(n_features), x]).T
    
    def model_func_wrong(theta):
        """Linear model (misspecified - should be quadratic)"""
        return X_wrong @ theta
    
    # Generate posterior samples for wrong model
    theta_samples_wrong = np.random.randn(n_samples, n_params) * 0.1 + np.array([5.0, 0.5])
    
    # Compute posterior predictive p-value
    pval, T_obs, T_rep = posterior_predictive_pvalue(
        data, model_func_wrong, cov, theta_samples_wrong
    )
    
    print(f"\nResults with misspecified model:")
    print(f"  Posterior predictive p-value: {pval:.4f}")
    print(f"  Mean T_obs: {np.mean(T_obs):.2f}")
    print(f"  Mean T_rep: {np.mean(T_rep):.2f}")
    
    print("\nInterpretation:")
    if pval < 0.05 or pval > 0.95:
        print("  ✓ P-value correctly indicates poor model fit!")
        print("  ✓ Posterior predictive check detected model misspecification")
    else:
        print("  ! P-value did not detect misspecification (may need more samples)")
    
    best_fit = np.mean(theta_samples_wrong, axis=0)
    chi2, dof, chi2_per_dof = compute_chi2_at_best_fit(
        data, model_func_wrong, cov, best_fit
    )
    
    print(f"\n  Chi-squared per DOF: {chi2_per_dof:.4f}")
    if chi2_per_dof > 2.0:
        print("  ✓ Chi-squared per DOF also indicates poor fit")
    
    print("\n" + "=" * 60)
    
    return pval, T_obs, T_rep, chi2, dof, chi2_per_dof


if __name__ == '__main__':
    # Run example with good model fit
    print("Running synthetic examples...")
    pval1, T_obs1, T_rep1, chi2_1, dof1, chi2_per_dof1 = create_synthetic_example()
    
    # Run example with poor model fit
    pval2, T_obs2, T_rep2, chi2_2, dof2, chi2_per_dof2 = create_bad_fit_example()
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 60)
    
    try:
        # Plot for good fit example
        print("\nGenerating plots for good fit example...")
        plot_ppc_summary_inline(T_obs1, T_rep1, pval1, chi2_1, dof1, chi2_per_dof1,
                       output_path='example_good_fit_summary.png')
        print("  ✓ Saved: example_good_fit_summary.png")
        
        # Plot for poor fit example
        print("\nGenerating plots for poor fit example...")
        plot_ppc_summary_inline(T_obs2, T_rep2, pval2, chi2_2, dof2, chi2_per_dof2,
                       output_path='example_poor_fit_summary.png')
        print("  ✓ Saved: example_poor_fit_summary.png")
        
        print("\nPlots generated successfully!")
    except Exception as e:
        print(f"\nWarning: Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n")
    print("=" * 60)
    print("These examples demonstrate how posterior predictive checks work.")
    print("=" * 60)
    print("")
    print("To use with your own data:")
    print("  1. Import the functions from posterior_predictive_checks.py")
    print("  2. Load your data, covariance matrix, and posterior samples")
    print("  3. Define a model function")
    print("  4. Call posterior_predictive_pvalue() and plotting functions")
    print("")
    print("Example:")
    print("  from posterior_predictive_checks import \\")
    print("      posterior_predictive_pvalue, plot_ppc_summary")
    print("  pval, T_obs, T_rep = posterior_predictive_pvalue(")
    print("      data, model_func, cov, theta_samples)")
    print("  plot_ppc_summary(T_obs, T_rep, pval, chi2, dof, chi2_per_dof)")
    print("")

