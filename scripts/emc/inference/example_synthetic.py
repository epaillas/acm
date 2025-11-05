"""
Example script demonstrating posterior predictive checks with synthetic data.

This script shows how to use the posterior_predictive_pvalue function
with synthetic data when actual EMC data or trained models are not available.
"""

import numpy as np


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
    n_params = len(best_fit_params) if isinstance(best_fit_params, (list, np.ndarray)) else len(best_fit_params)
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


if __name__ == '__main__':
    # Run example with good model fit
    create_synthetic_example()
    
    # Run example with poor model fit
    create_bad_fit_example()
    
    print("\n")
    print("These examples demonstrate how posterior predictive checks work.")
    print("For actual cosmological inference, use the main script with")
    print("real data and trained emulator models:")
    print("")
    print("  python posterior_predictive_checks.py \\")
    print("      --chain /path/to/chain.npy \\")
    print("      --observable spectrum \\")
    print("      --n-samples 1000")
    print("")
