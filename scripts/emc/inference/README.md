# EMC Inference Scripts

This directory contains scripts for performing inference on EMC (Emulator Mock Challenge) observables.

## Posterior Predictive Checks

The `posterior_predictive_checks.py` module provides functions for performing posterior predictive checks to assess goodness-of-fit from cosmological parameter inference.

### Core Functions

The module provides the following functions:

1. **`posterior_predictive_pvalue(data, model_func, cov, theta_samples)`**
   - Computes the posterior predictive p-value by comparing observed and replicated chi-squared statistics
   - Returns: `pval`, `T_obs`, `T_rep`

2. **`compute_chi2_at_best_fit(data, model_func, cov, best_fit_params)`**
   - Computes chi-squared and chi-squared per degree of freedom at the best fit
   - Returns: `chi2`, `dof`, `chi2_per_dof`

3. **Visualization Functions**:
   - `plot_ppc_results(T_obs, T_rep, pval, output_path=None)` - Histogram comparison
   - `plot_ppc_scatter(T_obs, T_rep, pval, output_path=None)` - Scatter plot
   - `plot_ppc_summary(T_obs, T_rep, pval, chi2, dof, chi2_per_dof, output_path=None)` - Comprehensive 4-panel summary

### Usage

Import the functions and use them with your own data:

```python
from posterior_predictive_checks import (
    posterior_predictive_pvalue,
    compute_chi2_at_best_fit,
    plot_ppc_summary
)
import numpy as np

# Your data, model function, covariance, and posterior samples
# data = ...
# model_func = ...
# cov = ...
# theta_samples = ...

# Compute p-value
pval, T_obs, T_rep = posterior_predictive_pvalue(data, model_func, cov, theta_samples)

# Compute chi-squared at best fit
best_fit = np.mean(theta_samples, axis=0)
chi2, dof, chi2_per_dof = compute_chi2_at_best_fit(data, model_func, cov, best_fit)

# Generate summary plot
plot_ppc_summary(T_obs, T_rep, pval, chi2, dof, chi2_per_dof, 
                output_path='ppc_summary.png')
```

### Example with Synthetic Data

Run the `example_synthetic.py` script to see posterior predictive checks in action with synthetic data:

```bash
python example_synthetic.py
```

This will:
- Run two examples: a well-fitting model and a misspecified model
- Print diagnostic statistics for each
- Generate plots demonstrating:
  - A well-fitting model (good p-value)
  - A misspecified model (poor p-value)

The example demonstrates how to:
- Set up synthetic data with known properties
- Define a model function
- Generate posterior samples
- Compute posterior predictive checks
- Visualize the results

### Visualization

The module provides three types of plots:

1. **Histogram Plot**: Overlaid histograms comparing the distributions of T_obs and T_rep with their means marked

2. **Scatter Plot**: Scatter plot of T_obs vs T_rep with a diagonal reference line showing where T_rep = T_obs

3. **Summary Plot**: Comprehensive 4-panel figure showing:
   - Histogram comparison
   - Scatter plot
   - Cumulative distribution functions
   - Summary statistics table with color-coded fit assessment

The plots visually indicate whether the model provides a good fit to the data by comparing the observed chi-squared statistics (T_obs) with those from replicated data (T_rep).

### Interpretation

- **P-value close to 0.5**: Good model fit
- **P-value < 0.05 or > 0.95**: Poor model fit, model may not adequately describe the data
- **T_rep and T_obs distributions should overlap**: If they're very different, the model is not capturing the data well

### Implementing with Real Data

To use this module with real EMC data:

1. Load your data vector and covariance matrix
2. Load your emulator or model for making predictions
3. Create a model function that takes parameters and returns predictions
4. Load your MCMC chain samples (e.g., using sunbird)
5. Call `posterior_predictive_pvalue()` with your data
6. Visualize results using the plotting functions

See `example_synthetic.py` for a complete working template that you can adapt to your real data.

### Reference

For more information on posterior predictive checks, see:
https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html
