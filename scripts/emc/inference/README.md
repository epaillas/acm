# EMC Inference Scripts

This directory contains scripts for performing inference on EMC (Emulator Mock Challenge) observables.

## Posterior Predictive Checks

The `posterior_predictive_checks.py` script implements posterior predictive checks to assess the goodness-of-fit from cosmological parameter inference.

### Usage

Basic usage:
```bash
python posterior_predictive_checks.py --chain /path/to/chain.npy --observable spectrum
```

Full options with visualization:
```bash
python posterior_predictive_checks.py \
    --chain /path/to/chain.npy \
    --observable spectrum \
    --n-samples 1000 \
    --burnin 0.1 \
    --divide-factor 64 \
    --output results.npy \
    --plot \
    --plot-output ./plots \
    --seed 42
```

### Arguments

- `--chain`: Path to the chain file (required)
- `--observable`: Observable type, either 'spectrum' or 'tpcf' (default: 'spectrum')
- `--model-path`: Path to emulator model checkpoint (optional, uses defaults)
- `--n-samples`: Number of posterior samples to use (default: 1000)
- `--burnin`: Fraction of samples to discard as burnin (default: 0.1)
- `--divide-factor`: Factor to divide covariance matrix by (default: 64)
- `--output`: Output file to save results (optional)
- `--plot`: Generate visualization plots (flag)
- `--plot-output`: Directory to save plots (defaults to same directory as output)
- `--seed`: Random seed for reproducibility (default: 42)

### Output

The script prints:
- Posterior predictive p-value
- Mean and standard deviation of observed and replicated chi-squared statistics
- Chi-squared and chi-squared per degree of freedom at the best fit (posterior mean)

If an output file is specified, results are saved as a numpy dictionary containing:
- `pval`: Posterior predictive p-value
- `T_obs`: Array of observed chi-squared statistics
- `T_rep`: Array of replicated chi-squared statistics
- `chi2_at_best`: Chi-squared at posterior mean
- `dof`: Degrees of freedom
- `chi2_per_dof`: Chi-squared per DOF

### Visualization

When the `--plot` flag is used, the script generates three types of plots:

1. **Histogram Plot** (`*_histogram.png`): Overlaid histograms comparing the distributions of T_obs and T_rep with their means marked
2. **Scatter Plot** (`*_scatter.png`): Scatter plot of T_obs vs T_rep with a diagonal reference line
3. **Summary Plot** (`*_summary.png`): Comprehensive 4-panel figure showing:
   - Histogram comparison
   - Scatter plot
   - Cumulative distribution functions
   - Summary statistics table with fit assessment

The plots visually indicate whether the model provides a good fit to the data by comparing the observed chi-squared statistics (T_obs) with those from replicated data (T_rep).

### Examples

Run the `example_synthetic.py` script to see posterior predictive checks in action with synthetic data:

```bash
python example_synthetic.py
```

This will generate example plots demonstrating:
- A well-fitting model (good p-value)
- A misspecified model (poor p-value)

### Interpretation

- **P-value close to 0.5**: Good model fit
- **P-value < 0.05 or > 0.95**: Poor model fit, model may not adequately describe the data
- **T_rep and T_obs distributions should overlap**: If they're very different, the model is not capturing the data well

### Reference

For more information on posterior predictive checks, see:
https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html
