# EMC Inference Scripts

This directory contains scripts for performing inference on EMC (Emulator Mock Challenge) observables.

## Posterior Predictive Checks

The `posterior_predictive_checks.py` script implements posterior predictive checks to assess the goodness-of-fit from cosmological parameter inference.

### Usage

Basic usage:
```bash
python posterior_predictive_checks.py --chain /path/to/chain.npy --observable spectrum
```

Full options:
```bash
python posterior_predictive_checks.py \
    --chain /path/to/chain.npy \
    --observable spectrum \
    --n-samples 1000 \
    --burnin 0.1 \
    --divide-factor 64 \
    --output results.npy \
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

### Interpretation

- **P-value close to 0.5**: Good model fit
- **P-value < 0.05 or > 0.95**: Poor model fit, model may not adequately describe the data

### Reference

For more information on posterior predictive checks, see:
https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html
