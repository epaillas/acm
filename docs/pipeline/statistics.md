# Computing the Statistics

The `acm` package provides several clustering statistics estimators that can be used to analyze galaxy clustering beyond the traditional 2-point correlation function. These statistics are designed to capture non-Gaussian information in the density field at small scales.

## Available Statistics

The `acm.estimators` module contains implementations of various clustering statistics:

### Galaxy Clustering Statistics

The main clustering statistics available in `acm.estimators.galaxy_clustering` include:

- **Density-Split Statistics**: Computes auto- and cross-correlations in different density quantiles
  - See the [Density-Split documentation](../statistics/densitysplit) for detailed usage
- **Two-Point Correlation Function (TPCF)**: Standard clustering measurement
- **Power Spectrum**: Fourier-space clustering measurement
- **Bispectrum**: Three-point statistics in Fourier space
- **Minkowski Functionals**: Geometric and topological measures of the density field
- **Minimum Spanning Tree (MST)**: Graph-based clustering statistics
  - See the [MST documentation](../statistics/mst) for details
- **k-Nearest Neighbors (kNN)**: Distance-based clustering measurements

### Computing Statistics

Each estimator has its own class and interface. The general workflow is:

1. Initialize the estimator with appropriate parameters
2. Assign data positions to the estimator
3. Compute the desired statistic
4. Save or analyze the results

```{tip}
Example notebooks showing how to compute statistics like density-split are available in the [Tutorials](../notebooks/densitysplit) section.
```

### Storage Conventions

After computing statistics, they should be stored following the conventions described in the [Data Storage](../code/data) section. This ensures compatibility with the emulator training pipeline.

```{seealso}
- For details on specific statistics, see the [Statistics](../statistics/densitysplit) section
- For information on how to integrate new statistics, see the [Projects](../code/projects) section
```

## API Reference

For detailed API documentation of all available estimators, see the [API Reference](../source/acm.estimators).
