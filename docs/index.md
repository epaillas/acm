<!-- acm documentation master file, created by
sphinx-quickstart on Tue Jan  7 14:24:03 2025.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive. -->

# Alternative Clustering Methods


```{warning}
This documentation is still under construction. Please check back later for more information.
```

## What is the ACM Pipeline ?

The Alternative Clustering Methods (ACM) pipeline is a Python package that provides a set of tools for **beyond-2pt galaxy clustering** statistics for DESI.
It allows the training of several simulation-based theoretical models to emulate the galaxy clustering statistics (see [Pipeline](pipeline/overview)).
It contains several clustering methods that can be used to estimate the galaxy clustering statistics, such as densitysplit, bispectrum, Minkowski functionals and more (see [Statistics](pipeline/statistics)).

## Why are we doing this ?

The **non-Gaussianity** of the density field at small scales limits the amount of information we can access with the galaxy power spectrum. 
Alternative clustering statistics can access this information at a limited computational cost, but most of them **lack analytic theory models**.

## How are we doing this ?

We train **neural networks** on clustering statistics measured on thousands of **Abacus-based mocks**, learning about how the clustering responds to cosmology and the galaxy-halo connection.

```{toctree}
:maxdepth: 2

WIP
```

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
```

```{toctree}
:maxdepth: 2
:caption: The pipeline

pipeline/overview
pipeline/galaxy_halo
pipeline/statistics
pipeline/emulator
pipeline/inference
```

```{toctree}	
:hidden:
:maxdepth: 1
:caption: Code structure

code/overview
code/projects
code/data
code/io
```

```{toctree}
:maxdepth: 1
:caption: Statistics

statistics/densitysplit
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Tutorials and examples

notebooks/observables
notebooks/densitysplit
```

```{toctree}
:maxdepth: 1
:caption: Credits

credits/contributors
credits/dependencies
credits/citations
```

:::{note}
🔎 See an issue with the documentation? Feel free to open an issue or a pull request on the [GitHub repository](https://github.com/epaillas/acm) to help us improve the documentation ! 
:::

```{toctree}
:hidden:
:maxdepth: 2
:caption: API Reference

source/api
```
