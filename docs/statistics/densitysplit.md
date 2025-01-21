# Density-Split statistics

:::{admonition} Question ?
:class: important
Do we keep the two statistics separate (CCF/ACF) or do we combine them ?
:::

:::{warning}
Right now, the `cosmodesi` environement on NERSC has a version of the `pyrecon` package that breaks the `densitysplit` code.

For now, the quick fix is to use the following command on the console before calling the `densitysplit` code:

```bash
module swap pyrecon/mpi pyrecon/main
```
:::

## Overview

:::{admonition} TODO
:class: error
Add a brief description of the `densitysplit` statistic. (see the article)
:::

## The `DensitySplit` class

### Initialization

The `DensitySplit` is imported from `acm.estimators.galaxy_clustering.densitysplit`.

:::{tip}
Some imports can fail at the first call of the `DensitySplit` class, if the `acm` package has not been installed with all the required dependencies (sometimes done if only one statistic is needed).
A quick fix is to import the `DensitySplit` class trough a `try`/`except` block, that will ignore the import error and continue the code execution.

```python
try:
    from acm.estimators.galaxy_clustering.densitysplit import DensitySplit
except ModuleNotFoundError: # On first import, some modules are not found but we still can import DensitySplit
    from acm.estimators.galaxy_clustering.densitysplit import DensitySplit
```
:::

The `DensitySplit` class is initialized with a `boxsize`, a `boxcenter` and a `cellsize` arguments.

```python
ds = DensitySplit(boxsize=1000, boxcenter=boxsize/2, cellsize=10) 
```

The data positions are assigned to the mesh with the `assign_data` method.

```python
ds.assign_data(data_positions, wrap=True) # wrap=True will wrap the data positions around the box assuming periodic boundary conditions
```

The `set_density_contrast` method calculates the density contrast of the data positions in a given smoothing radius.

```python
ds.set_density_contrast(smoothing_radius=10)
```

:::{tip}
When calling `set_density_contrast`, you can pass the argument `save_wisdom=True`, which will save a file on disk that will speed up FFT calculations next time you run the code with the same grid settings. This is highly recommended if you plan to run this on many mocks.
:::

The quantiles can be computed by calling the `set_quantiles` method. The `n_quantiles` argument sets the number of quantiles to compute, and the `query_method` argument sets the method to query the points on which the density is computed.

```python
ds.set_quantiles(n_quantiles=3, query_method='randoms')
quantiles = ds.quantiles
```

### Computing the statistic

#### In configuration space

The auto and cross-correlations can be computed for all the quantiles with the `quantile_correlation`  and `quantile_data_correlation` methods respectively.

```python
auto_correlation = ds.quantile_correlation(edges=(sedges, muedges), los='z')
cross_correlation = ds.quantile_data_correlation(data_positions, edges=(sedges, muedges), los='z')
```

:::{note}
The objects returned by the `quantile_correlation` and `quantile_data_correlation` methods are lists of `pycorr` objects, i.e. the correlation function estimators.
The multipoles can be obtained with : 

```python
# For quantile 0 auto-correlation
s, poles = auto_correlation[0](ells=(0, 2), return_sep=True)
```
:::



:::{seealso}
For more infomration on the `pycorr` objects, see the [pycorr documentation](https://pycorr.readthedocs.io/en/latest/).
:::


:::{seealso}
For more information on the `DensitySplit` class, see the [DensitySplit documentation](https://acm.readthedocs.io/en/latest/api/acm.estimators.galaxy_clustering.densitysplit.html).
:::