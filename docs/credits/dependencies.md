# Dependencies

The `acm` package builds upon many excellent open-source projects. Below is a list of the key dependencies:

## Core Dependencies

### Required Packages

- **NumPy**: Numerical computing library
- **SciPy**: Scientific computing library
- **pandas**: Data analysis and manipulation
- **PyYAML**: YAML file parsing
- **matplotlib**: Plotting and visualization
- **getdist**: MCMC chain analysis and plotting

### Scientific Computing

- **sunbird**: Neural network emulation and simulation-based inference ([GitHub](https://github.com/florpi/sunbird))
- **PyTorch**: Deep learning framework (required by sunbird)

## Estimator Dependencies

### Clustering Measurements

- **pycorr**: Two-point correlation function measurements ([GitHub](https://github.com/cosmodesi/pycorr))
  - Mesh routines for particle assignment and FFT calculations are based on [`pyrecon`](https://github.com/cosmodesi/pyrecon)
- **pypower**: Power spectrum estimation ([GitHub](https://github.com/cosmodesi/pypower))
- **PolyBin3D**: Bispectrum estimation ([GitHub](https://github.com/oliverphilcox/PolyBin3D))
  - Developed by Oliver Philcox & Thomas Flöss
- **Corrfunc**: Fast correlation function estimation (optional, via pycorr)

### Specialized Estimators

- **mistreeplus**: Minimum spanning tree statistics (optional, for MST estimator)
- **kymatio**: Wavelet scattering transforms (optional, for wavelet estimator)
- **pyfnntw**: Fast nearest neighbor searches (optional, for kNN estimator)
- **fast-histogram**: Efficient histogram computation (optional, for kNN estimator)
- **numba**: JIT compilation (optional, for kNN estimator)

### Void Finding

- **Revolver**: Voxel void finder ([GitHub](https://github.com/seshnadathur/Revolver))
  - Developed by Seshadri Nadathur

## Galaxy-Halo Connection

- **abacusutils**: Utilities for working with AbacusSummit simulations
- **cosmoprimo**: Cosmological calculations ([GitHub](https://github.com/cosmodesi/cosmoprimo))
- **mockfactory**: Mock catalog generation ([GitHub](https://github.com/cosmodesi/mockfactory))

## Documentation

- **Sphinx**: Documentation generation
- **sphinx-book-theme**: Documentation theme
- **myst-nb**: Markdown and Jupyter notebook support
- **sphinx-design**: Design elements for Sphinx

## Installation

Different sets of dependencies can be installed using optional extras:

```bash
pip install acm[nersc]         # NERSC environment packages
pip install acm[cosmodesi]     # COSMODESI collaboration packages
pip install acm[docs]          # Documentation building
pip install acm[knn]           # k-NN estimator
pip install acm[mst]           # MST estimator
pip install acm[minkowski]     # Minkowski functionals
pip install acm[sunbird]       # Neural network emulation
```

```{seealso}
For installation instructions, see the [Installation](../installation) page.
```

## Acknowledgments

We are grateful to the developers and maintainers of all these packages, without which the `acm` pipeline would not be possible.

```{seealso}
For citation information, see the [Citations](citations) page.
```