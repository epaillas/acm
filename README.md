# Alternative Clustering Methods (ACM)
`acm` is a cosmological analysis pipeline built for the DESI collaboration. It provides alternative methods to the standard two-point correlation function analysis, including higher-order statistics and machine learning techniques. The package is designed to be modular and extensible, allowing users to easily add new estimators and analysis methods.

> [!WARNING]
> The ACM project is a research project in constant evolution. The content of this repository may change frequently, and some features may be experimental or under development. Users are advised to check for updates regularly and to use the software with caution.

## Documentation
The complete documentation is available at [acm.readthedocs.io](https://acm.readthedocs.io).

## Installation

### Install with pip
To install `acm` at NERSC, the `cosmodesi` environment already has some required packages installed. You can install the package with:
```bash
pip install acm[sunbird] @ git+https://github.com/epaillas/acm
```

### Install from source
If you want to install the package from source, you can clone the repository and install it with:
```bash
git clone
pip install -e .[sunbird]
```

> [!NOTE]
> The `-e` flag is used to install the package in editable mode, which allows you to make changes to the code and have them reflected without reinstalling the package.

### Requirements
Strict requirements are:
- `numpy`
- `scipy`
- `pandas` 
- `pyyaml`
- `matplotlib`
- `getdist`

> [!WARNING]
> `Make` is required to compile some C files during the installation, so it should be available in your environment.

To run the emulators, you will need:
- [`sunbird`](https://github.com/florpi/sunbird)


The package can be installed with the following dependencies:
- `cosmodesi` to install extra cosmodesi dependencies (already included in the `cosmodesi` environment at NERSC)
- `docs` to install the documentation building dependencies
- some [estimators](acm/estimators/) also have their own dependencies, which can be installed trough the estimator name.
- the [estimator backends](acm/estimators/galaxy_clustering/backends/) also have their own dependencies, which must be installed trough the backend name.

> [!TIP]
> Add the dependency names separated by commas, e.g. `pip install acm[sunbird,cosmodesi,estimator1,estimator2]` to install several dependencies at once.


### Cython building
The Cython files can be rebuilt (*only in editable mode*) with:
```bash
python setup.py build_ext --inplace 
python setup.py clean --all # Clean up build files
```

> [!NOTE]
> The `cython` files are automatically built when installing the package, so you should not need to run this command unless you modify the Cython files.

## Examples
You can find notebooks examples in the `nb` folder.


## TODO (Github)
- [ ] Update the links to the doc  
