# Alternative Clustering Methods (ACM)
`acm` is a cosmological analysis pipeline built for the DESI collaboration

## Installation

### Install with pip
To install `acm` at NERSC, the `cosmodesi` environment already has some required packages installed. You can install the package with:
```bash
pip install acm[nersc,sunbird] @ git+https://github.com/epaillas/acm
```

### Install from source
If you want to install the package from source, you can clone the repository and install it with:
```bash
git clone
pip install -e .[nersc,sunbird]
```

> Note : `-e` flag is used to install the package in editable mode, which allows you to make changes to the code and have them reflected without reinstalling the package.

### Requirements
Strict requirements are:
- `numpy`
- `scipy`
- `pandas` 
- `pyyaml`
- `matplotlib`
- `getdist`

> Note : `Make` is required to compile some C files during the installation, so it should be available in your environment.

To run the estimators, you will also need:
- [`PolyBin3D`](https://github.com/oliverphilcox/PolyBin3D)
> Installed trough the `nersc` dependency.


To run the emulators, you will need:
- [`sunbird`](https://github.com/florpi/sunbird)


The package can be installed with the following dependencies:
- `nersc` for NERSC environment (containing `cosmodesi`)
- `cosmodesi` to install extra cosmodesi dependencies
- `docs` to install the documentation building dependencies
- some [estimators](acm/estimators) also have their own dependencies, which can be installed trough the estimator name.

> Add the dependency names separated by commas, e.g. `pip install acm[nersc,sunbird,cosmodesi,estimator1,estimator2]` to install several dependencies at once.


### Cython building
The Cython files can be rebuilt (*only in editable mode*) with:
```bash
python setup.py build_ext --inplace 
python setup.py clean --all # Clean up build files
```
> Note : The `cython` files are automatically built when installing the package, so you should not need to run this command unless you modify the Cython files.

## Documentation
The documentation is (*not yet*) available at [acm.readthedocs.io](https://acm.readthedocs.io).

## Examples
You can find notebooks examples in the `nb` folder.


## TODO (Github)
- [ ] Update the links to the doc  
