# Installation

## Requirements

Strict requirements are:
- `numpy`
- `scipy`
- `pandas` 
- `pyyaml`
- `matplotlib`
- `getdist`

```{note}
`Make` is required to compile some C files during the installation, so it should be available in your environment.
```

To run the estimators, you will also need:
- [`PolyBin3D`](https://github.com/oliverphilcox/PolyBin3D)

```{note}
PolyBin3D is installed through the `nersc` dependency.
```

To run the emulators, you will need:
- [`sunbird`](https://github.com/florpi/sunbird)

The package can be installed with the following optional dependencies:
- `nersc` for NERSC environment (containing `cosmodesi`)
- `cosmodesi` to install extra cosmodesi dependencies
- `docs` to install the documentation building dependencies
- Some estimators also have their own dependencies, which can be installed through the estimator name (e.g., `knn`, `mst`, `minkowski`)

```{tip}
Add the dependency names separated by commas, e.g. `pip install acm[nersc,sunbird,cosmodesi,estimator1,estimator2]` to install several dependencies at once.
```

## Pip Installation

### Install at NERSC

To install `acm` at NERSC, the `cosmodesi` environment already has some required packages installed. You can install the package with:

```bash
pip install acm[nersc,sunbird] @ git+https://github.com/epaillas/acm
```

### Install from PyPI

```{warning}
The package is not yet available on PyPI. Please install from source for now.
```

## For Developers

### Installing from Cloned Repository

If you want to install the package from source, you can clone the repository and install it with:

```bash
git clone https://github.com/epaillas/acm
cd acm
pip install -e .[nersc,sunbird]
```

```{note}
The `-e` flag is used to install the package in editable mode, which allows you to make changes to the code and have them reflected without reinstalling the package.
```

### Cython Building

The Cython files can be rebuilt (*only in editable mode*) with:

```bash
python setup.py build_ext --inplace 
python setup.py clean --all  # Clean up build files
```

```{note}
The Cython files are automatically built when installing the package, so you should not need to run this command unless you modify the Cython files.
```

### Building Documentation

The documentation is built using Sphinx. To build the documentation locally:

1. Install the documentation dependencies:
   ```bash
   pip install -e .[docs]
   ```

2. Navigate to the `docs` directory and build:
   ```bash
   cd docs
   make html
   ```

3. Open `docs/_build/html/index.html` in your browser to view the documentation.

```{note}
The `conf.py` file is designed to mock the packages that have C dependencies during the documentation build.
This allows the documentation to be built on ReadTheDocs without any problem.
```