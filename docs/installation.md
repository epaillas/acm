# Installation

```{warning}
The ACM project is a research project in constant evolution. The content of this repository may change frequently, and some features may be experimental or under development. Users are advised to check for updates regularly and to use the software with caution.
```

## Pip Installation
The recommended way to install the ACM package is via pip. You can install it directly from the GitHub repository using the following command:

```bash
pip install "acm[sunbird] @ git+https://github.com/epaillas/acm@cosmodesi"
```

```{note}
The current stable branch is `cosmodesi`. Other branches may be available for development or experimental features.
```

## Requirements
Strict requirements are:
- `numpy`
- `scipy`
- `pandas`
- `pyyaml`
- `matplotlib`
- `getdist`

```{warning}
`Make` and `MPICC` are required to compile some parts of the code, make sure those are installed on your system.
```

### Optional dependencies
The package can be installed with several optional dependencies.

```{tip}
If you are working on NERSC, the `cosmodesi` environment already has most of the required packages installed.
We choose to not include those packages as strict requirements to avoid conflicts with existing installations.
If you are *not* working on NERSC, you can add these dependencies trough the `cosmodesi` extra when installing via pip.
```

Some [estimators]() and their backends require additional packages to be installed. Those can be installed using the estimator or backend name as an extra. For example, to install the `wst` estimator with the `jaxpower` backend, you can use:

```bash
pip install "acm[wst,jaxpower] @ git+https://github.com/epaillas/acm@cosmodesi"
```

```{tip}
Check out the `pyproject.toml` file for a full list of available extras and their dependencies.
```

The [`sunbird`](https://github.com/florpi/sunbird) package for inference and samplers can be installed using the `sunbird` extra as shown in the pip installation command above.


## For Developers


### Installing from Cloned Repository
If you want to install the package from source, you can clone the repository and install it with:

```bash
git clone https://github.com/epaillas/acm
cd acm
git checkout cosmodesi
pip install -e .[sunbird]
```

```{note}
The -e flag is used to install the package in editable mode, which allows you to make changes to the code and have them reflected without reinstalling the package.
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

### Building documentation

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
3. Open the generated HTML files in your web browser to view the documentation.

<!-- The ``setup.py`` file is designed to ignore and mock the packages that have C dependencies if the environment name is ``READTHEDOCS``.
This *should* allow the documentation to be built on ReadTheDocs without any problem. -->