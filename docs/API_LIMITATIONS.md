# API Documentation Limitations

```{warning}
The automatic API documentation extraction (autodoc) is currently limited on ReadTheDocs due to dependencies that require compilation (Cython extensions) or are not installed in the documentation build environment.

As a result, the API reference pages show the module structure but may not include detailed docstrings for all classes and functions.
```

## Viewing Full API Documentation

To view the full API documentation with all docstrings and function signatures, you have two options:

### Option 1: Build Documentation Locally

1. Clone the repository and install the package with all dependencies:
   ```bash
   git clone https://github.com/epaillas/acm
   cd acm
   pip install -e ".[nersc,cosmodesi,docs]"
   ```

2. Build the documentation:
   ```bash
   cd docs
   make html
   ```

3. Open `docs/_build/html/index.html` in your browser

### Option 2: Read Docstrings in Source Code

You can view the docstrings directly in the source code on GitHub:
- [acm/hod](https://github.com/epaillas/acm/tree/main/acm/hod) - Galaxy-halo connection classes
- [acm/estimators](https://github.com/epaillas/acm/tree/main/acm/estimators) - Clustering statistics estimators
- [acm/model](https://github.com/epaillas/acm/tree/main/acm/model) - Emulator training and optimization
- [acm/observables](https://github.com/epaillas/acm/tree/main/acm/observables) - Observable handling classes

## For Contributors

If you're working on improving the API documentation on ReadTheDocs, the main challenges are:

1. **Cython Extensions**: The package includes C/Cython extensions that require compilation
2. **Heavy Dependencies**: Dependencies like `abacusnbody`, `torch`, `pytorch_lightning`, etc., are large and may have their own compilation requirements
3. **Import Chains**: Many modules import from modules that require these dependencies

Potential solutions being explored:
- Mock all heavy dependencies more comprehensively
- Create stub modules for documentation builds
- Generate API documentation during package build and commit static files
