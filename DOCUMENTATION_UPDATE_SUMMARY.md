# Documentation Update Summary

This document summarizes the updates made to the ACM documentation as part of the documentation update initiative.

## Changes Made

### 1. Updated Documentation Content

#### Installation (`docs/installation.md`)
- Expanded requirements section with detailed package lists
- Added comprehensive pip installation instructions
- Included developer setup instructions
- Added Cython building instructions
- Included documentation building guide

#### Pipeline Documentation
- **Statistics** (`docs/pipeline/statistics.md`): Added overview of available statistics, general workflow, and storage conventions
- **Inference** (`docs/pipeline/inference.md`): Added comprehensive guide to cosmological inference including methods, workflow, chain storage, and visualization

#### Credits
- **Citations** (`docs/credits/citations.md`): Added citation guidelines for ACM pipeline, statistics, methods, and dependencies with BibTeX entries
- **Dependencies** (`docs/credits/dependencies.md`): Expanded with categorized list of all dependencies including core packages, estimators, and specialized tools

### 2. Hosting Configuration

#### ReadTheDocs Configuration (`.readthedocs.yaml`)
Created comprehensive ReadTheDocs configuration including:
- Python 3.11 environment
- Build dependencies (numpy, cython, setuptools, wheel)
- Sphinx configuration
- Documentation requirements installation

#### Hosting Instructions (`HOSTING_INSTRUCTIONS.md`)
Created detailed guide covering:
- ReadTheDocs hosting (recommended approach)
  - Step-by-step setup instructions
  - Configuration guidelines
  - Custom domain setup
  - Version management
  - Troubleshooting
- GitHub Pages hosting (alternative)
  - GitHub Actions workflow
  - Manual deployment
  - Custom domain configuration
- Local documentation building
- Feature comparison between platforms
- Post-deployment checklist
- Maintenance guidelines

### 3. Documentation Infrastructure

#### Sphinx Configuration (`docs/conf.py`)
- Added additional packages to `autodoc_mock_imports` for better ReadTheDocs compatibility
- Includes: abacusutils, pycorr, Corrfunc, mistreeplus, numba, jax, jaxlib

#### Documentation README (`docs/README.md`)
Created quick reference guide including:
- Building instructions
- Structure overview
- Hosting reference
- Contribution guidelines
- Troubleshooting tips

#### Static Files (`docs/_static/`)
- Created directory structure to prevent build warnings

## Files Modified

- `docs/installation.md` - Major update
- `docs/pipeline/statistics.md` - Major update
- `docs/pipeline/inference.md` - Major update
- `docs/credits/citations.md` - Major update
- `docs/credits/dependencies.md` - Major update
- `docs/conf.py` - Minor update (mocked imports)

## Files Created

- `.readthedocs.yaml` - ReadTheDocs configuration
- `HOSTING_INSTRUCTIONS.md` - Comprehensive hosting guide
- `docs/README.md` - Documentation quick reference
- `docs/_static/.gitkeep` - Ensure _static directory exists
- `DOCUMENTATION_UPDATE_SUMMARY.md` - This file

## Documentation Status

### ✅ Complete Sections
- Installation guide
- Pipeline overview
- Galaxy-halo connection
- Statistics overview
- Emulator training
- Inference guide
- Density-split statistics
- MST statistics
- Code organization
- Project guidelines
- Data storage conventions
- I/O utilities
- Contributors list
- Dependencies list
- Citations guide
- API reference structure

### ⚠️ Sections with TODOs (from WIP.md)
These are intentional placeholders for future expansion:
- Detailed theory sections for HOD model
- Additional statistics beyond density-split and MST
- More tutorial notebooks
- Specific cutsky/lightcone construction details

## Next Steps for Hosting

### For ReadTheDocs (Recommended):
1. Visit https://readthedocs.org/ and sign in with GitHub
2. Import the project: `epaillas/acm` (or `cosmodesi/acm` if moved)
3. Configure project settings (default branch: main or cosmodesi)
4. Trigger initial build
5. Verify documentation at `https://acm.readthedocs.io/`
6. Update README.md with actual documentation URL
7. Add documentation badge to README

### For GitHub Pages (Alternative):
1. Add `.github/workflows/docs.yml` using the template in HOSTING_INSTRUCTIONS.md
2. Enable GitHub Pages in repository settings (branch: gh-pages)
3. Push changes to trigger workflow
4. Verify documentation at `https://epaillas.github.io/acm/` or `https://cosmodesi.github.io/acm/`

## Verification

Documentation builds successfully with:
```bash
cd docs
make html
```

All pages render correctly with only expected warnings:
- Cross-reference warnings (known issue with some internal links)
- Import warnings for unavailable modules (handled by mocking)

## Notes

- Documentation follows MyST Markdown format for better integration with Sphinx
- All external dependencies are properly mocked for ReadTheDocs compatibility
- Notebooks are set to not execute during build (`nb_execution_mode = 'off'`)
- The documentation is ready for immediate deployment to ReadTheDocs or GitHub Pages

## Repository Context

This update addresses issue related to updating the `docs` folder with current ACM documentation from the cosmodesi branch. The documentation is now:
- Comprehensive and well-organized
- Ready for ReadTheDocs hosting
- Includes clear instructions for both hosting options
- Follows best practices for scientific software documentation
