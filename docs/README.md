# Documentation README

This directory contains the Sphinx documentation for the ACM (Alternative Clustering Methods) pipeline.

## Quick Start

### Building Documentation Locally

1. Install dependencies:
   ```bash
   pip install -e ".[docs]"
   ```

2. Build the documentation:
   ```bash
   cd docs
   make html
   ```

3. View the documentation by opening `_build/html/index.html` in your browser.

### Other Build Commands

```bash
make clean     # Clean build files
make latexpdf  # Build PDF (requires LaTeX)
make epub      # Build EPUB
```

## Documentation Structure

- `index.md` - Main documentation landing page
- `installation.md` - Installation instructions
- `pipeline/` - Documentation for the ACM pipeline stages
  - `overview.md` - Pipeline overview
  - `galaxy_halo.md` - Galaxy-halo connection
  - `statistics.md` - Computing clustering statistics
  - `emulator.md` - Training emulators
  - `inference.md` - Cosmological inference
- `statistics/` - Detailed documentation for specific statistics
  - `densitysplit.md` - Density-split statistics
  - `mst.md` - Minimum spanning tree
- `code/` - Code organization and conventions
  - `overview.md` - Code structure
  - `projects.md` - Project organization guidelines
  - `data.md` - Data storage conventions
  - `io.md` - I/O utilities
- `credits/` - Contributors, dependencies, and citations
- `source/` - API reference documentation
- `notebooks/` - Tutorial notebooks
- `conf.py` - Sphinx configuration

## Hosting

For detailed instructions on hosting the documentation on ReadTheDocs or GitHub Pages, see the [HOSTING_INSTRUCTIONS.md](../HOSTING_INSTRUCTIONS.md) file in the repository root.

## Contributing to Documentation

When contributing to the documentation:

1. Use Markdown (`.md`) format with MyST syntax
2. Follow the existing structure and style
3. Test your changes by building locally before committing
4. Add your examples and tutorials to the appropriate sections
5. Update the API documentation if you add new modules

### MyST Markdown Features

The documentation uses [MyST](https://myst-parser.readthedocs.io/) which provides extended Markdown features:

- Admonitions: `:::{note}`, `:::{warning}`, `:::{tip}`, etc.
- Cross-references: `` [text](path/to/file.md) ``
- Code blocks with syntax highlighting
- Math equations with `$...$` or `$$...$$`
- Jupyter notebooks can be included directly

## Troubleshooting

### Common Build Issues

1. **Import errors during build**: Add missing packages to `autodoc_mock_imports` in `conf.py`
2. **Missing references**: Check that file paths are correct in cross-references
3. **Notebook execution errors**: Notebooks are set to not execute during build (`nb_execution_mode = 'off'`)

### Getting Help

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- Check the [MyST documentation](https://myst-parser.readthedocs.io/)
- Open an issue on the GitHub repository
