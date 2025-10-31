# Documentation Hosting Instructions

This document provides detailed instructions for hosting the ACM documentation on ReadTheDocs or GitHub Pages.

## Table of Contents

1. [ReadTheDocs Hosting](#readthedocs-hosting-recommended)
2. [GitHub Pages Hosting](#github-pages-hosting-alternative)
3. [Building Documentation Locally](#building-documentation-locally)

---

## ReadTheDocs Hosting (Recommended)

ReadTheDocs is the recommended platform for hosting the ACM documentation as it provides automatic builds, versioning, and a clean interface.

### Prerequisites

- Admin access to the GitHub repository (or the repository must be moved to an organization with appropriate permissions)
- A ReadTheDocs account (free for open-source projects)

### Step 1: Connect GitHub Repository to ReadTheDocs

1. **Sign up/Log in to ReadTheDocs**
   - Go to https://readthedocs.org/
   - Sign in with your GitHub account or create an account

2. **Import the Project**
   - Click on your username in the top right
   - Select "My Projects"
   - Click "Import a Project"
   - If the repository is under your personal account, you should see `epaillas/acm` in the list
   - If the repository is moved to the `cosmodesi` organization, you'll need to grant ReadTheDocs access to that organization first:
     - Go to GitHub Settings → Applications → ReadTheDocs
     - Grant organization access to `cosmodesi`
   - Click the "+" button next to the repository to import it

3. **Configure the Project**
   - **Project Name**: `acm` or `alternative-clustering-methods`
   - **Repository URL**: `https://github.com/epaillas/acm` (or `https://github.com/cosmodesi/acm` if moved)
   - **Default Branch**: `main` or `cosmodesi` (whichever is the primary documentation branch)
   - **Default Version**: `latest`
   - Click "Next"

### Step 2: Configure Build Settings

The `.readthedocs.yaml` file in the repository root already contains the necessary configuration. However, you may want to verify the settings:

1. **Go to Project Settings**
   - In ReadTheDocs, go to your project page
   - Click "Admin" → "Advanced Settings"

2. **Verify Python Interpreter**
   - Should be set to Python 3.11 (as specified in `.readthedocs.yaml`)

3. **Enable PDF/EPUB Builds (Optional)**
   - Go to "Admin" → "Advanced Settings"
   - Enable "Build PDFs" and/or "Build EPUBs" if desired

### Step 3: Configure Custom Domain (Optional)

If you want to host at `acm.readthedocs.io` (default) or a custom domain:

1. **Default Domain**
   - Your documentation will be available at: `https://acm.readthedocs.io/`
   - Or if using the organization: `https://cosmodesi-acm.readthedocs.io/`

2. **Custom Domain Setup**
   - Go to "Admin" → "Domains"
   - Click "Add Domain"
   - Follow the instructions to configure DNS records

### Step 4: Configure Webhooks (Automatic)

ReadTheDocs automatically sets up GitHub webhooks when you import the project. This means:
- Every push to the repository triggers a documentation build
- Pull requests can show documentation previews
- Multiple versions can be maintained for different branches/tags

### Step 5: Build and Verify

1. **Trigger Initial Build**
   - Go to "Versions" in your ReadTheDocs project
   - Click "Build Version" for the latest version
   - Monitor the build logs for any errors

2. **Verify Documentation**
   - Once the build completes, visit your documentation URL
   - Check that all pages render correctly
   - Verify that cross-references and API documentation work

### Step 6: Configure Version Management

1. **Active Versions**
   - Go to "Versions" in your project admin
   - Mark which versions should be publicly available
   - Set the default version (usually "latest" or "stable")

2. **Version Strategy**
   - `latest`: Built from the default branch (main or cosmodesi)
   - `stable`: Built from the latest tagged release
   - Individual version tags (e.g., `v0.2.0`, `v0.3.0`)

### Troubleshooting ReadTheDocs

- **Build Failures**: Check the build logs in the ReadTheDocs admin panel
- **Import Errors**: Ensure all mocked imports are listed in `docs/conf.py` under `autodoc_mock_imports`
- **Missing Dependencies**: The documentation uses `docs/requirements.txt` to avoid building the full package (which requires C compilation). This file lists only the dependencies needed for documentation building.
- **Package Build Errors**: If you see errors related to Cython or numpy during the build, ensure that ReadTheDocs is using the `docs/requirements.txt` approach rather than installing the full package with `.[docs]`

---

## GitHub Pages Hosting (Alternative)

GitHub Pages is an alternative hosting option, though it requires more manual setup and doesn't provide the same features as ReadTheDocs.

### Option A: Using GitHub Actions (Recommended for GitHub Pages)

1. **Create GitHub Actions Workflow**

   Create `.github/workflows/docs.yml`:

   ```yaml
   name: Build and Deploy Documentation

   on:
     push:
       branches:
         - main
         - cosmodesi
     pull_request:

   jobs:
     build:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         
         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: '3.11'
         
         - name: Install dependencies
           run: |
             pip install "numpy<2.0.0" cython setuptools wheel
             pip install -e ".[docs]"
         
         - name: Build documentation
           run: |
             cd docs
             make html
         
         - name: Deploy to GitHub Pages
           if: github.event_name == 'push' && github.ref == 'refs/heads/main'
           uses: peaceiris/actions-gh-pages@v4
           with:
             github_token: ${{ secrets.GITHUB_TOKEN }}
             publish_dir: ./docs/_build/html
   ```

2. **Enable GitHub Pages**
   - Go to repository Settings → Pages
   - Source: Deploy from a branch
   - Branch: `gh-pages` (will be created by the action)
   - Folder: `/ (root)`
   - Click Save

3. **Configure Custom Domain (Optional)**
   - For personal account: `https://epaillas.github.io/acm`
   - For organization: `https://cosmodesi.github.io/acm`
   - Custom domain: Add `CNAME` file to `docs/_build/html/`

### Option B: Manual GitHub Pages Deployment

1. **Build Documentation Locally**
   ```bash
   cd docs
   make html
   ```

2. **Push to gh-pages Branch**
   ```bash
   cd _build/html
   git init
   git add .
   git commit -m "Deploy documentation"
   git branch -M gh-pages
   git remote add origin https://github.com/epaillas/acm.git
   git push -f origin gh-pages
   ```

3. **Enable GitHub Pages**
   - Go to repository Settings → Pages
   - Source: Deploy from a branch
   - Branch: `gh-pages`
   - Folder: `/ (root)`

### GitHub Pages URLs

- **Personal Repository**: `https://epaillas.github.io/acm`
- **Organization Repository**: `https://cosmodesi.github.io/acm`
- **Custom Domain**: Configure in repository settings

---

## Building Documentation Locally

For testing changes before deploying:

### Prerequisites

```bash
pip install -e ".[docs]"
```

### Build HTML Documentation

```bash
cd docs
make html
```

Open `docs/_build/html/index.html` in your browser.

### Build Other Formats

```bash
make latexpdf  # PDF via LaTeX
make epub      # EPUB format
make clean     # Clean build files
```

### Live Reload During Development

For automatic rebuilding during development:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

Then open `http://127.0.0.1:8000` in your browser.

---

## Comparison: ReadTheDocs vs GitHub Pages

| Feature | ReadTheDocs | GitHub Pages |
|---------|-------------|--------------|
| Setup Complexity | Easy (automatic) | Moderate (manual or CI) |
| Automatic Builds | ✅ Yes | ✅ Yes (with Actions) |
| Version Management | ✅ Built-in | ❌ Manual |
| Search Functionality | ✅ Built-in | ❌ Requires setup |
| PDF/EPUB Downloads | ✅ Built-in | ❌ Manual |
| Custom Domain | ✅ Easy | ✅ Easy |
| Build Environment Control | ✅ Full control | ✅ Full control |
| PR Previews | ✅ Yes | ⚠️ Requires setup |

**Recommendation**: Use ReadTheDocs for the best documentation hosting experience, especially for collaborative projects with multiple contributors and versions.

---

## Post-Deployment Steps

After setting up documentation hosting:

1. **Update README.md**
   - Replace the placeholder documentation URL with the actual URL
   - Current line: `The documentation is (*not yet*) available at [acm.readthedocs.io](https://acm.readthedocs.io).`
   - Update to: `The documentation is available at [acm.readthedocs.io](https://acm.readthedocs.io).`

2. **Add Documentation Badge**
   Add a badge to the README:
   ```markdown
   [![Documentation Status](https://readthedocs.org/projects/acm/badge/?version=latest)](https://acm.readthedocs.io/en/latest/?badge=latest)
   ```

3. **Update Repository Description**
   - Add the documentation URL to the GitHub repository description
   - Add topics/tags: `cosmology`, `clustering`, `statistics`, `desi`, `documentation`

4. **Announce to Team**
   - Inform contributors about the documentation URL
   - Encourage contributions to improve documentation

---

## Maintenance

### Updating Documentation

1. **Make changes** to `.md` files in the `docs/` directory
2. **Test locally** with `make html`
3. **Commit and push** changes
4. Documentation will automatically rebuild (ReadTheDocs) or via GitHub Actions

### Monitoring Builds

- **ReadTheDocs**: Check build status at `https://readthedocs.org/projects/acm/builds/`
- **GitHub Pages**: Check Actions tab for workflow status

### Troubleshooting

Common issues and solutions:

1. **Build timeouts**: Reduce dependencies or optimize build process
2. **Import errors**: Add packages to `autodoc_mock_imports` in `conf.py`
3. **Missing files**: Check that all required files are committed
4. **Version conflicts**: Pin specific package versions in `.readthedocs.yaml`

---

## Contact

For questions or issues with documentation hosting, contact the ACM Topical Group or open an issue on the GitHub repository.
