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
> `Make` and `mpicc` are required to compile some C files during the installation, make sure they are available in your environment.

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

## Contributing
The project follows the [git-flow](https://nvie.com/posts/a-successful-git-branching-model/) branching model, with a `main` branch for stable releases and a `dev` branch for ongoing development. To contribute to the project, you can fork the repository, create a new branch for your feature or bug fix, and submit a pull request to the `dev` branch.

> [!TIP]
> When relevant, before publishing your branch and creating a pull request, locally `rebase` the branch on the latest `dev` to avoid merge conflicts and to keep the commit history clean. 
> 
> **Do not rebase any collaborative branch that has already been published, as this can cause issues for other collaborators.** 

To keep track of the branch objectives, we enforce the following syntax:
- `feature/<cool-new-feature>` for new features, to be merged into `dev` when ready
- `hotfix/<sorry-imessedup>` for urgent bug fixes, to be merged into `main` *and* `dev` when ready
- `docs/**` for documentation improvements
- `scripts/**` for the projects using `acm` that are hosted in the same repository, to be merged into `dev` when ready. Those scripts will be updated on `main` when the next release is made.

> [!NOTE]
> Personal branches exist on this repository, for specific collaborators that are not able to create forks. Those branches are not meant to be merged into `dev` and should be used for testing and development purposes only. They usually have the username of the collaborator as the branch name (e.g. `epaillas`). 

If you are not sure about the purpose of a branch, please ask the maintainers.

## Workflows

### Automated versions

The package version follows the [semantic versioning](https://semver.org/) scheme (`<major>.<minor>.<patch>-dev.<dev_version>`), and is automatically bumped on any closed PR to either `dev` or `main` branches. The version bumping is handled by the [bump2version](https://github.com/c4urself/bump2version) tool, and the workflow is defined in the `.github/workflows/ci-version.yml` file.

On any closed PR to `dev`, the version of the package will be automatically bumped to the next dev version. For example, if the current version is `0.1.0`, it will be bumped to `0.1.0-dev.1`.

On any closed PR to `main`, the version of the package will be automatically bumped to the next release version depending on the bump type specified in the PR tags (one of `bump:patch`, `bump:minor`, or `bump:major`). For example, if the current version is `0.1.0` and the PR is tagged with `bump:minor`, it will be bumped to `0.2.0`. A tag will also be created for the new version, and a release will be published on GitHub.

> [!NOTE]
> A workflow is also triggered on any PR to `main` to ensure exactly one of the bump type tags is present. If not, the workflow will fail and the PR cannot be merged until the issue is resolved.

After any PR merged to `main` with a version update, the `main` branch will be automatically merged back to `dev` to ensure the `dev` branch is up to date with the latest release. 

> [!WARNING]
> A merge to `main` without a version update will neither trigger a version bump nor a merge back to `dev`.

> [!TIP]
> To avoid conflicts, ensure your `dev` branch is up to date with the latest `main` branch before submitting a PR.
> 
> If you need to resolve a conflict on the version numbers, please update the version number with the current version number in the target branch; the version will be automatically bumped to the next version after the PR is merged.