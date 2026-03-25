# Scripts for the BGS measurements

> [!NOTE]
> At the moment, only box mocks are considered, cutsky, lightcone & systematics will be added later.

This project aims to infer the cosmological parameters from the BGS sample of DESI DR2 using different Alternative Clustering Methods (ACM).

> [!TIP]
> This pipeline was designed to be able to run different tracers/surveys with minimal changes within the code itself. The main changes are done through the configuration files and file reading methods if they differ from the ones implemented here.

## Pipeline

The pipeline is composed of the following steps:
1. Prepare the AbacusSummit Dark Matter simulations with a halo mass filter adapted to the BGS sample.
2. Sample HOD parameters from a LatinHyperCube (LHC) and populate the halos to create galaxy catalogs.
3. Measure the different ACM statistics on the galaxy catalogs, for `base` Abacus boxes.
4. Find the best-fit HOD index to the reference* for the `c000` cosmology.
5. Measure the different ACM statistics on the galaxy catalogs, for `small` Abacus boxes, at the best-fit HOD parameters for all phases.
6. Compress the measurements in the `acm.observables` format.

> [!TIP]
> Most of the scripts are using `argparse` to handle the input arguments. You can check the available arguments by running the script with the `-h` or `--help` flag.
> Those scripts also include a header docstring to explain their purpose and usage.


*\* The reference is the SecondGen mock BGS measurements*

### Prepare simulations
The AbacusSummit Dark Matter catalogs are precomputed in the `prepare_simulations` folder, to apply a halo mass filter adapted to the BGS sample.

Two scripts are available:
- `prepare_sim_bgs.py`: prepare the `base` Abacus boxes, using the `config.yaml` file.
- `prepare_small_bgs.py`: prepare the `small` Abacus boxes, using the `config_small.yaml` file.

> [!NOTE]
> Both scripts are runing one cosmology at a time, using `SLURM_ARRAY_TASK_ID` to select the cosmology from a provided bash array in the submission script.
> This array provides respectively the cosmology indice or the phase indice for the `base` and `small` boxes.

Two notebooks are also available in this folder:
- `phases_numbers.ipynb`: to get the list of available phases for the small boxes (and convert it to a bash array to put in `prepare_small_bgs.py`).
- `control_plots.ipynb`: to check the completeness of the halo catalogs files after applying the BGS halo mass filter.

> [!TIP]
> The ouptuts of these scripts are defined in the config files. A logger is also set up to keep track of the progress of the scripts, and defined within the script itself.

The LRG filter provided in `abacusnbody` did not allow the population of galaxies to reach the expected BGS number density. A new halo mass filter is therefore applied in these scripts.

> [!WARNING]
> This script is meant to run on the bgs_prep branch of https://github.com/SBouchard01/abacusutils installed locally.
> The call to `abacusnbody.hod.prepare_sim` only works if the call is made from the root folder of the repository (if the branch is not installed as a package).

### Measurements
HOD parameters are sampled for from a LatinHyperCube (LHC) in `sample_hods.py`.
Measurements are done in `measure_box.py`. This script handles base and small Abacus boxes.
Measurements are launched one cosmology (for `base` boxes) or one phase (for `small` boxes) at a time, using `SLURM_ARRAY_TASK_ID` to select the cosmology/phase from a provided bash array in the submission scripts for each case: `measure_base.sh` and `measure_small.sh`.

The measurements are compresed in the `acm.observables` format in `compress_files.py`.

The best HOD fit for each cosmology is found with `best_fit.py`.
The outliers are identified in `find_outliers.py`.

> [!NOTE]
> Both scripts use the `acm.observables` compression functions to read the measurements on `base` and `small` boxes.

> [!TIP]
> The `outliers.py` scripts can save the outlier indexes in a `.npy` file, that can be read to re-run only those outliers.
> In particular, this is useful for the `base` boxes, where each cosmology can have different outliers. 
> In practice, this can be run by hand using the `--hods` argument of `measure_box.py`, but for all 85 cosmologies it can be quite tedious.
> Instead, you can use the `--parameters_override` argument to provide a `.npy` file that will loop over the cosmologies, phases, seeds and HODs provided in the file.
> Not that this is only efficient for a low number of outliers: I recommend choosing the highest possible value of sigma in `find_outliers.py` to limit the number of outliers to the actual outliers, as some high-amplitude statistics can also be clipped as outliers.

> [!TIP]
> The `visual_tools.py` file contains some plotting functions to visualize the measurements and outliers.

### Training
The training and optimization of the emulator are done in `optimize_model.py`, using the `sunbird` package. The training is done for each statistic independently, keeping the first 6 cosmologies as a test set and the rest as a training set.

> [!NOTE]
> The `train_model.py` script is an older version of the training script for a non-optimized emulator, and is not used in the current pipeline. It is kept in the repository for reference, but it is not recommended to use it for training the emulator. 

### Inference
The inference is done in `inference_pocomc.py`, using the `pocomc` package to sample the posterior distribution of the cosmological parameters given the measurements and the emulator predictions.

The `bgs_mocks.py` and `utils.py` files contain some functions to handle the BGS mocks and utilities for the inference and output visualization.

> [!NOTE]
> The `jobs/` folder contains sub-folder with config files and submission scripts for the inference, for several cases (here different magnitude cuts). The config files are used to define the input parameters for the inference script. The submission scripts are used to submit the inference jobs on the cluster using `SLURM`. A similar folder is available for the measurements.

## Data

The data is stored in `/pscratch/sd/s/sbouchar/acm/bgs/` on NERSC. Let's detail the different subfolders:

### Parameters
Cosmological parameters are stored in an `AbacusSummit.csv` file in `parameters/cosmo_params/`.
HOD parameters sampled from a LatinHyperCube (LHC) are stored in `parameters/hod_params/`, splitted by cosmology as `Bouchard25_cxxx.csv`.
Combined cosmo + HOD parameters are stored in `parameters/cosmo+hod_params/`, splitted by cosmology as `AbacusSummit_cxxx.csv`.

### Measurements
The mocks measurements are stored in `measurements/`, with a subfolder per mock, using the following structure:
```
measurements/
├── logs/ # logs of the measurement scripts
├── base/
│   ├── c000_ph000/
│       ├── seed0/
│           ├── hod000/
│               ├── galaxies.fits # Not saved in the BGS case due to size constraints
│               ├── tpcf_los_x.npy
│               └──  ...
└── small/
```

> [!TIP]
> Each statistic that depends on the line-of-sight (los) has its own file, named accordingly, with the `measure_box.py` script looping over the los directions (x, y, z).

> [!NOTE]
> Due to the size of the BGS galaxy catalogs, we can't store them all on disk., We have to compute the statistics on the fly in the same script.


## TODO

- [x] Finish the readme
- [x] Recompute statistics with new boxes 
- [x] Update the file compression to use new raw file storage (unified)
- [x] Add plots scripts and notebooks
- [ ] Write the new statistic computation when needed again (+ control notebooks and small boxes !)