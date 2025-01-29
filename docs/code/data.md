# Storing the data

Throughout the pipeline, we will need to store some data at specific points. To make sure that the data is stored in a consistent way, we will use some storage conventions, that allows for the use of i/o functions that are already implemented in the pipeline (see [i/o](../code/io)).

:::{important}
Before the emulator training, the statistics have to be computed independently for the data.
Each statistics having its own storage convention (see [Statistics](../pipeline/statistics/)), you're on your own ! 
Come back here once the statistics are computed 😉
:::

## Emulator input files

### LHC files

After the statistics are computed (see [Statistics](../pipeline/statistics)), the data is stored in a `LHC` file (for Latin HyperCube, which is the way the simulations are sampled). 

This file is a `<statistic>_lhc.npy` file containing a dictionary with the input parameters (`lhc_x`), the name of the parameters (`lhc_x_names`), 
the output statistics concatenated (`lhc_y`), the covariance *array* of the statistic (`cov_y`) and `bin_values` the bin values on which the statistics has been computed. (e.g. the separation bins for the correlation functions, the k bins for the power spectrum, etc.)

:::{note}
`lhc_x` is a 2D array with the shape `(n_samples, n_parameters)`, where `n_samples` is the number of samples in the LHC 
and `n_parameters` is the number of parameters of the model in the same order as `lhc_x_names`.

`lhc_y` is a 2D array with the shape `(n_samples, n_bins)`, where `n_samples` is the number of samples in the LHC and `n_bins` is the number of bins on which the statistics has been computed.
:::

:::{tip}
The covariance is stored as the covariance *array*, not the covariance *matrix*. This allows for the combination of the covariance of different statistics in a simple way, by concatenating the covariance arrays.
It also allows for a prefactor to be applied to the covariance, which can be useful.
The covariance matrix can be easily computed from the covariance array as follows:

```python
cov_array = lhc['cov_y']
cov_matrix = np.cov(cov_array, rowvar=False) # each line is a simulation, so rowvar=False
```

:::

## Emulator output files

### Model optimization

The optimization of the models is done with `optuna`. The summary of the optimization is stored in a `optuna/` directory, in a `<statistic>.pkl` file. 
This file contains the `optuna` study object, which can be used to retrieve the best parameters of the model.

:::{tip}
A logger can also be used to track the progress of the study (see [emulator](../pipeline/emulator)), in which case the logger is also stored in the `optuna/` directory as a `<statistic>.log` file.
:::


### Models

The models are stored in a `trained_models/` directory, with a subdirectory for each statistic. The model file is a `last.ckpt` PyTorch checkpoint file.

:::{tip}
Other models can be loaded with different directories and names, given the right path in the [`read_model`](../code/io) function.
:::


:::{note}
While `PyTorch` usually names the last model `last-v<version>.ckpt`, a symbolic link to the last version of the model, 
we recommend to store the final version of the model (after training) as `last.ckpt`.
This allows for the pipeline to always use the last version of the model, without having to worry about the file name in the code, and only change the statistics name.

:::{admonition} Question ?
:class: important
I'm not sure about this convention, to check.
:::

:::


### Errors

:::{admonition} Project dependant
:class: caution
The model error definition depends on the project. (For example, it can be defined as the mean of the difference between the model prediction and the data for the test set)
:::

The emulator error is saved in a `emulator_error/` directory, with a subdirectory for each statistic. 
The error is saved as a `<statistic>_emulator_error.npy` file, containing a dictionary
with `bin_values` the bins values on which the emulator predicts the statistic, `emulator_error` the error on the emulator and `emulator_cov_y` the covariance array of the emulator error.


## Inference chains

:::{admonition} Project dependant
:class: caution
The chain file nomenclature depends on the project. 
Feel free to name them as you wish, be creative ! ✨
:::

The inference chains are stored in a `chains/` directory. The chains are computed trough the inference wrappers in `sunbird.inference`, and stored as a `<statistic>_chain.npy` file with the `save_chain` method of the infernce base class. The file contains a dictionary with `samples` the chain values, `weights` the chain weights, `ranges` the varying parameter ranges, `names`the varying parameters names, `labels` the labels of the varying parameters in latex mode (with the `$` symbols) that will be used in the plots of the chains (see [Inference](../pipeline/inference)).

:::{seealso}
Depending on the inference method, the chains can also contain the `log_likelihood` and `log_prior` values, as well as the `log_posterior` values. See the documentation of the [inference](../pipeline/inference) method for more details.
:::

## Example file tree

Here is an example of the file tree that can is recommended after the training of the emulator for a `tpcf` statistic:

```bash
ACM_data/
├── input_data/
│   ├── tpcf_lhc.npy
├── trained_models/
│   ├── optuna/
│   │   ├── tpcf.pkl
│   │   ├── tpcf.log
│   ├── tpcf/
│   │   ├── last.ckpt
├── emulator_error/
│   ├── tpcf/
│   │   ├── tpcf_emulator_error.npy
├── chains/

```

:::{note}
Other paths can be used, as the pipeline is modular and the i/o functions can be adapted to the user's needs.
However, we recommend to use the same paths as the ones in the example, as the i/o functions are already implemented for these paths and it will make the use of the pipeline easier, 
as well as using the default path format (see [Default paths](#default-paths)).
:::


(default-paths)=
## Default paths 

The pipeline implements some default paths, that link to existing models, data and statistics on NERSC.
This allows for easy testing of the pipeline, as well as sharing of the data and models between users.

The default paths are stored in the `acm/data/paths.py` file, and can be imported in the pipeline with the `acm.data.paths` module.

A defalut path is a dictionary with the following structure:

```python
default_path = {
    # dir of the input data and covariance
    'lhc_dir': 'ACM_data/input_data/',
    'covariance_dir': 'ACM_data/input_data/',
    # dir of the errors
    'error_dir': 'ACM_data/emulator_error/',
    'emulator_covariance_dir': 'ACM_data/emulator_error/',
    'save_dir': 'ACM_data/emulator_error/',
    # dir of the trained models
    'study_dir': 'ACM_data/trained_models/optuna/',
    'model_dir': 'ACM_data/trained_models/',
    'checkpoint_name': 'last.ckpt',
    # dir of the inference
    'chain_dir': 'ACM_data/chains/',
}
```

:::{note}
Some paths are repeated for code reability when calling the dictionary, as the same directories can be used for different purposes.

For example, when calling the statistics in `read_lhc`, we use `default_path['lhc_dir']`, while when calling the covariance matrix in `read_covariance`, we use `default_path['covariance_dir']`.

While the same file is called, the readibility of the code is improved by using different keys for the same file. This also allows for the user to change the path of the covariance matrix without changing the path of the LHC file, if needed.
:::

Default paths are implemented for two tracers : LRG (Luminous Red Galaxies) and BGS (Bright Galaxy Survey).

:::{admonition} TODO
:class: error
Implement `acm.data.paths` in the pipeline :D

Then, provide the names of the dictionaries here
:::