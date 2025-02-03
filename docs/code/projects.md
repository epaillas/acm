# Projects

## Introduction

Projects are the recommended way to organize your work in `acm`. They are a way to group together all the components of a research project, such as the statistic computation, the emulator training, the cosmological inference, etc. This makes it easier to keep track of the different components of your project, and to share your work with others.

The projects are stored under the `projects` directory in the `acm` repository. Each project is a subdirectory of `projects`, and contains all the scripts, tests, notebooks, etc. specific to a dataset and/or an analysis.

:::{admonition} Project dependant
:class: caution
Each project defines its own way to handle the statistics and files, as well as the organization of the code. 
:::

## Bridges with the `acm` package

### `acm.observables` classes

The `acm.observables` classes are used to define a class for each statistic, that handles :
- the formatting of the statistic to a single file (see [Storing the data](../code/data))
- retrieving the statistic from the data files, the model and its errors
- applying filters to the data and model outputs

The base class defines an abstract class that should be inherited by the project classes.

The methods to implement are :
- `stat_name` : the name of the statistic
- `paths` : a dictionary with the paths to the data, model and errors (see [Storing the data](../code/data.md#default-paths))
- `summary_coords_dict` : a dictionary with the coordinates of the summary statistics (see [Coordinates](../code/io.md#coordinates))

:::{note}
Some methods are not implemented in the base class, but are not mandatory to implement if they are not needed. However, calling those methods will raise a `NotImplementedError` if they are not implemented in the project class.
Those methods are :
- `create_lhc` : to create the LHC file
- `create_covariance` : to create the covariance array that will be stored in the LHC file
- `create_emulator_error` : to create the emulator error file
- `create_emulator_covariance` : to create the emulator covariance file

Those methods depend on the file format of the statistics, and cannot be defined in the base class. However, if the creation of the files is not needed, the methods can be left unimplemented.
:::

Then, the class can be used to call the statistics, the parameters, the model, get a prediction, etc.

To combine statistics, a `CombinedObservables` class is available, that takes a list of `Observables` classes and combines them in a single class.
The methods are the same, with the filters applied to each statistic.

:::{seealso}
For examples of how to use these classes, see [examples](../notebooks/observables)
:::

### Global parameters

Some parameters are shared between the different components of the project, such as the paths, and the coordinates.
To avoid copying all this information trough the classes, we recommend creating a path and a coordinates dictionary in a `default.py` file in the project, that can be accessed by all the components of the project.

### Integration in the `acm` package

You can integrate the project to the `acm` package, by adding a subfolder in `acm/projects` with the name of the project, and adding the project classes names in the `__init__.py` file.
In this file, you can add the `default.py` file, and all the statistics classes created trough the `acm.observables` classes (we recommend one file per statistic).

This way, the statistic handling can be easily accessed trough the `acm` package, and the project can be shared with others.


## API

```{eval-rst}
.. automodule:: acm.observables.base
    :members:
    :undoc-members:
    :show-inheritance:
```

```{eval-rst}
.. automodule:: acm.observables.combined
    :members:
    :undoc-members:
    :show-inheritance:
```