# I/O functions

:::{warning}
The i/o functions defined here are designed to work with the data format described in [Storing the data](../code/data). If you are using a different data format, you will need to adapt the functions accordingly.
:::

## Reading the files

:::{note}
Most of these functions are designed to accept a list of statistics as input. They will concatenate the outputs of the different statistics in a single array.
This is to allow a retro-compatibility with the previous versions of the code, but we now recommend using the `acm.observable` classes to handle the data, and to combine them trough a `CombinedObservable` class.
:::

:::::{tab-set}

::::{tab-item} LHC files
To read the data from LHC files, you can use the `read_lhc` function. This function reads the data from the files and returns the parameters of the lhc, the statistics, the parameters names and the separation bins (if required) : 
```python
from acm.data.io_tools import read_lhc

bin_values, lhc_x, lhc_y, lhc_x_names = read_lhc(statistics=['tpcf'], data_dir='path/to/data', return_sep=True)
```
:::{tip}
The name of the file is defined in the `lhc_fname` function !
:::
::::

::::{tab-item} Data covariance
The covariance array of **a** statistic (only one here !) can be read with the `read_covariance_y` function : 

```python
from acm.data.io_tools import read_covariance_y

cov_y = read_covariance_y(statistic='tpcf', data_dir='path/to/data')
```

:::{note}
The covariance matrix can also be read with the `read_covariance` function, that has an extra `volume_factor` argument (usually 64, the ratio between the small and large box volumes of the Abacus simulations).
:::
::::

::::{tab-item} Models
The models can be obtained trough the `read_model` function : 

```python
from acm.data.io_tools import read_model

model = read_model(statistics=['tpcf'], model_fn='path/to/model.ckpt')
```

:::{note}
In this case, the model_fn is either the path to the model, or a dictionary with the statistics names as keys and the paths to the models as values.
:::
::::

::::{tab-item} Emulator covariance
The emulator covariance array and covariance can be read with the `read_emulator_covariance_y` and `read_emulator_covariance` functions respectively : 

```python
from acm.data.io_tools import read_emulator_covariance_y, read_emulator_covariance

emulator_cov_y = read_emulator_covariance_y(statistic='tpcf', data_dir='path/to/data')
emulator_cov = read_emulator_covariance(statistics=['tpcf'], data_dir='path/to/data')
```

The emulator error can be read with the `read_emulator_error` function : 

```python
from acm.data.io_tools import read_emulator_error

emulator_error = read_emulator_error(statistics=['tpcf'], data_dir='path/to/data')
```
::::

:::::


## Filtering the data

Most of the i/o functions accept two extra arguments : `select_filters` and `slice_filters`. These arguments are dictionnaries passed to [`xarray`](https://xarray.dev/) instances to filter the data.

- `select_filters` are used to select specific values of the parameters. For example, to select the `multipoles` parameter 0 and 2, you can use : 
```python
select_filters = {'multipoles': [0, 2],}
```
- `slice_filters` are used to slice the data. For example, to select the `bin_values` parameter between 0 and 0.5, you can use : 
```python
slice_filters = {'bin_values': (0, 0.5),}
```

:::{tip}
These filters are the initialization arguments of the `acm.observables` classes, that will filter the data trough the class methods.
:::

:::{note}
The `filter` function is the one used to filter all the arrays. It requires the filters, the coordinates of the provided array (see [Coordinates](#coordinates)), and if not in the coordinates keys, the final number of simulations (to allow the correct reshaping of the output)
:::

(coordinates)=
### Coordinates

:::{admonition} Project dependant
:class: caution
The coordinates are a project-dependant information ! A default dictionnary is provided in the `acm.data.default` module, to avoid some code crashes, but it is highly recommended to define the coordinates of each statistic in the `acm.observable` classes.
:::

The `summary_coords` functions is used to get the coordinates of the data, i.e. the parameters of the objects called.
It requires a dictionnary with several informations :

- `cosmo_idx` : the list of indexes of the cosmology parameters (from AbacusSummit) 
- `hod_number` : the number of HOD samples used **by cosmology**
- `param_number` : the number of parameters defining **each cosmology**
- `phase_number` : the number of phases (AbacusSummit small boxes) used for the covariance computation
- `statistics` : a dictionnary of the different parameters of the statistics, in an order that defines the shape of the data

Then, if provided, the bin_values are added to the statistics coordinates.

:::{note}
The `statistics` dictionnary is the most important one, as it defines the shape of the data. It is a dictionnary with the statistics names as keys, and the parameters of the statistics in a dictionary, with the names of the parameters and their values : 

```python
'statistics' = {
    'tpcf': {
        'multipoles': [0, 2],
    },
    'dsc_conf': {
        'statistics': ['quantile_data_correlation', 'quantile_correlation'],
        'quantiles': [0, 1, 3, 4],
        'multipoles': [0, 2],
    },
}
```
This dictionnary will define the shape of the data, with the `multipoles` and `bin_values` parameters for the `tpcf` statistic, and the `statistics`, `quantiles`, `multipoles` and `bin_values` parameters for the `dsc_conf` statistic.
Therefore, for 150 bins, the shape of the data will be `(2, 150)` for the `tpcf` statistic, and `(2, 4, 2, 150)` for the `dsc_conf` statistic.
:::

:::{important}
You need to provide the same parameters as those that were used to generate the data, otherwise the data will not be correctly reshaped ! 
The filtering process will also be based on these coordinates, so if you provide the wrong coordinates, you will not get the right data.
:::

Depending on the required data, the `summary_coords` function will return only certain the coordinates : 
- `lhc_x` will return the `cosmo_ixd`, `hod_idx`, and `param_idx` coordinates
- `lhc_y` will return the `cosmo_ixd`, `hod_idx`, and `statistics` coordinates 
- `smallbox` will return the `phase_idx` and `statistics` coordinates
- `emulator_error` will only return the `statistics` coordinates

:::{note}
To go from `_number` to `_idx`, we just use `list(range(_number))` to get a list of indexes with the right length.
:::

:::{tip}
These coordinates can also be used to reshape the data into an `xarray` dataset, with the right summary shape : 

```python
from sunbird.data.data_utils import convert_to_summary
dimensions = list(coords.keys())
y = y.reshape([len(coords[d]) for d in dimensions])
y = convert_to_summary(data=y, dimensions=dimensions, coords=coords)
```	
:::