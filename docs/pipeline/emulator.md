# Emulating the galaxy clustering

## The `sunbird` emulator

:::{admonition} TODO
:class: error
Explain the emulator
:::

## Training the emulator

Training the emulator is done trough the `sunbird.train` module.
The `acm.model.train` module contains the `TrainFCN` function, which trains the emulator on a given statistic, with a given set of hyperparameters.

:::{warning}
This function uses the `acm.data.io_tools` module to load the training and validation sets. This assumes that the data follows the standard storage format.
If it is not the case, you will need to adapt the `TrainFCN` function to your data storage format **locally**.
:::

:::{tip}
An example script on how to use the TrainFCN function is provided in the `acm.model.train.py` file.
:::

## Optmizing the emulator

The optimization of the emulator is done with `optuna`. The summary of the optimization is stored in a `optuna` directory, in a `<statistic>.pkl` file.
This file contains the `optuna` study object, which can be used to retrieve the best parameters of the model.

The `acm.model.optimize` module contains the `objective` function, which is used to optimize the hyperparameters of the model : 

```python
import optuna
from acm.model.optimize import objective
# TrainFCN_kwargs are the parameters of the TrainFCN function (not shown here)

study = optuna.create_study(study_name='tpcf')
optimize_objective = lambda trial: objective(trial, same_n_hidden=True, **TrainFCN_kwargs) 
study.optimize(optimize_objective, n_trials=100) # 100 is the number of trials to run
```

The best model path can then be retrieved with the `get_best_model` function : 

```python
from acm.model.optimize import get_best_model
best_model = get_best_model(statistic='tpcf', study_dir='path/to/study')
```

:::{note}
If provided with a `copy_to` argument, the best model will be copied to the `copy_to` directory. (useful to move it to the `models` directory 😉)
:::

:::{tip}
An example script on how to use the `objective` and `get_best_model` functions is provided in the `acm.model.optimize.py` file.
:::

## Computing the emulator error file

The emulator error file is computed trough the `acm.observables` class method `create_emulator_error`

:::{seealso}
For more information on the `acm.observables` class, see the [observables](../code/projects.md#acmobservables-classes) page.
:::


## API

```{eval-rst}
.. automodule:: acm.model.train
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: acm.model.optimize
    :members:
    :undoc-members:
    :show-inheritance:
```