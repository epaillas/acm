# Galaxy-Halo connection

## The HOD model

:::{admonition} TODO
:class: error
Explain the HOD model w/ references
:::

## Implementation

The `acm.hod` module is a wrapper around the [`AbacusHOD`](https://doi.org/10.1093/mnras/stab3355) class. It is used to populate galaxies in AbacusSummit Dark Matter catalogs.

It contains several classes, to generate different types of catalogs : `BoxHOD` to generate cubic boxes, `CutskyHOD` to generate cutsky catalogs, and `LightconeHOD` to generate lightcone catalogs.

:::{note}
A different dark matter catalog can be provided to `BoxHOD` and `CutskyHOD`, if necessary. The `acm.data.paths` module provides different Dark matter catalogs.
The default catalog used is tailored for LRGs (`LRG_Abacus_DM`), but a BGS catalog (`BGS_Abacus_DM`) is also available.
:::

To select a Latin HyperCube (LHC) of HOD parameters, the `acm.hod.parameters` module provides the `HODLatinHypercube` class:

```python
from acm.hod.parameters import HODLatinHypercube
from sunbird.inference.priors import Yuan23

ranges = Yuan23().ranges # priors for the HOD parameters

lhc = HODLatinHypercube(ranges)
params = lhc.sample(50_000)  # number of HOD variations

params = lhc.split_by_cosmo(cosmos=[0, 1])
lhc.save_params('./')
```

:::{admonition} TODO
:class: error
Maybe explain how the cutsky/lightcones are constructed ?
:::

## API

```{eval-rst}
.. automodule:: acm.hod.box
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: acm.hod.cutsky
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: acm.hod.lightcone
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: acm.hod.parameters
    :members:
    :undoc-members:
    :show-inheritance:
```