# Code overview

The code is organised in a modular way, with different packages and projects that are connected together to form the pipeline.

The ACM pipeline is composed of four main components:

- The `sunbird` package
- The `acm` package
- The *bridges* between the packages and the projects
- The **projects** 

## The `sunbird` package

The `sunbird` package contains the more abstract components of the pipeline, such as the inference, the emulator, etc.

## The `acm` package

The `acm` package contains the code that is called in the projects, and that is used to run the pipeline:

:::{seealso}
The halo population is handled by [`acm.hod`](../pipeline/galaxy_halo.md#implementation) classes.
The statistics are computed trough the [`acm.estimators`](../pipeline/statistics.md) module.
The emulator, from `sunbird`, is optimized with `optuna` trough the [`acm.model`](../pipeline/emulator.md) module.
:::

:::{note}
The data storage is handled by the [`acm.data`](../code/io.md) module trough the [`acm.observables`](../code/projects.md#acmobservables-classes) classes if the conventions are followed, otherwise the methods to read the data need to be provided.
:::

## The bridges

Bridges are code snippets and components that are defined in the `acm` package, but have to be redefined or extended in the projects.
Examples of bridges are the `acm.observables` classes that have to be extended in the projects to define the statistics. In those classes, the user can define (or override) the data reading functions (see [`acm.data.io_tools`](../code/io)), the data paths (see [data](../code/data)), or the project *coordinates*.

```{seealso}
More information on the bridges can be found in the [Bridges](../code/projects) section.
```

## The projects

Projects are where the pipeline is actually run. They contain the scripts, the computations, etc.
Each project has its own structure, but some tools (the *bridges*) and some guidelines are provided to help the user to build a project that is compatible with other projects in the pipeline.
This allows for easy sharing of the data and models between users, and for easy testing of the pipeline. However, the user is free to design their projects as they see fit, as the code allows for an easy modularity and adaptability.

```{seealso}
Guidelines for project organisation can be found in the [Projects](../code/projects) section.
```

:::{admonition} Project dependant
:class: caution
These boxes are here to note the things that depend on the project and need to be adapted, especially if guidelines are followed.
:::