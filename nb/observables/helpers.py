from pathlib import Path  # noqa: INP001

import numpy as np
import xarray as xr
from lsstypes import (
    Mesh2SpectrumPole,
    Mesh2SpectrumPoles,
    ObservableLeaf,
    ObservableTree,
)

parameters = ["h0", "om_m", "om_b", "sigma8", "n_s"] # Some example parameters


def get_spectrum(
    ells: tuple = (0, 2, 4),
    size: int = 40,
    seed: int = 42,
) -> Mesh2SpectrumPoles:
    """Make a random Mesh2SpectrumPoles object for testing."""
    rng = np.random.RandomState(seed=seed)
    poles = []
    for _ in ells:
        k_edges = np.linspace(0., 0.2, size + 1)
        k_edges = np.column_stack([k_edges[:-1], k_edges[1:]])
        k = k_edges.mean(axis=-1)
        poles.append(Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=rng.uniform(size=k.size)))
    spec = Mesh2SpectrumPoles(poles, ells=list(ells))
    # Add some random parameters to the spectrum
    spec.attrs.update({p: rng.uniform() for p in parameters})
    return spec

def _make_xarray() -> xr.Dataset:
    """Make an xarray Dataset with random spectra for testing."""
    y_specs = [get_spectrum(ells=(0, 2, 4), size=40, seed=i) for i in range(10)]
    y = xr.DataArray(
        data=np.array(y_specs).reshape((2, 5, 3, 40)),
        coords = {
            "i": np.arange(2),
            "j": np.arange(5),
            "ells": [0, 2, 4],
            "k": np.linspace(0., 0.2, 40),
        },
        attrs = {
            "sample": ["i", "j"],
            "features": ["ells", "k"],
        },
    )
    x = xr.DataArray(
        data=np.array([list(spec.attrs.values()) for spec in y_specs]).reshape((2, 5, len(parameters))),
        coords = {
            "i": np.arange(2),
            "j": np.arange(5),
            "parameters": parameters
        },
        attrs = {
            "sample": ["i", "j"],
            "features": ["parameters"],
        }
    )
    cy_specs = [get_spectrum(ells=(0, 2, 4), size=40, seed=i) for i in range(150)]
    covariance_y = xr.DataArray(
        data=np.array(cy_specs).reshape((150, 3, 40)),
        coords = {
            "ci": np.arange(150),
            "ells": [0, 2, 4],
            "k": np.linspace(0., 0.2, 40),
        },
        attrs = {
            "sample": ["ci"],
            "features": ["ells", "k"],
        }
    )
    x_test = x.sel(i=[1])
    y_test = y.sel(i=[1])
    x_test.attrs.update({"nan_dims": ["i"]})
    y_test.attrs.update({"nan_dims": ["i"]})
    return xr.Dataset({
        "x": x,
        "y": y,
        "covariance_y": covariance_y,
        "x_test": x_test,
        "y_test": y_test
    })

def _make_lsstypes() -> ObservableTree:
    """Make an ObservableTree with random spectra for testing."""
    y_specs = [get_spectrum(ells=(0, 2, 4), size=40, seed=i) for i in range(10)]
    x_vals = np.array([list(spec.attrs.values()) for spec in y_specs])
    labels = np.array([(i, j) for i in range(2) for j in range(5)])
    labels = {"i": labels[:, 0], "j": labels[:, 1]}
    y = ObservableTree(branches=y_specs, **labels)
    x = ObservableTree(
        branches=[ObservableTree(
            branches=[ObservableLeaf(value=x_vals[i, j]) for j in range(x_vals.shape[1])],
            parameters=parameters,
        ) for i in range(x_vals.shape[0])],
        **labels,
    )
    cy_specs = [get_spectrum(ells=(0, 2, 4), size=40, seed=i) for i in range(150)]
    covariance_y = ObservableTree(branches=cy_specs, ci=np.arange(150))
    x_test = x.select(i=[1])
    y_test = y.select(i=[1])
    return ObservableTree(
        branches=[x, y, covariance_y, x_test, y_test],
        name=["x", "y", "covariance_y", "x_test", "y_test"]
    )

def make_file(fn: str | Path, backend: str = "lsstypes"):  # noqa: ANN201
    """Make a test file with random data for the specified backend."""
    if backend == "xarray":
        cout = _make_xarray()
        cout.to_netcdf(fn)
    elif backend == "lsstypes":
        cout = _make_lsstypes()
        cout.write(fn)
    return cout
