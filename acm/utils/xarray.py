from typing import Any

import xarray as xr


def _ensure_dataset(dataset: xr.Dataset | xr.DataArray) -> xr.Dataset:
    """Return a Dataset view for serialization helpers."""
    if isinstance(dataset, xr.DataArray):
        name = dataset.name or "data"
        return dataset.to_dataset(name=name)
    return dataset


def dataset_to_dict(dataset: xr.Dataset | xr.DataArray) -> Any:
    """Convert an xarray.Dataset to a dictionary with numpy arrays.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset to convert.

    Returns
    -------
    dict
        A dictionary containing the data from the dataset with numpy arrays.
    """
    dataset = _ensure_dataset(dataset)
    data_dict = {}
    for var_name in dataset.data_vars:
        data_dict[var_name] = {
            "data": dataset[var_name].values,
            "dims": dataset[var_name].dims,
            "coords": {
                coord: dataset[coord].values for coord in dataset[var_name].coords
            },
            "attrs": getattr(dataset[var_name], "attrs", None),
            "name": getattr(dataset[var_name], "name", None),
        }
    return data_dict


def dataset_from_dict(data_dict: dict[str, Any]) -> xr.Dataset:
    """Convert a dictionary with numpy arrays to an xarray.Dataset.

    Parameters
    ----------
    data_dict : dict
        A dictionary containing the data to convert. Each key should correspond to a variable name,
        and each value should be a dictionary with keys 'data', 'coords', 'attrs', and 'name'.

    Returns
    -------
    xarray.Dataset
        The resulting dataset.
    """
    data_vars = {}
    for var_name, var_info in data_dict.items():
        data_vars[var_name] = xr.DataArray(
            data=var_info["data"],
            dims=var_info.get("dims", None),
            coords=var_info["coords"],
            attrs=var_info.get("attrs", None),
            name=var_info.get("name", None),
        )
    return xr.Dataset(data_vars=data_vars)


def split_vars(*data_vars, **kwargs):
    """
    Splits variables of a DataSet in two: the selected values, and the non-selected values.

    Parameters
    ----------
    *data_vars : xr.DataArray
        The data variables to split.
    **kwargs : dict
        The selection criteria (e.g., dim=value) to pass to sel and drop_sel.

    Yields
    -------
    v_in : xr.DataArray
        The selected values.
    v_out : xr.DataArray
        The non-selected values.
    """
    for v in data_vars:
        v_in = v.sel(**kwargs)
        v_out = v.drop_sel(**kwargs)
        yield v_in, v_out
