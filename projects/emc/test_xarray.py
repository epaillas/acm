import numpy as np
import xarray as xr
import xarray_jax as xj
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def transform_filters_to_slices(filters: Dict) -> Dict:
    """Transform a dictionary of filters into slices that select from min to max

    Example:
        filters = {'r': (10,100)} , will select the summary statistics for 10 < r < 100

    Args:
        filters (Dict): dictionary of filters.
    Returns:
        Dict: dictionary of filters with slices
    """
    slice_filters = filters.copy()
    for filter, (min, max) in filters.items():
        slice_filters[filter] = slice(min, max)
    return slice_filters

def convert_to_summary(
    data: np.array,
    dimensions: List[str],
    coords: Dict,
    select_filters: Optional[Dict] = None,
    slice_filters: Optional[Dict] = None,
) -> xr.DataArray:
    """Convert numpy array to DataArray summary to filter and select from

    Example:
        slice_filters = {'s': (0, 0.5),}, select_filters = {'multipoles': (0, 2),}
        will return the summary statistics for 0 < s < 0.5 and multipoles 0 and 2

    Args:
        data (np.array): numpy array containing data
        dimensions (List[str]): dimensions names (need to have the same ordering as in data array)
        coords (Dict): coordinates for each dimension
        select_filters (Dict, optional): filters to select values in coordinates. Defaults to None.
        slice_filters (Dict, optional): filters to slice values in coordinates. Defaults to None.

    Returns:
        xr.DataArray: data array summary
    """
    if select_filters:
        select_filters = {k: v for k, v in select_filters.items() if k in dimensions}
    if slice_filters:
        slice_filters = {k: v for k, v in slice_filters.items() if k in dimensions}
    summary = xr.DataArray(
        data,
        dims=dimensions,
        coords=coords,
    )
    if select_filters:
        summary = summary.sel(**select_filters)
    if slice_filters:
        slice_filters = transform_filters_to_slices(slice_filters)
        summary = summary.sel(**slice_filters)
    return summary

def lhc_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod'
    return Path(data_dir) / f'{statistic}.npy'

def summary_coords_lhc_y(statistic, sep):
    if statistic == 'tpcf':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(100)),
            'multipoles': [0, 2],
            's': sep,
        }


statistic = 'tpcf'

data_fn = lhc_fnames(statistic)
data = np.load(data_fn, allow_pickle=True).item()
sep = data['s']
coords = summary_coords_lhc_y(statistic, sep)
lhc_y = data['lhc_y']
select_filters = {'cosmo_idx': [0, 2], 'asd_idx': [1]}
slice_filters = {'s': (0, 150)}

# select_filters = {key: value for key, value in select_filters.items() if key in coords}
# slice_filters = {key: value for key, value in slice_filters.items() if key in coords}
dimensions = list(coords.keys())
lhc_y = lhc_y.reshape([len(coords[d]) for d in dimensions])
lhc_y = convert_to_summary(data=lhc_y, dimensions=dimensions, coords=coords,
                           select_filters=select_filters, slice_filters=slice_filters)

lhc_y = xj.from_xarray(lhc_y)

print(lhc_y)