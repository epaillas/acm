import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lsstypes
import numpy as np
import pandas as pd
import xarray
from pycorr import TwoPointEstimator

from acm.utils.xarray import dataset_to_dict, split_vars

logger = logging.getLogger(__name__)


# %% Readers and processors for different file formats
def lsstypes_reader(files: list[Path]) -> Any:
    """
    Read and average a list of lsstypes files.

    Parameters
    ----------
    files : list[Path]
        List of file paths to read using lsstypes.

    Returns
    -------
    Any
        Averaged lsstypes object across all input files.
    """
    loaded = [lsstypes.read(f) for f in files]
    data = lsstypes.mean(loaded)
    return data


def lsstypes_postprocess(
    data: list[Any],
    last_dim: str,
    select: dict,
    get: dict,
    rebin: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Apply selection, optional rebinning, and coordinate extraction to a list of lsstypes objects.

    Filters all elements in ``data`` to match the shape and coordinates of the
    first element after applying the selection and get operations, then stacks
    the results into a numpy array.

    Parameters
    ----------
    data : list[Any]
        List of lsstypes objects to process.
    last_dim : str
        Name of the last (feature) dimension, used to extract coordinates
        from the processed data and append to the returned dimension dict.
    select : dict
        Keyword arguments passed to ``.select()`` to slice the data along
        one or more dimensions (e.g. ``{'k': (0.01, 0.7)}``).
    get : dict
        Keyword arguments passed to ``.get()`` to extract specific entries
        along one or more dimensions (e.g. ``{'ells': [0, 2]}``).
    rebin : dict, optional
        If provided, keyword arguments passed to a first ``.select()`` call
        before the main selection, used to downsample the data
        (e.g. ``{'k': slice(0, None, 13)}``).

    Returns
    -------
    data_out : np.ndarray
        Array of shape ``(len(data), ...)`` containing the processed and
        matched data for each input object.
    coords : dict
        Dictionary mapping dimension names to their coordinate arrays,
        combining the ``get`` axes and the ``last_dim`` coordinates.
    """
    if rebin is not None:
        d0 = data[0].select(**rebin).select(**select).get(**get)
    else:
        d0 = data[0].select(**select).get(**get)

    last_dim_dict = {last_dim: d0.flatten(level=None)[0].coords(last_dim)}
    coords = {**get, **last_dim_dict}

    def lsstypes_match(d):
        return d.match(d0)

    data_out = np.asarray(
        [lsstypes_match(d) for d in data]
    )  # TODO: use lsstypes.match when available ?

    tmp_coords = {"data": np.arange(len(data)), **coords}
    data_out = reshape_to_coords(data_out, tmp_coords)

    return data_out, coords


def pycorr_reader(files: list[Path]) -> TwoPointEstimator:
    """
    Read and sum a list of pycorr TwoPointEstimator files.

    Parameters
    ----------
    files : list[Path]
        List of file paths to load as TwoPointEstimator objects.

    Returns
    -------
    TwoPointEstimator
        Summed TwoPointEstimator object across all input files.
    """
    loaded = [TwoPointEstimator.load(f) for f in files]
    data = sum(loaded)
    return data


def pycorr_postprocess(
    data: list[TwoPointEstimator],
    ells: list[int],
    rebin: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Apply optional rebinning and extract specified multipoles from a list of TwoPointEstimator objects.

    For each object in ``data``, selects the specified multipoles with rebinning,
    and stacks the results into a numpy array. Also extracts the separation coordinates
    from the first object to include in the returned coordinates dict.

    Parameters
    ----------
    data : list[TwoPointEstimator]
        List of TwoPointEstimator objects to process.
    ells : list[int]
        List of multipole orders to extract (e.g. ``[0, 2]``).
    rebin : int, optional
        If provided, step size for rebinning the data along the separation axis
        (e.g. ``3`` to take every 3rd separation bin).
        Defaults to ``None`` (no rebinning).

    Returns
    -------
    data_out : np.ndarray
        Array of shape ``(len(data), len(ells), ...)`` containing the extracted multipole data for each input object.
    coords : dict
        Dictionary mapping dimension names to their coordinate arrays, including:
        - ``'ells'``: the list of multipole orders extracted.
        - ``'s'``: the separation coordinates corresponding to the extracted multipoles, taken from the first object in ``data``.
    """
    rebin = rebin or 1

    s, _ = data[0][::rebin](ells=ells, return_sep=True)  # ty:ignore[not-subscriptable]
    coords = {"ells": ells, "s": s}

    data_out = []
    for d in data:
        poles = d[::rebin](ells=ells, return_sep=False)  # ty:ignore[not-subscriptable]
        data_out.append(poles)
    data_out = np.stack(data_out)

    tmp_coords = {"data": np.arange(len(data)), **coords}
    data_out = reshape_to_coords(data_out, tmp_coords)

    return data_out, coords


def ds_reader(files: list[Path]) -> list[TwoPointEstimator]:
    """
    Read a list of numpy files containing lists of TwoPointEstimator objects.

    Useful for the densitysplit estimator storage.

    Parameters
    ----------
    files : list[Path]
        List of file paths to load as with numpy, containing a list of TwoPointEstimator objects.

    Returns
    -------
    list[TwoPointEstimator]
        List of loaded TwoPointEstimator objects from the input files.
    """
    loaded = [np.load(f, allow_pickle=True) for f in files]
    data = sum(loaded)
    return data.tolist()


def ds_postprocess(
    data: list[list[TwoPointEstimator]],
    quantiles: list[int],
    ells: list[int],
    rebin: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Apply optional rebinning and extract specified quantiles and multipoles from a list of lists of TwoPointEstimator objects.

    For each object in ``data``, selects the specified quantiles and multipoles with rebinning,
    and stacks the results into a numpy array. Also extracts the separation coordinates
    from the first object to include in the returned coordinates dict.

    Parameters
    ----------
    data : list[list[TwoPointEstimator]]
        List of lists of TwoPointEstimator objects to process.
    quantiles : list[int]
        List of quantile indices to extract.
    ells : list[int]
        List of multipole orders to extract (e.g. ``[0, 2]``).
    rebin : int, optional
        If provided, step size for rebinning the data along the separation axis
        (e.g. ``3`` to take every 3rd separation bin).
        Defaults to ``None`` (no rebinning).

    Returns
    -------
    data_out : np.ndarray
        Array of shape ``(len(data), len(ells), ...)`` containing the extracted multipole data for each input object.
    coords : dict
        Dictionary mapping dimension names to their coordinate arrays, including:
        - ``'ells'``: the list of multipole orders extracted.
        - ``'s'``: the separation coordinates corresponding to the extracted multipoles, taken from the first object in ``data``.
    """
    rebin = rebin or 1

    s, _ = data[0][0][::rebin](ells=ells, return_sep=True)  # ty:ignore[not-subscriptable]
    coords = {"quantiles": quantiles, "ells": ells, "s": s}

    data_out = []
    for d in data:
        for q in quantiles:
            poles = d[q][::rebin](ells=ells, return_sep=False)  # ty:ignore[not-subscriptable]
            data_out.append(poles)
    data_out = np.stack(data_out)

    tmp_coords = {"data": np.arange(len(data)), **coords}
    data_out = reshape_to_coords(data_out, tmp_coords)

    return data_out, coords


# %% Utility functions for index handling and re-indexing
def reshape_to_coords(arr: np.ndarray, coords: dict) -> np.ndarray:
    """
    Reshape an array to match the lengths of the provided dimension coordinate.

    Parameters
    ----------
    arr : np.ndarray
        Array to reshape.
    coords : dict
        Dictionary mapping dimension names to their coordinate, used to determine the target shape.

    Returns
    -------
    np.ndarray
        Reshaped array with shape matching the lengths of the coordinates.
    """
    shape = [len(v) for v in coords.values()]
    if arr.size != np.prod(shape):
        raise ValueError(
            f"Cannot reshape array of size {arr.size} to shape {shape} based on provided coordinates."
        )
    return arr.reshape(shape)


def cast_coords(d: dict) -> dict:
    """
    Cast dictionary values to float or int where possible, leaving others unchanged.

    Tries float first to preserve precision, then int if the float values are
    all whole numbers, otherwise keeps the original string values.

    Parameters
    ----------
    d : dict
        Dictionary whose values are to be cast.

    Returns
    -------
    dict
        Dictionary with values cast to int or float where possible.
    """

    def cast(arr: np.ndarray) -> np.ndarray:
        try:
            float_arr = np.asarray(arr, dtype=float)
            if np.all(float_arr == float_arr.astype(int)) and np.issubdtype(
                float_arr.dtype, np.floating
            ):
                float_arr = float_arr.astype(int)
            return float_arr
        except (ValueError, TypeError):
            return np.asarray(arr)

    return {k: cast(v) for k, v in d.items()}


def reindex_samples(
    index_arrays: dict[str, list],
    reindex: list[str],
    group_by: list[str] | None = None,
) -> dict[str, list]:
    """
    Re-index selected index arrays from 0 to n within each sub-group.

    Parameters
    ----------
    index_arrays : dict[str, list]
        Dictionary mapping index names to their raw string values,
        as collected during file grouping (e.g. ``{'cosmo_idx': ['000', '000', ...], 'hod_idx': ['006', '008', ...]}``)
    reindex : list[str]
        Names of indexes to re-index. Defaults to all indexes.
    group_by : list[str]
        Index names defining the sub-groups within which re-indexing is
        performed independently (e.g. ``['cosmo_idx']``).
        If None, re-indexing is global across all samples.

    Returns
    -------
    dict[str, list]
        Updated index_arrays with re-indexed values as integers for the
        specified indexes.

    Examples
    --------
    >>> index_arrays = {
    ...     'cosmo_idx': ['000', '000', '000', '001', '001', '001'],
    ...     'hod_idx':   ['006', '008', '010', '008', '010', '014'],
    ... }
    >>> reindex_samples(index_arrays, reindex=['hod_idx'], group_by=['cosmo_idx'])
    {
        'cosmo_idx': ['000', '000', '000', '001', '001', '001'],
        'hod_idx':   [0, 1, 2, 0, 1, 2],
    }
    """
    n = len(
        next(iter(index_arrays.values()))
    )  # Assume all index arrays have the same length
    result = {k: list(v) for k, v in index_arrays.items()}
    if group_by is None:
        group_keys = [()] * n  # Single global group
    else:
        group_keys = [tuple(index_arrays[g][i] for g in group_by) for i in range(n)]

    for idx_name in reindex:
        if idx_name not in index_arrays:
            raise ValueError(f"Index '{idx_name}' not found in index_arrays.")

        vals = index_arrays[idx_name]  # Original raw values for this index

        # Build a (group_key, raw_value) -> new_index map in one pass
        local_maps: dict[
            tuple, dict
        ] = {}  # Map reindexing the values to integers within each group key
        for gk, raw in zip(group_keys, vals):
            local_map = local_maps.setdefault(gk, {})
            local_map.setdefault(
                raw, len(local_map)
            )  # Assign a new index if this raw value hasn't been seen in this group key

        result[idx_name] = [local_maps[gk][raw] for gk, raw in zip(group_keys, vals)]

    return result


def split_test_set(
    ds: xarray.Dataset,
    filters: dict,
    to_split: list[str] | None = None,
) -> xarray.Dataset:
    """
    Split DataArrays into test/train sets based on filters and merge into a single Dataset.

    Based on split_vars. "in" matches the filters and is suffixed with "_test",
    "out" is the complementary subset suffixed with "_train".

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset containing the variables to split.
        Must contain all variables listed in to_split (or defaulting to "x" and "y").
    filters : dict
        Dictionary of dimension names and values to filter the DataArrays
        (see "in" variables in split_vars).
    to_split : list[str], optional
        List of variable names in the dataset to apply the split to. If None,
        defaults to ``x`` and ``y`` if they exist in the dataset.

    Returns
    -------
    xarray.Dataset
        Split dataset with filtered variables. New data variables have a ``nan_dims``
        attribute listing the dimensions that were filtered out and filled with NaNs
        in the complementary variable.
    """
    to_split = to_split or ["x", "y"]
    for v in to_split:
        if v not in ds:
            raise ValueError(
                f"Variable '{v}' not found in dataset. Available variables: {list(ds.data_vars)}"
            )

    logger.debug(f"Splitting variables: {to_split} with filters: {filters}")

    data_vars = [ds[v] for v in to_split]
    for v_in, v_out in split_vars(*data_vars, **filters):
        v_in.name = v_in.name + "_test"
        v_out.name = v_out.name + "_train"

        # Mark filtered dimensions that will be filled with NaNs
        v_in.attrs["nan_dims"] = list(filters)
        v_out.attrs["nan_dims"] = list(filters)

        ds = xarray.merge([ds, v_in, v_out], join="outer")
    return ds


# %% Main function to compress mocks into an xarray DataArray
def collect_mocks(
    root_dir: str | Path,
    glob_pattern: str,
    ignore_index: list[str] | None = None,
) -> tuple[dict[tuple, list[Path]], dict[str, list]]:
    """
    Collect files matching a glob pattern and group them by index combinations.

    Parameters
    ----------
    root_dir : str or Path
        Root directory from which the glob search is performed.
    glob_pattern : str
        Glob pattern relative to ``root_dir`` with named index placeholders
        in curly braces (e.g. ``'c{cosmo_idx}_ph{phase_idx}/hod{hod_idx}/power_spectrum_los_{los}.h5'``).
        Each ``{name}`` placeholder is extracted as an index dimension unless
        listed in ``ignore_index``.
    ignore_index : list[str], optional
        List of placeholder names to exclude from the output dimensions.
        Files differing only in these indices are grouped and averaged together.
        Defaults to ``[]``.

    Returns
    -------
    groups : dict[tuple, list[Path]]
        Dictionary mapping unique index combinations (as tuples of (index_name, value))
        to lists of file paths that share those index values.
    index_arrays : dict[str, list]
        Dictionary mapping tracked index names to lists of their raw string values,
        collected in the same order as the files are read for later use in coordinate construction.
    """
    root_dir = Path(root_dir)  # Ensure Path behavior
    ignore_index = ignore_index or []  # Ensure list behavior

    # Identify indexes from glob pattern
    indexes = re.findall(r"{(.+?)}", glob_pattern)
    track_indexes = [idx for idx in indexes if idx not in ignore_index]
    logger.debug(
        f"Identified indexes: {indexes}, tracking indexes: {track_indexes}, ignored indexes: {ignore_index}"
    )

    regex_pattern = re.escape(
        str(root_dir / glob_pattern)
    )  # Escape special characters for regex
    for idx in indexes:
        placeholder = re.escape(f"{{{idx}}}")
        if idx in ignore_index:
            # Non-capturing group for ignored indexes
            regex_pattern = regex_pattern.replace(placeholder, r"[^/]+", 1)
        else:
            # Named capture group for known indexes
            regex_pattern = regex_pattern.replace(placeholder, f"(?P<{idx}>[^/]+)", 1)

    # Replace remaining wildcards (the glob * not from named indexes) with a non-capturing group
    regex_pattern = regex_pattern.replace(r"\*", r"[^/]+")
    regex_pattern = re.compile(regex_pattern)

    logger.debug(f"Constructed regex pattern: {regex_pattern.pattern}")

    # Build glob pattern for file discovery (replace all {idx} with *)
    flat_glob = glob_pattern
    for idx in indexes:
        flat_glob = flat_glob.replace(f"{{{idx}}}", "*")

    files = sorted(root_dir.glob(flat_glob))

    # Group files sharing the same index combination
    def get_index_key(path) -> tuple | None:
        m = regex_pattern.match(str(path))
        if m:
            return tuple((idx, m.group(idx)) for idx in track_indexes)
        return None

    groups = {}
    for f in files:
        key = get_index_key(f)
        if key is not None:
            groups.setdefault(key, []).append(f)

    index_arrays = {idx: [] for idx in track_indexes}
    for key in groups:
        for idx, value in key:
            index_arrays[idx].append(value)

    logger.debug(
        f"Found {len(files)} files grouped into {len(groups)} unique index combinations."
    )

    return groups, index_arrays


def compress_mocks(
    groups: dict[tuple, list[Path]],
    index_arrays: dict[str, list],
    reindex: list[str] | None = None,
    reindex_group_by: list[str] | None = None,
    drop_singleton_dims: bool = True,
    reader: Callable = lsstypes_reader,
    postprocess: Callable = lsstypes_postprocess,
    **kwargs,
) -> xarray.DataArray:
    """
    Average, processes, and package mock measurement files into an xarray DataArray.

    Averages each file group using ``reader``, applies ``postprocess`` to extract features
    and coordinates, and assembles the results into a labelled multi-dimensional array.

    Parameters
    ----------
    groups : dict[tuple, list[Path]]
        Dictionary mapping unique index combinations (as tuples of (index_name, value))
        to lists of file paths that share those index values, as returned by ``collect_mocks``.
    index_arrays : dict[str, list]
        Dictionary mapping tracked index names to lists of their raw string values,
        collected in the same order as the files are read for later use in coordinate construction.
    reindex : list[str], optional
        Names of indexes to re-map to contiguous integers starting from 0.
        Useful when an index (e.g. ``hod_idx``) does not share the same values
        across all sub-groups. Defaults to ``None`` (no re-indexing).
    reindex_group_by : list[str], optional
        Index names defining the sub-groups within which re-indexing is performed
        independently (e.g. ``['cosmo_idx']``). Must be provided if ``reindex``
        is set. Defaults to ``None``.
    drop_singleton_dims : bool, optional
        If ``True``, squeeze out any dimensions of length 1 from the output
        DataArray. Defaults to ``True``.
    reader : callable, optional
        Function with signature ``(files: list[Path]) -> Any`` used to load
        and combine a group of files. Defaults to :func:`lsstypes_reader`.
    postprocess : callable, optional
        Function with signature ``(data: list[Any], **kwargs) -> tuple[np.ndarray, dict]``
        used to select, rebin, and extract coordinates from the loaded data.
        Defaults to :func:`lsstypes_postprocess`.
    **kwargs
        Additional keyword arguments forwarded to ``postprocess`` (e.g. ``select``,
        ``get``, ``rebin``, ``last_dim``).

    Returns
    -------
    xarray.DataArray
        Multi-dimensional labelled array with:

        - **sample dimensions** — one axis per tracked index placeholder
          (e.g. ``cosmo_idx``, ``hod_idx``), with coordinates set to the
          unique values found across all matched files.
        - **feature dimensions** — axes returned by ``postprocess`` representing
          the measurement coordinates (e.g. ``ells``, ``k``).
        - **attrs** — metadata dict with keys ``'sample'`` and ``'features'``
          listing the corresponding dimension names.

    Notes
    -----
    Files are sorted before grouping, ensuring a deterministic ordering of
    index values in the output coordinates.
    """

    # Read all files in order
    t0 = time.time()
    results = [reader(group_files) for group_files in groups.values()]
    logger.debug(
        f"Read {sum(len(v) for v in groups.values())} files in {time.time() - t0:.2f} seconds."
    )

    selected_results, features_coords = postprocess(results, **kwargs)
    logger.debug(
        f"Processed data shape: {selected_results.shape}, feature coordinates: {features_coords}"
    )

    if reindex:
        index_arrays = reindex_samples(
            index_arrays, reindex=reindex, group_by=reindex_group_by
        )
        logger.debug(
            f"Re-indexed samples for indexes: {reindex} with group_by: {reindex_group_by}"
        )

    sample_dims = {idx: np.unique(values) for idx, values in index_arrays.items()}

    coords = cast_coords({**sample_dims, **features_coords})
    data = reshape_to_coords(selected_results, coords)

    cout = xarray.DataArray(
        data=data,
        coords=coords,
        attrs={
            "sample": list(sample_dims),
            "features": list(features_coords),
        },
    )

    if drop_singleton_dims:
        cout = cout.squeeze(drop=True)

    return cout


# %% NOTE: Examples compression function (project-dependent, to be moved somewhere else ?)
def compress_x(
    root_dir: str | Path,
    index_arrays: dict[str, list],
) -> xarray.DataArray:
    """
    Compress "x" data from csv files based on collected index arrays.

    Example of a custom compression function to read and stack "x" data from csv files based on the collected index arrays.
    This is a placeholder implementation and should be adapted to the specific file structure and data format of the "x" data.

    Parameters
    ----------
    root_dir : str or Path
        Root directory where the csv files are located.
    index_arrays : dict[str, list]
        Dictionary mapping index names to lists of their raw string values, as collected during file grouping.
        Expected to contain keys 'cosmo_idx' and 'hod_idx' corresponding to the file naming convention.

    Returns
    -------
    xarray.DataArray
        DataArray containing the compressed "x" data with appropriate dimensions and coordinates based on the index arrays.
    """
    logger.warning(
        "Using a placeholder compress_x function. Please adapt this function to your specific file structure and data format."
    )
    root_dir = Path(root_dir)
    cosmos = np.asarray(index_arrays["cosmo_idx"], dtype=int)
    hods = np.asarray(index_arrays["hod_idx"], dtype=int)

    x = []  # Placeholder for the compressed data
    for c in np.unique(cosmos):
        fn = root_dir / f"AbacusSummit_c{c:03d}.csv"
        hod_c = hods[cosmos == c]

        x_c = pd.read_csv(fn).iloc[hod_c]
        x.append(x_c)

    df = pd.concat(x, ignore_index=True)
    x_names = df.columns.str.replace(r"[ #]", "", regex=True)

    # Create coordinates
    index_arrays = reindex_samples(
        index_arrays, reindex=["hod_idx"], group_by=["cosmo_idx"]
    )
    sample_coords = {
        idx: np.unique(index_arrays[idx]) for idx in ["cosmo_idx", "hod_idx"]
    }
    features_coords = {"parameters": x_names}

    coords = cast_coords({**sample_coords, **features_coords})
    data = reshape_to_coords(df.to_numpy(), coords)

    cout = xarray.DataArray(
        data=data,
        coords=coords,
        attrs={
            "sample": list(sample_coords),
            "features": list(features_coords),
        },
    )
    return cout


def compress_data(
    glob_fn: str,
    paths: dict[str, str],
    covariance_hod: int = 157,
    test_filters: dict[str, list] | None = None,
    override_last_dim: bool = True,
    save_fn: str | Path | None = None,
    **kwargs,
) -> xarray.Dataset:
    """
    Compress mock measurement data into an xarray Dataset.

    Parameters
    ----------
    glob_fn : str
        Glob pattern for collecting mock measurement filenames,
        relative to the encoded glob pattern.
    paths : dict[str, str]
        Dictionary containing paths for measurements and parameters,
        with keys "measurements_dir" and "param_dir".
    covariance_hod : int, optional
        HOD index to use for the covariance data, if available.
        If None, covariance data will not be included in the output dataset.
        Defaults to 157.
    test_filters : dict[str, list], optional
        Dictionary of filters to apply for splitting the test set,
        passed to ``split_test_set``.
        Defaults to None (no splitting).
    override_last_dim : bool, default=True
        If True and covariance data is included, override the coordinates of
        the last dimension of the covariance DataArray to match those of
        the "y" DataArray, ensuring they are aligned for later use.
    save_fn : str or Path, optional
        If provided, path to save the compressed dataset as a numpy file.
        The dataset is converted to a dictionary and saved as a numpy array
        for later loading. Defaults to None (no saving).
    **kwargs
        Additional keyword arguments forwarded to ``compress_mocks``.

    Returns
    -------
    xarray.Dataset
        Dataset containing the compressed "x" and "y", and optionally "covariance_y" data,
        with appropriate dimensions and coordinates based on the collected index arrays.
        If test_filters are provided, the dataset will also
        include split test/train variables with "_test" and "_train" suffixes.
    """
    logger.warning(
        "Using a placeholder compress_data function. Please adapt this function to your specific file structure, data format, and desired processing steps."
    )
    groups, index_arrays = collect_mocks(
        root_dir=Path(paths["measurements_dir"]) / "base",
        glob_pattern="c{cosmo_idx}_ph{phase_idx}/seed{seed}/hod{hod_idx}/" + glob_fn,
        # ignore_index = ['los'],
    )

    x = compress_x(
        root_dir=paths["param_dir"],
        index_arrays=index_arrays,
    )

    y = compress_mocks(
        groups=groups,
        index_arrays=index_arrays,
        reindex=["hod_idx"],
        reindex_group_by=["cosmo_idx", "phase_idx", "seed"],
        **kwargs,
    )
    logger.info(
        f"Compressed x and y data with shapes {x.shape} and {y.shape} respectively."
    )
    data_vars = {"x": x, "y": y}

    # Covariance
    if covariance_hod is not None:
        groups, index_arrays = collect_mocks(
            root_dir=Path(paths["measurements_dir"]) / "small",
            glob_pattern="c{cosmo_idx}_ph{phase_idx}/seed{seed}/"
            + f"hod{covariance_hod:03d}/"
            + glob_fn,
            # ignore_index = ['los'],
        )
        covariance_y = compress_mocks(
            groups=groups,
            index_arrays=index_arrays,
            **kwargs,
        )
        if override_last_dim:
            # Check consistency of last dimensions
            lcy = covariance_y.coords[covariance_y.dims[-1]]
            ly = y.coords[y.dims[-1]]
            if lcy.name != ly.name:
                logger.warning(
                    f"Inconsistent last dimensions between covariance_y and y: {lcy.name} vs {ly.name}."
                )
            if lcy.shape != ly.shape:
                raise ValueError(
                    f"Cannot override last dimension of covariance_y due to shape mismatch: {lcy.shape} vs {ly.shape}. "
                    "Ensure that the coordinates along this dimension are compatible for alignment."
                )
            covariance_y = covariance_y.assign_coords({lcy.name: ly})
            logger.debug(
                f"Overriding last dimension of covariance_y with y coordinates along {ly.name}."
            )
        else:
            logger.warning(
                "Not overriding last dimension of covariance_y. Ensure that the coordinates are aligned with y to avoid nan values."
            )
        logger.info(f"Compressed covariance_y data with shape {covariance_y.shape}.")
        data_vars["covariance_y"] = covariance_y

    cout = xarray.Dataset(data_vars=data_vars)

    if test_filters is not None:
        cout = split_test_set(cout, test_filters)

    if save_fn is not None:
        Path(save_fn).parent.mkdir(parents=True, exist_ok=True)
        payload = np.array(dataset_to_dict(cout), dtype=object)
        np.save(save_fn, payload)
        logger.info(f"Saved compressed dataset to {save_fn}.")

    return cout
