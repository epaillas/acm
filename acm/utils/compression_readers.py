"""
Contains legacy readers and postprocessing functions for compressed TwoPointEstimator objects, such as those produced by pycorr or the densitysplit estimator.

They are provided primarily for backward compatibility with existing data formats, and may be deprecated in the future in favor of more efficient and flexible data handling approaches.
"""

from pathlib import Path

import numpy as np
from pycorr import TwoPointEstimator

from acm.utils.compression import reshape_to_coords

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