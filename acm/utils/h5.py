"""Utilities for reading/writing h5 files."""

import json
from typing import overload

import numpy as np
from h5py import Group

# %% Helper functions for saving state dicts to h5 files. Arrays become datasets, everything else becomes JSON __meta__ attr.
# Borrowed from cosmoprimo.emulators.tools.utils


@overload
def _prepare_for_json(obj: np.ndarray) -> dict: ...
@overload
def _prepare_for_json(obj: tuple) -> dict: ...
@overload
def _prepare_for_json(obj: list) -> list: ...
@overload
def _prepare_for_json(obj: dict) -> dict: ...
@overload
def _prepare_for_json(obj: np.integer) -> int: ...
@overload
def _prepare_for_json(obj: np.floating) -> float: ...
@overload
def _prepare_for_json(obj: object) -> object: ...
def _prepare_for_json(obj: object) -> object:
    """
    Recursively convert obj to JSON-serializable form, encoding tuples and numpy arrays.

    Parameters
    ----------
    obj : object
        The object to convert.

    Returns
    -------
    object
        The JSON-serializable representation of obj.
    """
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist(), "__dtype__": str(obj.dtype)}  # ty:ignore[no-matching-overload] ???
    if isinstance(obj, tuple):
        return {"__tuple__": [_prepare_for_json(v) for v in obj]}
    if isinstance(obj, list):
        return [_prepare_for_json(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _prepare_for_json(v) for k, v in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


@overload
def _restore_from_json(obj: dict) -> np.ndarray: ...
@overload
def _restore_from_json(obj: dict) -> tuple: ...
@overload
def _restore_from_json(obj: dict) -> dict: ...
@overload
def _restore_from_json(obj: list) -> list: ...
@overload
def _restore_from_json(obj: object) -> object: ...
def _restore_from_json(obj: object) -> object:
    """
    Recursively restore obj from JSON-serializable form, decoding tuples and numpy arrays.

    Inverse of _prepare_for_json.

    Parameters
    ----------
    obj : object
        The JSON-serializable representation of an object.

    Returns
    -------
    object
        The restored object.
    """
    if isinstance(obj, dict):
        if set(obj.keys()) == {"__ndarray__", "__dtype__"}:
            _object = obj.get("__ndarray__")
            dtype = obj.get("__dtype__")
            return np.array(_object, dtype=dtype)  # ty:ignore[no-matching-overload]
        if set(obj.keys()) == {"__tuple__"}:
            _tuple = obj.get("__tuple__")
            if not isinstance(_tuple, list):
                raise TypeError(
                    f"Expected list for __tuple__ attribute, got {type(_tuple)}"
                )
            return tuple(_restore_from_json(v) for v in _tuple)
        return {k: _restore_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_from_json(v) for v in obj]
    return obj


def _h5_write_state(h5grp: Group, state: dict) -> None:
    """
    Write a state dict to h5 group. Arrays become datasets, everything else becomes JSON __meta__ attr.

    Parameters
    ----------
    h5grp : Group
        The h5 group to write to.
    state : dict
        The state dict to write.
    """
    arr_keys = {k for k, v in state.items() if isinstance(v, np.ndarray)}
    for key in arr_keys:
        h5grp.create_dataset(key, data=state[key])
    meta = {k: v for k, v in state.items() if k not in arr_keys}
    h5grp.attrs["__meta__"] = json.dumps(_prepare_for_json(meta))


def _h5_read_state(h5grp: Group) -> dict:
    """
    Read a state dict from h5 group written by _h5_write_state.

    Parameters
    ----------
    h5grp : Group
        The h5 group to read from.

    Returns
    -------
    dict
        The state dict read from the h5 group.
    """
    state = {}
    for key in h5grp:
        state[key] = h5grp[key][...]
    meta_str = h5grp.attrs.get("__meta__", None)
    if meta_str is not None:
        meta_state = _restore_from_json(json.loads(str(meta_str)))
        if not isinstance(meta_state, dict):
            raise ValueError(
                f"Expected dict for __meta__ attribute, got {type(meta_state)}"
            )
        state.update(meta_state)
    return state
