"""Default values and helper methods used in the acm package."""

import os
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from typing import Any

import numpy as np

# List of cosmologies in AbacusSummit
cosmo_list = (
    list(range(5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
)

# Flag to indicate if running on NERSC
is_nersc = os.environ.get("NERSC_HOST") == "perlmutter"


def _make_array(
    value: Any,  # noqa: ANN401
    shape: int | tuple[int, ...],
    dtype: str | type = np.float64,
) -> np.ndarray:
    """Return a numpy array by broadcasting the value on the expected shape."""
    toret = np.full(shape, np.nan)
    toret[...] = value
    if np.any(np.isnan(toret)):
        raise ValueError(f"Broadcasted {value} to array but found NaN values inside.")
    return toret.astype(dtype=dtype)


def short_hash(txt: str, length: int | None = 8) -> str:
    """Return a shortened hash of the input string."""
    hashed = sha256(txt.encode()).hexdigest()
    return hashed[:length]


def recursive_update(base: dict, updates: Mapping) -> dict:
    """Recursively merge ``updates`` into ``base`` and return ``base``."""
    for key, value in updates.items():
        current = base.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            base[key] = recursive_update(dict(current), value)
        else:
            base[key] = deepcopy(value)
    return base
