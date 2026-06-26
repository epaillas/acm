"""Default values and helper methods used in the acm package."""

import os
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
