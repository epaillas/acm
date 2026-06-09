import numpy as np


def check_catalog(
    positions: np.ndarray,
    boxsize: np.ndarray | list[float] | float,
    check_in_float32: bool = True,
    center_at_zero: bool = False,
) -> None:
    """
    Perform safety checks on a periodic cubic galaxy catalog.

    It should be called before any clustering statistic is
    measured and any failed checks will cause an assertion error

    Parameters
    ----------
    positions : np.ndarray
        Positions of the galaxies in the catalog. Should be of shape (N_galaxies, 3).
    boxsize : np.ndarray | list[float] | float
        Size of the periodic box. Can be a single float (same size for all dimensions) or an array of shape (3,).
    check_in_float32 : bool, optional
        If True, all checks are performed in single precision (float32). Default is True.
    center_at_zero : bool, optional
        If True, positions are required to be in the range [-L_i/2, L_i/2) for each axis. If False, positions should be in [0, L_i). Default is False.

    Raises
    ------
    ValueError
        If any of the positions fall outside the specified box boundaries.
    """
    # Pick precision
    _dtype = np.float32 if check_in_float32 else np.float64
    positions = positions.astype(_dtype)

    boxsize = np.atleast_1d(np.array(boxsize, dtype=_dtype))
    if len(boxsize) == 1:
        boxsize = np.repeat(boxsize, 3)
    elif len(boxsize) != 3:
        raise ValueError(
            f"boxsize should be a float or an array of shape (3,), but got {boxsize.shape}"
        )

    # Pick right and left edges for each dimension
    offset = boxsize / 2 if center_at_zero else 0.0
    L = np.array([0.0, 0.0, 0.0], dtype=_dtype) - offset
    R = boxsize - offset

    # Do checks
    for i in range(positions.shape[1]):
        left_bound_check = np.all(positions[:, i] >= L[i])
        right_bound_check = np.all(positions[:, i] < R[i])

        min_left = np.min(positions[:, i])
        max_right = np.max(positions[:, i])

        # Build error message:
        em = ""
        if not left_bound_check:
            em += f"{min_left!r} falls out of the box on the left edge {L[i]!r} along the {i}-th axis. "
        if not right_bound_check:
            em += f"{max_right!r} falls out of the box on the right edge {R[i]!r} along the {i}-th axis."
        if em:
            raise ValueError(em)
