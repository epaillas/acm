import numpy as np

def check_catalog(
        positions: np.ndarray, 
        boxsize: np.ndarray | list[float] | float,
        check_in_float32: bool = True,
        center_at_zero: bool = False
    ):
    '''
    This function perofms all reasonable safety checks on a provided catalog
    in a periodic box. It should be called before any clustering statistic is
    measured and any failed checks will cause an assertion error

    Parameters:
    - positions: np.array of shape (N_galaxies,3,)
    - boxsize: np.array of shape (3,) or (1,) or list of floats of same lenghts or float
    - check_in_float32: bool. If True, all checks are performed in single precision
    - center_at_zero: bool. If True, positions are required to be in [-L_i/2,L_i/2) for each axis.
                      If False, [0, L_i) is used.
    '''
    boxsize = np.atleast_1d(np.array(boxsize))
    # Convert boxsize to (3,) array, if required
    if isinstance(boxsize, float):
        boxsize = np.array([boxsize, boxsize, boxsize])
    elif len(boxsize)==1:
        boxsize = np.array([boxsize[0], boxsize[0], boxsize[0]])
    else:
        pass

    # Pick precision
    if check_in_float32:
        positions = positions.astype(np.float32)
        boxsize   = boxsize.astype(np.float32)
        dtype     = np.float32
    else:
        positions = positions.astype(np.float64)
        boxsize   = boxsize.astype(np.float64)
        dtype     = np.float64

    # Pick right and left edges for each dim
    if center_at_zero:
        L = -boxsize.astype(dtype) / 2
        R = boxsize.astype(dtype) / 2
    else:
        L = np.array([0.0,0.0,0.0], dtype=dtype)
        R = boxsize.astype(dtype)

    # Do checks
    for i in range(positions.shape[1]):
        assert np.all(positions[:,i] >= L[i]), f'{repr(np.min(positions[:,i]))} falls out of the box on the left edge {repr(L[i])} along the 0-th axis'
        assert np.all(positions[:,i] < R[i]), f'{repr(np.max(positions[:,i]))} falls out of the box on the right edge {repr(R[i])} along the 0-th axis'
