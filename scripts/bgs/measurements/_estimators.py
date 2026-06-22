"""Temporary estimator classes, that will eventually be replaced by ACM standardized classes.""" # noqa: INP001
import logging
import pickle
from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np
from lsstypes.external import from_pycorr
from pycorr import TwoPointCorrelationFunction

from acm.estimators.galaxy_clustering.density_split import DensitySplit
from acm.estimators.galaxy_clustering.spectrum import PowerSpectrumMultipoles
from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform
from acm.utils.compression import LsstypeObject

logger = logging.getLogger('_estimators')

def save_lsstype(filename: str | Path, obj: LsstypeObject, overwrite: bool = False) -> None:
    """Save an lstype object to an h5 file."""
    fn = Path(filename)
    if jax.process_index() != 0:  # Only process 0 saves to disk
        return  # Exit early for non-zero processes
    if fn.suffix not in ('.h5', '.hdf5'):
        raise ValueError(f"{fn} must have one of the following extensions: {('.h5', '.hdf5')}")
    if fn.exists() and overwrite is False:
        logger.info(f'File {fn} exists and {overwrite=}. Skipping...')
        return

    fn.parent.mkdir(exist_ok=True, parents=True)
    tmp_fn = fn.with_name(fn.stem + ".tmp" + fn.suffix)
    obj.write(tmp_fn)
    logger.info(f"Writing estimator to {fn}.")
    tmp_fn.replace(fn)  # Atomic move to avoid partial writes


def compute_tpcf(positions: np.ndarray, save_fn: str | Path, **kwargs) -> None:
    """
    Compute the two-point correlation function (2PCF) in (s, mu) bins using the pycorr package.

    Parameters
    ----------
    positions : np.ndarray
        The positions of the galaxies, with shape (N, 3).
    save_fn : str | Path, optional
        The filename to save the LSStypes estimator to.
    **kwargs
        Additional keyword arguments to pass to the TwoPointCorrelationFunction constructor.
    """
    tpcf = TwoPointCorrelationFunction(data_positions1 = positions, **kwargs)
    save_lsstype(save_fn, from_pycorr(tpcf))

def compute_power_spectrum(
    positions: np.ndarray,
    save_fn: str | Path,
    boxsize: float | list | np.ndarray,
    boxcenter: float | list | np.ndarray,
    **kwargs) -> None:
    """
    Compute the power spectrum using ACM's PowerSpectrumMultipoles wrapper around jaxpower.

    Parameters
    ----------
    positions : np.ndarray
        The positions of the galaxies, with shape (N, 3).
    save_fn : str | Path, optional
        The filename to save the LSStypes estimator to.
    boxsize : float | list | np.ndarray
        The size of the simulation box.
    boxcenter: float | list | np.ndarray
        The center of the simulation box.
    **kwargs
        Additional keyword arguments to pass to the compute_spectrum method. Can also contan 'meshsize'.
    """
    meshsize = kwargs.pop('meshsize', 512)
    ps = PowerSpectrumMultipoles(data_positions=positions, boxsize=boxsize, boxcenter=boxcenter, meshsize=meshsize)
    ps.set_density_contrast(resampler='tsc', interlacing=3, compensate=True) # FIXME: Remove hardcoded values!
    ps.compute_spectrum(save_fn=save_fn, **kwargs)  # ty:ignore[invalid-argument-type]

def compute_density_split(
    positions: np.ndarray,
    save_fn: str | Path,
    ds_type: str,
    method: str,
    boxsize: float | list | np.ndarray,
    boxcenter: float | list | np.ndarray,
    cellsize: float = 5.0,
    smoothing_radius: float = 10.0,
    nquantiles: int = 5,
    query_method: str = 'randoms',
    **kwargs
) -> None:
    """
    Compute the density split statistics: the cross-correlation between the quantile regions and the data, and the auto-correlation of the quantile regions.

    Parameters
    ----------
    positions : np.ndarray
        The positions of the galaxies, with shape (N, 3).
    save_fn: str | Path
        The filename to save the LSStypes estimator to.
    ds_type: str
        Type of density-split to compute. Can be 'correlation' or 'power'.
    method: str
        Which density-split estimator to compute. Can be 'cross' or 'auto'.
    boxsize : float | list | np.ndarray
        The size of the simulation box.
    boxcenter: float | list | np.ndarray
        The center of the simulation box.
    cellsize : float, optional
        The size of the cells for the density field. Defaults to 5.0.
    smoothing_radius : float, optional
        The radius for smoothing the density field. Defaults to 10.0.
    nquantiles : int, optional
        The number of quantiles to split the density field into. Defaults to 5.
    query_method: str
        Query method to use when defining quantiles. Defaults to 'randoms'.
    **kwargs
        Additional keyword arguments to pass to the computation functions.
    """
    ds = DensitySplit(data_positions=positions, boxsize=boxsize, boxcenter=boxcenter, cellsize=cellsize)
    ds.set_density_contrast(smoothing_radius=smoothing_radius)
    ds.set_quantiles(nquantiles=nquantiles, query_method=query_method)

    methods = {
        'cross':'quantile_data',
        'auto': 'quantile',
    }
    allowed_types = ('correlation', 'power')

    if ds_type not in allowed_types:
        raise ValueError(f"Unknown type '{ds_type}'. Available types: 'correlation', 'power'")
    if method not in methods:
        raise ValueError(f'Unknown method {method}. Available methods:')

    if method == 'cross':
        kwargs.update(data_positions=positions)

    _callable = getattr(ds, f'{methods[method]}_{ds_type}')
    _callable(
        save_fn = save_fn,
        **kwargs
    )

def compute_bispectrum() -> None:
    pass

def compute_wst(
    positions: np.ndarray,
    save_fn: str | Path,
    init_fn: str | Path | None = None,
    **kwargs
    ) -> None:
    """
    Compute wst statistics on the positions.

    Parameters
    ----------
    positions : np.ndarray
        The positions of the galaxies, with shape (N, 3).
    save_fn: str | Path
        The filename to save the LSStypes estimator to.
    init_fn: str | Path, optional
        Path to a file containing a kymatio pickled object to bypass kymatio initialization.
    """
    if init_fn is not None and Path(init_fn).exists():
        with Path(init_fn).open('rb') as f:
            init_kymatio = pickle.load(f)  # noqa: S301
    else:
        init_kymatio = None
    wst = WaveletScatteringTransform(data_positions=positions, backend='pypower', init_kymatio=init_kymatio, **kwargs)
    wst.set_density_contrast()
    wst.run(save_fn=save_fn)  # ty:ignore[invalid-argument-type]

def get_estimator(stat_name: str) -> Callable:
    """Return the relevant estimator method for a given stat_name value."""
    if stat_name == 'tpcf':
        _callable = compute_tpcf
    elif stat_name == 'spectrum':
        _callable = compute_power_spectrum
    elif stat_name.startswith('wst'):
        _callable = compute_wst
    elif stat_name.startswith('ds_'):
        _callable = compute_density_split
    else:
        raise ValueError(f"{stat_name} is not a known estimator.")
    return _callable
