import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.random import RandomState
from cosmoprimo import Cosmology

logger = logging.getLogger(__name__)

def _apply_rsd(data: pd.DataFrame, los: str, hubble: float, az: float) -> pd.DataFrame:
    """
    Apply RSD shift along the los axis. 
    
    Expects velocity columns named 'vx', 'vy', 'vz' 
    corresponding to the position columns 'x', 'y', 'z'.

    Parameters
    ----------
    data : pd.DataFrame
        Galaxy data containing position and velocity columns.
    los : str
        Line-of-sight axis, one of 'x', 'y', 'z'.
    hubble : float
        Hubble parameter H(z) in km/s/(Mpc/h) for the simulation cosmology.
    az : float
        Scale factor a(z) at the snapshot's redshift.

    Returns
    -------
    pd.DataFrame
        Transformed galaxy data with RSD applied.
    """
    data = data.copy()
    v_col = f"v{los}"
    data[los] = data[los] + data[v_col] / (hubble * az)
    return data


def _apply_ap(
    data: pd.DataFrame,
    los: str,
    q_par: float,
    q_perp: float,
    pos_columns: tuple[str],
) -> pd.DataFrame:
    """
    Apply AP scaling: q_par along los, q_perp along transverse axes.
    
    Expects position columns named 'x', 'y', 'z' (or as specified in pos_columns).

    Parameters
    ----------
    data : pd.DataFrame
        Galaxy data containing position columns.
    los : str
        Line-of-sight axis, one of the columns specified in pos_columns.
    q_par : float
        AP scaling factor along the line-of-sight.
    q_perp : float
        AP scaling factor along the transverse directions.
    pos_columns : tuple[str]
        Names of the position columns, e.g. ('x', 'y', 'z').

    Returns
    -------
    pd.DataFrame
        Transformed galaxy data with AP scaling applied.
    """
    data = data.copy()
    for ax in pos_columns:
        data[ax] = data[ax] * (q_par if ax == los else q_perp)
    return data


def _apply_downsample(
    data: pd.DataFrame,
    tracer: str,
    n_gal: int | None = None,
    f_gal: float | None = None,
    nbar: float | None = None,
    volume: Callable[[], np.ndarray] | None = None,
    seed: RandomState | None = None,
) -> pd.DataFrame:
    """
    Randomly downsample a tracer DataFrame.
    
    Volume is expected as a callable to allow for transforms that change the effective volume, e.g. AP scaling.

    Parameters
    ----------
    data : pd.DataFrame
        Galaxy data for a specific tracer.
    tracer : str
        Tracer name, used for logging.
    n_gal : int, optional
        Target number of galaxies.
    f_gal : float, optional
        Fraction of galaxies to keep, between 0 and 1.
    nbar : float, optional
        Target number density in (Mpc/h)^-3.
    volume : callable, optional
        Function that returns the current volume, needed to compute target n_gal when downsampling by nbar.

    Returns
    -------
    pd.DataFrame
        Downsampled galaxy data.

    Raises
    ------
    ValueError
        If not exactly one of n_gal, f_gal or nbar is provided.
    ValueError
        If boxsize is needed but not provided.
    """
    provided = sum(p is not None for p in (n_gal, f_gal, nbar))
    if provided != 1:
        raise ValueError("Exactly one of n_gal, f_gal or nbar must be provided.")

    n_current = len(data)
    if f_gal is not None:
        n_target = round(n_current * f_gal)
    elif n_gal is not None:
        n_target = n_gal
    else:  # nbar
        if volume is None:
            raise ValueError(
                "volume function must be provided when downsampling by nbar."
            )
        # Callable to get current volume, which may include transforms, e.g. AP scaling
        n_target = round(nbar * volume())

    if n_target >= n_current:
        logger.warning(
            f"Target n_gal={n_target} >= current n_gal={n_current} for tracer '{tracer}', skipping downsample."
        )
        return data

    return data.sample(n=n_target, random_state=seed).reset_index(drop=True)

def _add_distance_column(df: pd.DataFrame, cosmo: Cosmology) -> pd.DataFrame:
    """Add a comoving distance column to the DataFrame based on the redshift column."""
    if "z" not in df.columns:
        raise ValueError("DataFrame must contain a 'z' column to compute distances.")
    df = df.copy()
    df["distance"] = cosmo.comoving_radial_distance(df["z"])
    return df