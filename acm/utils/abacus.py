import logging
import re
from pathlib import Path
from typing import overload

import pandas as pd

logger = logging.getLogger(__name__)

BOXSIZES = {
    "base": 2000,
    "high": 1000,
    "highbase": 1000,
    "huge": 7500,
    "hugebase": 2000,
    "fixedbase": 1185,
    "small": 500,
    "png": 2000,
}

ABACUS_MAP = {
    "logM1": ["log_1"],
    "Acent": ["A_cen"],
    "Asat": ["A_sat"],
    "Bcent": ["B_cen"],
    "Bsat": ["B_sat"],
}

def get_abacus_simname(sim_type: str, cosmo_idx: int, phase_idx: int) -> str:
    """Build the Abacus simulation name based on the provided parameters."""
    if sim_type == "png":
        return f"Abacus_{sim_type}base_c{cosmo_idx:03d}_ph{phase_idx:03d}"
    else:
        return f"AbacusSummit_{sim_type}_c{cosmo_idx:03d}_ph{phase_idx:03d}"

@overload
def map_params(params: dict, mapping: dict[str, list[str]] | None = None) -> dict:
    ...
@overload
def map_params(params: list[str], mapping: dict[str, list[str]] | None = None) -> list[str]:
    ...
def map_params(
    params: dict | list[str], 
    mapping: dict[str, list[str]] | None = None,
) -> dict | list[str]:
    """
    Map custom parameters names to fixed parameters.

    Parameters
    ----------
    params : dict | list[str]
        Dictionary or list of custom parameters.
    mapping : dict[str, list[str]]
        Mapping from custom parameter names to fixed parameter names.
        Keys are fixed parameter names, values are lists of custom parameter names that map to the fixed parameter name.

    Returns
    -------
    dict | list[str]
        Dictionary or list of fixed parameters. Use the same type as the input params.

    Raises
    ------
    ValueError
        If the type of params is not dict or list.
    """
    mapping = mapping or ABACUS_MAP
    
    if type(params) not in [dict, list]:
        raise ValueError("Invalid type for params. Must be either dict or list.")

    for abacus_key, custom_keys in mapping.items():
        for custom_key in custom_keys:
            if custom_key in params:  # Check if the custom key is used
                # Replace custom key with Abacus key
                if isinstance(params, dict):
                    params[abacus_key] = params.pop(custom_key)
                else:  # is list
                    params[params.index(custom_key)] = abacus_key
    return params

def load_abacus_cosmologies(
    filename: Path | str,
    cosmologies: list[int],
    parameters: list[str],
    mapping: dict[str, str] | None = None,
) -> dict:
    """
    Load the AbacusSummit cosmology parameters from the AbacusSummit cosmologies csv file.

    Select the `cosmologies` indexes and the parameters to keep. Renames the parameters according to mapping.

    Parameters
    ----------
    filename : Path | str
        Filename (csv) with the AbacusSummit cosmology parameters.
    cosmologies : list[int]
        List of cosmologies indexes to select.
    parameters : list[str]
        List of parameters to keep.
    mapping : dict[str, str] | None, optional
        Dictionary with the mapping from the original parameter names to the desired names.

    Returns
    -------
    dict
        Dictionary with the selected cosmology parameters for the selected cosmologies.
    """
    filename = Path(filename)  # Ensure filename is a Path object
    csv = pd.read_csv(filename, usecols=["root", *parameters])

    root = csv["root"]
    params = csv[parameters]

    cnames = [f"abacus_cosm{c:03d}" for c in cosmologies]  # cosmology names to select
    index = pd.Index([f"c{c:03d}" for c in cosmologies])  # New cosmology indexes

    cosmo_params = params[root.isin(cnames)]
    if not cosmo_params.empty:
        cosmo_params = cosmo_params.set_index(index)
    if mapping is not None:
        cosmo_params = cosmo_params.rename(columns=mapping)
    return cosmo_params.to_dict(orient="index")


def get_abacus_phases(
    phase_dir: str | Path,
    z: float,
    cosmo: int = 0,
) -> tuple[list[Path], list[int]]:
    """
    Find the simulation phases for a given redshift.

    Parameters
    ----------
    phase_dir : str | Path
        Directory containing the simulation data.
        Files are expected to follow the structure:
        `AbacusSummit_small_c{cosmo:03d}_ph{phase:03d}/.../z{z:.3f}/`
    z : float
        Redshift value for which to find the simulation phases.
    cosmo : int, optional
        Cosmology index to search phases for (default is 0).

    Returns
    -------
    tuple[list[Path], list[int]]
        A tuple containing a list of file paths and a list of phase indices.
    """
    phase_dir = Path(phase_dir)  # Ensure phase_dir is a Path object

    if not phase_dir.is_dir() or not phase_dir.exists():
        raise ValueError(f"Provided phase_dir {phase_dir} is not a valid directory.")

    # Patterns (NOTE: hardcoded structure !)
    z_str = f"{z:.3f}".replace(".", r"\.")  # Convert z to a string suitable for regex
    re_expr = rf"AbacusSummit_small_c{cosmo:03d}_ph(?P<phase>\d+)\/.+\/z{z_str}"
    glob_pattern = f"AbacusSummit_small_c{cosmo:03d}_ph*/*/z{z:.3f}/"

    re_pattern = re.compile(re_expr)
    fns = sorted(phase_dir.glob(glob_pattern))

    phases = []
    out_fns = []
    for f in fns:
        match = re_pattern.search(str(f.as_posix()))
        if match:
            phases.append(int(match.group("phase")))
            out_fns.append(f)
        else:
            logger.warning(
                f"File {f} does not match the expected pattern and will be skipped in the phase indexes."
            )

    return out_fns, phases
