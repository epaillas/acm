import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_abacus_cosmologies(
    filename: str,
    cosmologies: list[int],
    parameters: list[str],
    mapping: dict[str, str] | None = None,
) -> dict:
    """
    Load the AbacusSummit cosmology parameters from the AbacusSummit cosmologies csv file.

    Select the `cosmologies` indexes and the parameters to keep. Renames the parameters according to mapping.

    Parameters
    ----------
    filename : str
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
    cosmo_params = pd.read_csv(
        filename,
        usecols=["root", *parameters],
    )
    cosmo_params = cosmo_params[
        cosmo_params["root"].isin([f"abacus_cosm{c:03d}" for c in cosmologies])
    ]
    cosmo_params = cosmo_params.drop(columns=["root"])
    cosmo_params = cosmo_params.set_index(pd.Index([f"c{c:03d}" for c in cosmologies]))
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

    glob_pattern = f"AbacusSummit_small_c{cosmo:03d}_ph*/*/z{z:.3f}/"
    abacus_fns = sorted(phase_dir.glob(glob_pattern))

    z_str = f"{z:.3f}".replace(".", r"\.")  # Convert z to a string suitable for regex
    re_pattern = re.compile(
        rf"AbacusSummit_small_c{cosmo:03d}_ph(?P<phase>\d+)\/.+\/z{z_str}"
    )

    phases = []
    for f in abacus_fns:
        match = re_pattern.search(str(f))
        if match:
            phases.append(int(match.group("phase")))
        else:
            logger.warning(
                f"File {f} does not match the expected pattern and will be skipped."
            )

    fns = [Path(phase_dir) / fn for fn in abacus_fns]
    return fns, phases
