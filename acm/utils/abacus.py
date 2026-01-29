import glob
from pathlib import Path
import pandas as pd

def load_abacus_cosmologies(
    filename: str, 
    cosmologies: list[int], 
    parameters: list[str],
    mapping: dict[str, str] = None,
    ) -> dict:
    """
    Loads the AbacusSummit cosmology parameters from the AbacusSummit cosmologies csv file and selects
    the `cosmologies` indexes. Also selects the parameters to keep. Renames the parameters according to mapping.

    Parameters
    ----------
    filename : str
        Filename (csv) with the AbacusSummit cosmology parameters.
    cosmologies : list[int]
        List of cosmologies indexes to select.
    parameters : list[str]
        List of parameters to keep.
    mapping : dict[str, str], optional
        Dictionary with the mapping from the original parameter names to the desired names.
    
    Returns
    -------
    dict
        Dictionary with the selected cosmology parameters for the selected cosmologies.
    """
    cosmo_params = pd.read_csv(
        filename,
        usecols = ['root'] + parameters,
    )
    cosmo_params = cosmo_params[cosmo_params['root'].isin([f'abacus_cosm{c:03d}' for c in cosmologies])]
    cosmo_params.drop(columns=['root'], inplace=True)
    cosmo_params.set_index(pd.Index([f'c{c:03d}' for c in cosmologies]), inplace=True)
    if mapping is not None:
        cosmo_params.rename(columns=mapping, inplace=True)
    return cosmo_params.to_dict(orient='index')

def get_abacus_phases(dir: str|Path, z: float, cosmo: int = 0) -> tuple[list[str], list[int]]:
    """
    Finds the simulation phases for a given redshift.

    Parameters
    ----------
    dir : str | Path
        Directory containing the simulation data.
        Files are expected to follow the structure:
        `AbacusSummit_small_c{cosmo:03d}_ph{phase:03d}/.../z{z:.3f}/`
    z : float
        Redshift value for which to find the simulation phases.
    cosmo : int, optional
        Cosmology index to search phases for (default is 0).

    Returns
    -------
    tuple[list[str], list[int]]
        A tuple containing a list of file paths and a list of phase indices.
    """
    dir = Path(dir) # Ensure dir is a Path object
    glob_pattern = str(dir / f'AbacusSummit_small_c{cosmo:03d}_ph*' / '**' / f'z{z:.3f}/')
    abacus_fns = sorted(glob.glob(glob_pattern))
    phases = [int(Path(f).relative_to(dir).parts[0].split('_')[-1].lstrip('ph')) for f in abacus_fns]
    return abacus_fns, phases