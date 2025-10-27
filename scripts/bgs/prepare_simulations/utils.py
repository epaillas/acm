import glob
from pathlib import Path

#%% Phase index utilities
def find_phases(dir: str|Path, z: float, cosmo: int = 0) -> tuple[list[str], list[int]]:
    """
    Finds the simulation phases for a given redshift.

    Parameters
    ----------
    dir : str | Path
        Directory containing the simulation data.
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
    glob_pattern = str(dir / f'AbacusSummit_small_c{cosmo:03d}_ph*' / '*' / f'z{z:.3f}/')
    abacus_fns = sorted(glob.glob(glob_pattern))
    phases = [int(abacus_fn.split('/')[-3].split('_')[-1].lstrip('ph')) for abacus_fn in abacus_fns]
    return abacus_fns, phases

def list_to_sequence(l: list[int]) -> list[tuple[int, int] | int]:
    """
    Converts a list of integers into a list of tuples representing consecutive sequences.

    Parameters
    ----------
    l : list[int]
        A list of integers.
    
    Returns
    -------
    list[tuple[int, int] | int]
        A list of tuples and integers, where each tuple contains the start and end of a consecutive sequence, and standalone integers are included as is.
    """
    l = sorted(set(l)) # Remove duplicates and sort
    sequences = []
    i = 0
    while i < len(l): # Iterate through the list
        j = 0
        while l[i + j] == l[i] + j: # Check for consecutive numbers
            j += 1
            if i + j >= len(l): # Prevent index out of range
                break
        if j > 1: # Add the sequence as a tuple if sequence found (more than 1 consecutive number)
            sequences.append((l[i], l[i + j - 1])) 
        else: # Add the single number if no sequence found
            sequences.append(l[i]) 
        i += j # Move to the next number
    return sequences

#%% Control plots utilities
def find_mocks(dir: str|Path, pattern: str) -> list[str]:
    """
    Finds mock files in a given directory matching a specified pattern.
    
    Parameters
    ----------
    dir : str | Path
        Directory to search for mock files.
    pattern : str
        Pattern to match mock files.
    
    Returns
    -------
    list[str]
        A sorted list of file paths matching the pattern.
    """
    dir = Path(dir)
    files = sorted(dir.glob(pattern))
    files = [str(f) for f in files]

    return files

def get_file_count(files: list[str], z: float, indexes: list[int] = None) -> tuple[dict[int, int], dict[int, int]]:
    """
    Counts the number of halo and particle files for each mock at a given redshift.
    Files should follow the naming convention from prepare_sim:
    - halos_xcom_*_seed600_abacushod_oldfenv_new.h5
    - particles_xcom_*_seed600_abacushod_oldfenv_withranks_new.h5

    Parameters
    ----------
    files : list[str]
        List of file paths to check.
    z : float
        Redshift value to filter files by.
    indexes : list[int], optional
        List of mock indexes (cosmologies, phases, ...) corresponding to the files. If None, uses the range of the length of files.

    Returns
    -------
    halo_counts : dict[int, int]
        Dictionary mapping mock index to number of halo files.
    particle_counts : dict[int, int]
        Dictionary mapping mock index to number of particle files.
    """
    halo_counts = {}
    particle_counts = {}
    
    if indexes is None:
        indexes = range(len(files))

    for f, i in zip(files, indexes):
        f = Path(f)
        hc = len(list(f.glob(f'z{z:.03f}/halos_xcom_*_seed600_abacushod_oldfenv_new.h5')))
        pc = len(list(f.glob(f'z{z:.03f}/particles_xcom_*_seed600_abacushod_oldfenv_withranks_new.h5')))
        halo_counts[i] = hc
        particle_counts[i] = pc
        
    return halo_counts, particle_counts