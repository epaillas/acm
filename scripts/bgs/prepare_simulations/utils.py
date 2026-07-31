from pathlib import Path  # noqa: INP001


#%% Phase index utilities
def list_to_sequence(val: list[int]) -> list[tuple[int, int] | int]:
    """
    Convert a list of integers into a list of tuples representing consecutive sequences.

    Parameters
    ----------
    val : list[int]
        A list of integers.

    Returns
    -------
    list[tuple[int, int] | int]
        A list of tuples and integers, where each tuple contains the start and end
        of a consecutive sequence, and standalone integers are included as is.
    """
    val = sorted(set(val)) # Remove duplicates and sort
    sequences = []
    i = 0
    while i < len(val): # Iterate through the list
        j = 0
        while val[i + j] == val[i] + j: # Check for consecutive numbers
            j += 1
            if i + j >= len(val): # Prevent index out of range
                break
        if j > 1: # Add the sequence as a tuple if sequence found (more than 1 consecutive number)
            sequences.append((val[i], val[i + j - 1]))
        else: # Add the single number if no sequence found
            sequences.append(val[i])
        i += j # Move to the next number
    return sequences

#%% Control plots utilities
def find_mocks(directory: str|Path, pattern: str) -> list[str]:
    """
    Find mock files in a given directory matching a specified pattern.

    Parameters
    ----------
    directory : str | Path
        Directory to search for mock files.
    pattern : str
        Pattern to match mock files.

    Returns
    -------
    list[str]
        A sorted list of file paths matching the pattern.
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    files = [str(f) for f in files]

    return files

def get_file_count(
    files: list[str],
    z: float,
    indexes: list[int] | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Count the number of halo and particle files for each mock at a given redshift.

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
    indexes = indexes or list(range(len(files)))

    for f, i in zip(files, indexes, strict=True):
        f = Path(f)  # noqa: PLW2901
        hc = len(list(f.glob(f'z{z:.03f}/halos_xcom_*_seed600_abacushod_oldfenv_new.h5')))
        pc = len(list(f.glob(f'z{z:.03f}/particles_xcom_*_seed600_abacushod_oldfenv_withranks_new.h5')))
        halo_counts[i] = hc
        particle_counts[i] = pc
    return halo_counts, particle_counts
