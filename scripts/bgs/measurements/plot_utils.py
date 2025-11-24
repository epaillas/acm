import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def csv_to_structured_array(filename: str | Path, **kwargs) -> np.ndarray:
    """
    Load a CSV file into a structured array.

    Parameters
    ----------
    filename : str | Path
        Path to the CSV file.
    **kwargs
        Additional keyword arguments to pass to `np.genfromtxt`.

    Returns
    -------
    np.ndarray
        A structured array containing the data from the CSV file.
    """
    delimiter = kwargs.pop('delimiter', ',') # Ensure default delimiter is comma
    names = kwargs.pop('names', True) # Use header names by default
    data = np.genfromtxt(filename, delimiter=delimiter, names=names, **kwargs)
    # data = {name: data[name] for name in data.dtype.names}
    return data


def load_hod_params(hod_params_dir: str | Path, keys: list[str] | None = None) -> dict[str, np.ndarray]:
    """
    Load HOD parameters from CSV files in a specified directory into a dictionary of structured arrays.

    Parameters
    ----------
    hod_params_dir : str | Path
        Path to the directory containing HOD parameter CSV files.
    keys : list[str] | None, optional
        List of keys to use for the dictionary.
        Must be at least as long as the number of files and correspond to the sorted order of the files.
        If None, the filenames (without extension) are used as keys. Defaults to None.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary where keys are specified or derived from filenames,
        and values are structured arrays of HOD parameters.

    Raises
    ------
    ValueError
        If the length of keys is less than the number of HOD parameter files.
    """
    hod_params_dir = Path(hod_params_dir)
    hod_params_files = sorted(hod_params_dir.glob('*.csv'))
    
    if keys is not None and len(keys) < len(hod_params_files):
        raise ValueError("Length of keys must at least match number of HOD parameter files")
    
    hod_params = {}
    for i, fn in enumerate(hod_params_files):
        key = keys[i] if keys is not None else fn.stem
        hod_params[key] = csv_to_structured_array(fn)
    return hod_params


