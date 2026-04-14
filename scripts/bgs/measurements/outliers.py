"""
Identify outliers in measurements using different methods, including sigma clipping, 
checking for missing files, and checking for corrupted h5 files.
The script provides utility functions for handling nested dictionaries to store the outlier information 
in a structured way based on the simulation parameters (e.g., cosmology, phase, seed, hod index).

Usage:
    python outliers.py --method sigma_clip --measurements tpcf power_spectrum --data_dir /path/to/data --simtype base --extra_args phase=0 seed=0 --save_dir /path/to/save/outliers

Extra notes:
    measurements: power_spectrum quantile_correlation quantile_data_correlation quantile_power quantile_data_power tpcf
"""

import re
import logging
import argparse
from pathlib import Path

import lsstypes
import numpy as np
from astropy.stats import sigma_clip

from acm.observables import Observable
from acm.utils.logging import setup_logging

available_methods = ['sigma_clip', 'missing_files', 'corrupted_h5']
logger = logging.getLogger(__file__.split('/')[-1]) # Use the filename as the logger name

#%% Utility functions
def keep_string_int(s):
    """Extracts the integer part from a string, removing any non-numeric characters."""
    return re.sub("[^0-9]", "", s)

def extract_nested_info(path: str | Path, nest_level: int = 4, split_chars: list[str] = ['_', '/']) -> list[int]:
    """
    Extracts the last `nest_level` integers from a given path string.
    
    Parameters
    ----------
    path : str | Path
        The input path from which to extract integers.
    nest_level : int, optional
        The number of nested levels to extract (default is 4).
    split_chars : list[str], optional
        The characters to split the path on (default is ['_', '/']).
    
    Returns
    -------
    list[int]
        A list of the last `nest_level` integers extracted from the path.
    """
    str_values = re.split('|'.join(split_chars), str(path))
    int_values = []
    for s in str_values:
        s_int = keep_string_int(s)
        if s_int != '':
            int_values.append(int(s_int))
    return int_values[-nest_level:]

def nested_set(dic: dict, keys: list, value, extend: bool = False) -> None:
    """
    Recursively sets a value in a nested dictionary given a list of keys.
    
    Parameters
    ----------
    dic : dict
        The dictionary in which to set the value.
    keys : list
        A list of keys representing the nested structure of the dictionary.
    value : any
        The value to set at the specified location in the nested dictionary.
    extend : bool, optional
        If True and the existing value is a list, extend it with the new value (if the new value is also a list). Otherwise, overwrite the existing value (default is False).
        
    Returns
    -------
    None
        The function modifies the input dictionary in place and does not return anything.
    """
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    if extend and isinstance(dic.get(keys[-1]), list) and isinstance(value, list):
        dic[keys[-1]].extend(value)
    else:
        dic[keys[-1]] = value

def nested_count(dic: dict, is_list: bool = False) -> int:
    """Recursively counts the total number of values in a nested dictionary."""
    count = 0
    for v in dic.values():
        if isinstance(v, dict):
            count += nested_count(v, is_list=is_list)
        elif is_list and isinstance(v, list):
            count += len(v)
        else:
            count += 1
    return count

def nested_update(d1: dict, d2: dict, extend: bool = False, keep_unique: bool = False) -> None:
    """Recursively updates a nested dictionary d1 with values from d2."""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            nested_update(d1[k], v, extend=extend, keep_unique=keep_unique)
        else:
            if extend and isinstance(v, list):
                d1[k].extend(v)
                if keep_unique:
                    d1[k] = sorted(set(d1[k])) # Keep unique values only
            else:
                d1[k] = v

def sigma_clip_outliers(data: np.ndarray, index: np.ndarray = None, **kwargs) -> np.ndarray:
    """
    Identify outlier indices in the data using sigma clipping.
    
    Parameters
    ----------
    data: np.ndarray
        Data to apply sigma clipping on, of shape (n_samples, n_features)
    index: np.ndarray, optional
        Index array to map back to original samples, of shape (n_samples, ...)
    **kwargs
        Additional keyword arguments to pass to `astropy.stats.sigma_clip`
        
    Returns
    -------
    outlier_indices: np.ndarray
        Indices of the identified outlier samples.
    """
    if index is None:
        index = np.arange(len(data))
        
    clipped_data = sigma_clip(data, masked=True, **kwargs)
    mask = clipped_data.mask.any(axis=1) # True for outlier samples
    return index[mask]

#%% Outlier detection methods
def check_missing_files(measurement: str, data_dir: str | Path, n_files: int = 3, n_folders: int | None = None, **kwargs):
    """
    Check for missing files in the specified directory.
    
    Parameters
    ----------
    measurement: str
        The name of the measurement to check for missing files.
    data_dir: str | Path
        The directory containing the measurement files.
    n_files: int, optional
        The expected number of files for each measurement (default is 3).
    n_folders: int | None, optional
        If set, will check the expected total number of folders for the measurement and
        raise a warning if the number of folders found does not match the expected number (default is None).
    **kwargs
        Additional keyword arguments to specify the glob patterns for directories and files, 
        and the nesting level for extracting information from the directory structure.
            - glob_pattern: str, pattern to match the directories containing the measurement files (default is 'c*_ph*/seed*/hod*').
            - files_pattern: str, pattern to match the measurement files (default is '{measurement}_los_*.*').
            - nest_level: int, the number of nested levels to extract from the directory structure for identifying the outliers (default is 4).
    
    Returns
    -------
    dict
        A nested dictionary containing the indices of the identified outliers.
    """
    directory_glob = kwargs.get('glob_pattern', 'c*_ph*/seed*/hod*')
    files_glob = kwargs.get('files_pattern', f'{measurement}_los_*.*')
    nest_level = kwargs.get('nest_level', 4)
    
    cout = {}
    folders = sorted(Path(data_dir).glob(directory_glob)) # Find folders where measurements are done
    if n_folders is not None and len(folders) != n_folders:
        logger.warning(f'Found {len(folders)} folders for measurement {measurement}, expected {n_folders}.')
    for f in folders:
        fns = sorted(f.glob(files_glob)) # Check for files matching the measurement pattern
        if len(fns) != n_files:
            logger.debug(f'Folder {f} has {len(fns)} files for measurement {measurement}, expected {n_files}. Marking as outlier.')
            nested_info = extract_nested_info(f, nest_level=nest_level)
            nested_indices = [str(i) for i in nested_info[:-1]] # Convert to strings for dictionary keys
            nested_values = [nested_info[-1]] # The last integer is the hod index, stored as a value
            nested_set(cout, nested_indices, nested_values, extend=True) # Store the missing file information in a nested dictionary
    return cout

def check_corrupted_h5(measurement, data_dir: str | Path, **kwargs):
    """
    Check for corrupted h5 files in the specified directory by attempting to read them with lsstypes.
    
    Parameters
    ----------
    measurement: str
        The name of the measurement to check for corrupted files.
    data_dir: str | Path
        The directory containing the measurement files.
    **kwargs
        Additional keyword arguments to specify the glob patterns for directories and files,
        and the nesting level for extracting information from the directory structure.
            - glob_pattern: str, pattern to match the directories containing the measurement files (default is 'c*_ph*/seed*/hod*').
            - files_pattern: str, pattern to match the measurement files (default is '{measurement}_los_*.h5').
            - nest_level: int, the number of nested levels to extract from the directory structure for identifying the outliers (default is 4).
    
    Returns
    -------
    dict
        A nested dictionary containing the indices of the identified outliers.
    """
    directory_glob = kwargs.get('glob_pattern', 'c*_ph*/seed*/hod*')
    files_glob = kwargs.get('files_pattern', f'{measurement}_los_*.h5')
    nest_level = kwargs.get('nest_level', 4)

    h5_files = sorted(data_dir.glob(f'{directory_glob}/{files_glob}'))
    logger.debug(f'Checking {len(h5_files)} h5 files for measurement {measurement} in directory {data_dir} for corruption.')
    
    cout = {}
    for fn in h5_files:
        try:
            lsstypes.read(fn)
        except OSError as e:
            logger.debug(f'File {fn} is corrupted and cannot be read as a valid h5 file. Marking as outlier.')
            nested_info = extract_nested_info(fn.parent, nest_level=nest_level)
            nested_indices = [str(i) for i in nested_info[:-1]] # Convert to strings for dictionary keys
            nested_values = [nested_info[-1]] # The last integer is the hod index, stored as a value
            nested_set(cout, nested_indices, nested_values, extend=True) # Store the missing file information in a nested dictionary
    return cout

def check_sigma_clip(
    measurement, 
    data_dir: str | Path, 
    mapping_dir: str | Path | None = None, 
    **kwargs
) -> dict:
    """
    Checks the given measurement for outliers using sigma clipping. 
    The function loads the measurement data using the Observable class, 
    applies sigma clipping to identify outliers, and returns a nested dictionary 
    containing the indices of the identified outliers. 
    
    Parameters
    ----------
    measurement: str
        The name of the compressed file to check for outliers.
    data_dir: str | Path
        The directory containing the compressed files. 
    **kwargs
        Additional keyword arguments to pass to the `sigma_clip_outliers` function. 
        Can also contain:
        - simtype: str, type of simulation (e.g., 'base', 'small') to determine how to load the data and construct the index.
        - Extra arguments needed to construct the index based on the simtype (e.g., phase, seed, cosmo, hod).
        
    Returns
    -------
    dict
        A nested dictionary containing the indices of the identified outliers, structured according to the simulation parameters
        
    Note
    ----
    The structure of the index expects the following order: (cosmology, phase, seed, hod).
    Depending on the simtype, some of these parameters may be fixed and not vary across the samples.
    In that case, those should be provided as extra arguments and will be added to the index to maintain a consistent structure for the outlier information.
    Default values will be added if not provided.
    """
    simtype = kwargs.pop('simtype', 'base')
    
    paths = dict(data_dir = str(data_dir))
    observable = Observable(stat_name=measurement, paths=paths, flat_output_dims=2)
    
    if simtype == 'base':
        # Needs fixed phase and seed to complete the index shape
        phase = kwargs.pop('phase', 0)
        seed = kwargs.pop('seed', 0)
                
        y = observable.y.values
        index = observable.y.sample.values # Array of tuples (cosmo, hod_idx)
        index = np.array([[cosmo, phase, seed, hod] for (cosmo, hod) in index]) # Add phase and seed to the index array
        
        if mapping_dir is not None:
            logger.debug(f"Using mapping directory {mapping_dir} to map HOD indices to folder names.")
            for cosmo, phase, seed in np.unique(index[:, :3], axis=0):
                directory_glob = f'c{cosmo:03d}_ph{phase:03d}/seed{seed}/hod*'
                hod_folders = sorted(Path(mapping_dir).glob(directory_glob))
                hod_idx = [extract_nested_info(f, nest_level=1)[0] for f in hod_folders]
                mask = (index[:, 0] == cosmo) & (index[:, 1] == phase) & (index[:, 2] == seed)
                index[mask, -1] = np.array(hod_idx)[:sum(mask)]

    elif simtype == 'small':
        # Needs fixed cosmology, seed and hod to complete the index shape
        cosmo = kwargs.pop('cosmo', 0)
        seed = kwargs.pop('seed', 0)
        hod = kwargs.pop('hod', 157)
        
        y = observable.covariance_y.values
        index = observable.covariance_y.sample.values # Array of tuples (phase,)
        index = np.array([[cosmo, phase, seed, hod] for (phase,) in index]) # Add cosmology, seed and hod to the index array
        
    else:
        raise ValueError(f"Unknown simtype: {simtype}. Supported simtypes are: 'base', 'small'.")

    outlier_indices = sigma_clip_outliers(y, index=index, axis=0, **kwargs)
    cout = {}
    for idx in outlier_indices:
        nested_indices = [str(i) for i in idx[:-1]] # Convert to strings for dictionary keys
        nested_values = [idx[-1]] # The last integer is the hod index, stored as a value
        nested_set(cout, nested_indices, nested_values, extend=True) # Store the outlier information in a nested dictionary
    return cout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Identify outliers in measurements.",
        epilog = """
        For the `sigma_clip` method, the data_dir should be the directory of the compressed files, 
        and the `measurement` values should match the compressed filenames.
        For the other methds, the data_dir should match the raw measurement dir (excluding the simtype) 
        and the `measurement` values should match the filename root (e.g., 'tpcf', 'power_spectrum').
        """
    )
    parser.add_argument('--method', type=str, choices=available_methods, required=True, help="Method to identify outliers.")
    parser.add_argument('--measurements', nargs='+', default='tpcf', help="Measurements to check for outliers.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the measurement files.")
    parser.add_argument('--simtype', type=str, default='base', help="Type of simulation (e.g., 'base', 'small').")
    parser.add_argument('--save_dir', type=str, default=None, help="Directory to save the identified outliers information (optional).")
    parser.add_argument('--mapping_dir', type=str, default=None, help="Directory containing the mapping of HOD indices to folder names (optional, used for sigma_clip method with base simtype).")
    parser.add_argument('--extra_args', nargs='*', help="Additional arguments for the outlier detection method in the form of key=value (optional).")
    parser.add_argument('--log_level', type=str, default='INFO', help="Logging level.")
    args = parser.parse_args()
    
    setup_logging(level=args.log_level)
    
    data_dir = Path(args.data_dir) / args.simtype
    mapping_dir = Path(args.mapping_dir) / args.simtype if args.mapping_dir else None 
    extra_args = {}
    if args.extra_args:
        for arg in args.extra_args:
            key, value = arg.split('=')
            try:
                value = int(value) # Try to convert to integer if possible
            except ValueError:
                pass # Keep as string if not an integer
            extra_args[key] = value
            
    # Depending on the method, call the appropriate function to identify outliers for the measurement
    if args.method == 'missing_files':
        caller = lambda m: check_missing_files(m, data_dir=data_dir, **extra_args)
    elif args.method == 'corrupted_h5':
        caller = lambda m: check_corrupted_h5(m, data_dir=data_dir, **extra_args)
    elif args.method == 'sigma_clip':
        # Do not include simtype in the data directory for that case since the observable will handle it
        caller = lambda m: check_sigma_clip(m, data_dir=data_dir.parent, simtype=args.simtype, mapping_dir=mapping_dir, **extra_args)
    else:
        raise ValueError(f"Unknown method: {args.method}. Available methods are: {available_methods}")
    
    logger.info(f"Starting {args.simtype} outlier identification using method '{args.method}' for measurements: {args.measurements}")
    
    total_outliers = {}
    
    save_file_root = f'{args.method}-simtype_{args.simtype}' # Root name for the saved outliers files, including the method and simtype for clarity

    for m in args.measurements:
        
        logger.debug(f"Identifying outliers for measurement '{m}' using method '{args.method}' with extra arguments: {extra_args}")
                
        stat_outliers = caller(m)
        n_outliers = nested_count(stat_outliers, is_list=True)
        nested_update(total_outliers, stat_outliers, extend=True, keep_unique=True)
        
        # Log the number of outliers identified for the current measurement and save the information if a save directory is provided
        if n_outliers > 0:
            logger.info(f"Identified {n_outliers} outliers for measurement '{m}' using method '{args.method}'.")
            logger.debug(f"Outliers information for measurement '{m}': {stat_outliers}")
            if args.save_dir:
                outliers_path = Path(args.save_dir) / f'{save_file_root}-{m}.npy'
                np.save(outliers_path, stat_outliers)
                logger.info(f"Saved {m} outliers information to {outliers_path}")
        else:
            logger.info(f"No outliers identified for measurement '{m}' using method '{args.method}'.")
    
    # Log the total number of unique outliers identified across all measurements and save the combined information if a save directory is provided
    n_total_outliers = nested_count(total_outliers, is_list=True)
    if n_total_outliers > 0:
        logger.info(f"Total unique outliers identified across all measurements: {n_total_outliers}")
        logger.debug(f"Combined outliers information across all measurements: {total_outliers}")
        if args.save_dir:
            total_outliers_path = Path(args.save_dir) / f'{save_file_root}-all_measurements.npy'
            np.save(total_outliers_path, total_outliers)
            logger.info(f"Saved combined outliers information to {total_outliers_path}")
    else:
        logger.info("No outliers identified across all measurements.")