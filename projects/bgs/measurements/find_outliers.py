import re # Regular expressions
import numpy as np
from pathlib import Path
from acm.projects.bgs import *

import logging

# Sometimes, pycorr doesn't outputs the same separation values, so we need to find the outliers (because we can't compare them to the expected values)
# We will filter the separation values that are not within a certain interval of the expected values

def bins_not_ok(bin_values, expected_bin_values, interval=0.3):
    """
    Finds the bins that are not within the interval of the bin_values.

    Parameters
    ----------
    bin_values : np.ndarray
        Array of the bin values.
    expected_bin_values : np.ndarray
        Array of the expected bin values.
    interval : float, optional
        Interval within the bin_values are considered to be the same as the expected_bin_values. The default is 0.2.

    Returns
    -------
    np.ndarray
        Array of the indices of the bins that are not within the interval of the bin_values.
    """
 
    mask = (expected_bin_values - interval > bin_values) | (expected_bin_values + interval < bin_values)
    return (np.where(mask))[0]

def glob_to_regex(glob_pattern: str) -> str:
    """
    Converts a glob pattern to a regex pattern.

    Parameters
    ----------
    glob_pattern : str
        Glob pattern to convert.

    Returns
    -------
    str
        Regex pattern.
    """
    regex_pattern = glob_pattern.replace('.', '\.').replace('*', '(.*)').replace('?', '(.)')
    return regex_pattern

def find_outliers(
    glob_pattern: str ,
    data_dir: str,
    read_data: callable,
    expected_bin_values: np.ndarray,
    interval: float = 0.3
    )-> np.ndarray: 
    """
    Find outliers in the computation of a statistic.
    To find those outliers, the function compares the bin_values of the statistic with expected bin_values.
    
    Parameters
    ----------
    glob_pattern : str
        Glob pattern to match the files.
    data_dir : str
        Path to the directory containing the data files.
    read_data : callable
        Function to read the data to check (e.g. separation values) from the filenames.
        The function must take a filename as input and return the data to check.
    expected_bin_values : np.ndarray
        Array of the expected bin values to check the data against.
    interval : float, optional
        Interval within the bin_values are considered to be the same as the expected_bin_values. The default
        is 0.3.
    
    Returns
    -------
    np.ndarray
        Array with the indices of the outliers.
        
    Warning
    -------
    The function assumes that the pattern left unknown by the glob pattern are numbers, that will be converted to integers.
    Otherwise, the function will raise an error.
    """
    # Logging
    logger = logging.getLogger('outliers')
    
    regex_pattern = glob_to_regex(glob_pattern)
    
    # Load the files
    data_dir = Path(data_dir)
    data_fns = list(sorted(data_dir.glob(glob_pattern)))
    
    index_list = []
    for data_fn in data_fns:
        index = re.findall(regex_pattern, str(data_fn))[0]
        bin_values = read_data(data_fn)
        outliers = bins_not_ok(bin_values, expected_bin_values, interval=interval)
        if len(outliers) > 0:
            logger.warning(f'Outliers for index {index}: {bin_values[outliers]}')
            if type(index) == str:
                index = int(index) # Only one indice in the pattern
            elif type(index) == tuple:
                index = [int(i) for i in index] # Multiple indices in the pattern
            index_list.append(index)
    
    return index_list

if __name__ == '__main__':
    # Logging
    logger = logging.getLogger('outliers')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-28s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    global_outliers = []
    
    # Covariance check
    root_dir = '/pscratch/sd/s/sbouchar/ACM_DS_small/' # Can be changed to also check the data ! 
    glob_pattern = '_c000_ph*_hod096.npy'
    save = True
    
    # Data check
    # root_dir = '/pscratch/sd/s/sbouchar/ACM_DS_data/'
    # glob_pattern = '_c*_hod*.npy'
    # save = False
    
    # tpcf
    def read_tpcf_sep(data_fn):
        data = np.load(data_fn, allow_pickle=True).item()
        s, poles = data(ells=(0, 2), return_sep=True)
        return s
    
    data_dir = root_dir + 'tpcf/'
    edges = np.linspace(0, 30, 31)
    bins = (edges[:-1] + edges[1:]) / 2
    expected_bin_values = bins
    
    tpcf_outliers = find_outliers(
        glob_pattern='tpcf' + glob_pattern,
        data_dir=data_dir,
        read_data=read_tpcf_sep,
        expected_bin_values=expected_bin_values,
    )
    
    if save:
        np.save(data_dir + 'tpcf_outliers_idx.npy', tpcf_outliers)
    logger.info(f'TPCF outliers: {tpcf_outliers}')
    global_outliers.extend(tpcf_outliers)
    
    # Density split correlation
    def read_dsc_conf_sep(data_fn):
        data = np.load(data_fn, allow_pickle=True)
        s, poles = data[0](ells=(0, 2), return_sep=True)
        return np.concatenate([s for _ in data])
    
    data_dir = root_dir + 'dsc_conf/'
    expected_bin_values = np.concatenate([bins for _ in range(5)])
    
    acf_outliers = find_outliers(
        glob_pattern='acf' + glob_pattern,
        data_dir=data_dir,
        read_data=read_dsc_conf_sep,
        expected_bin_values=expected_bin_values,
    )
    ccf_outliers = find_outliers(
        glob_pattern='ccf' + glob_pattern,
        data_dir=data_dir,
        read_data=read_dsc_conf_sep,
        expected_bin_values=expected_bin_values,
    )
    
    if save:
        dsc_conf_outliers = np.unique(np.concatenate([acf_outliers, ccf_outliers]))
        np.save(data_dir + 'dsc_conf_outliers_idx.npy', dsc_conf_outliers)
    logger.info(f'ACF outliers: {acf_outliers}')
    logger.info(f'CCF outliers: {ccf_outliers}')
    global_outliers.extend(acf_outliers)
    global_outliers.extend(ccf_outliers)
    
    global_outliers = np.unique(global_outliers) # Remove duplicates if any
    logger.info(f'{len(global_outliers)} outliers found: {global_outliers}')
    
    # Question : Save outliers as global or differenciate between statistics ??
    if save:
        np.save(root_dir + 'tpcf/outliers_idx.npy', global_outliers)
        np.save(root_dir + 'dsc_conf/outliers_idx.npy', global_outliers)
        np.save(root_dir + 'outliers_idx.npy', global_outliers)