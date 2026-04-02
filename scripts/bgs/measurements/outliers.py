"""
Identify outliers in measurements using sigma clipping.
Note that this requires all the HOD folders to have all their measurements computed for the expected mocks (otherwise the compression functions will crash).

Usage:
    python outliers.py --sim_type small -md /pscratch/sd/s/sbouchar/acm/bgs-20/measurements -pd /pscratch/sd/s/sbouchar/acm/bgs-20/parameters/cosmo+hod_params --compress_args "{ells: [0, 2], seed: 0}" --measurements tpcf ds_xiqg ds_xiqq --sigma 6.0 --log_level info 

    python outliers.py --sim_type base -md /pscratch/sd/s/sbouchar/acm/bgs-20/measurements --method h5_corruption --measurements power_spectrum ds_pkqg ds_pkqq --log_level info
"""

import logging
import argparse
from pathlib import Path

import lsstypes
import numpy as np
from astropy.stats import sigma_clip

from acm.utils.modules import get_class_from_module
from acm.utils.logging import setup_logging
from visual_tools import nested_set

def get_covariance(obs, **kwargs):
    """
    Compresses the covariance of the given statistic.
    
    Parameters
    ----------
    obs
        Observable class to load the data for.
    **kwargs
        Additional keyword arguments to pass to `compress_covariance`
        
    Returns
    -------
    index: np.ndarray
        List of cosmologies, phases, seeds and HOD indexes associated to each mock. Shape (n_samples, 4)
    cov: np.ndarray
        Covariance array, of shape (n_sample, n_features)
    """
    data = obs.compress_covariance(**kwargs)
    cov = obs.flatten_output(data.covariance_y, flat_output_dims=2).values # to 2D numpy array
    
    phases = data.phase_idx.values
    cosmo_idx = kwargs.get('cosmo_idx', 0)
    hod_idx = kwargs.get('hod_idx', 157)
    seed = kwargs.get('seed', 0)
    index = []
    for phase in phases:
        index.append([cosmo_idx, phase, seed, hod_idx])
    index = np.asarray(index)

    return index, cov

def get_y(obs, **kwargs):
    """
    Compresses the measurements of the given statistic.
     
    Parameters
    ----------
    obs
        Observable class to load the data for.
    **kwargs
        Additional keyword arguments to pass to `compress_data`
        
    Returns
    -------
    index: np.ndarray
        List of cosmologies, phases, seeds and HOD indexes associated to each mock. Shape (n_samples, 4)
    y: np.ndarray
        Data array, of shape (n_sample, n_features)
    """    
    data = obs.compress_data(**kwargs)
    y = obs.flatten_output(data.y, flat_output_dims=2).values # to 2D numpy array
    
    # Get the indexes as tuples of (cosmo, hod)
    cosmologies = data.cosmo_idx.values
    phase = kwargs.get('phase', 0)
    seed = kwargs.get('seed', 0)
    paths = kwargs.get('paths', None)
    index = []
    for cosmo_idx in cosmologies:
        hod_kwargs = dict(phase = phase, seed = seed, density_threshold=kwargs.get('density_threshold', None))
        hods = obs.get_hod_from_files(paths=paths, cosmo_idx=cosmo_idx, **hod_kwargs)
        for hod in hods:
            index.append([cosmo_idx, phase, seed, hod])
    index = np.asarray(index)
    return index, y

def get_outliers(data: np.ndarray, index: np.ndarray = None, **kwargs):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify outliers in TPCF measurements using sigma clipping.")
    parser.add_argument('--module', type=str, default='acm.observables.bgs', help='Base module path for observables')
    parser.add_argument('--method', type=str, default='sigma_clip', help='Method to identify outliers. "h5_corruption" identifies outliers based on corrupted h5 files, while "sigma_clip" uses sigma clipping on the data.')
    parser.add_argument('--measurements_dir', '-md', type=str, default=None, required=True, help='Directory containing measurement files')
    parser.add_argument('--param_dir', '-pd', type=str, default=None, help='Directory containing HOD parameter files')
    parser.add_argument('--sim_type', type=str, choices=['small', 'base'], default='small', help='Type of simulation to analyze.')
    parser.add_argument('--compress_args', type=str, default='{}', help=r'Additional keyword arguments to pass to the compression functions, as a stringified dictionary. E.g., "{ells: [0, 2], seed: 0}"')
    parser.add_argument('--sigma', type=float, default=6.0, help='Sigma threshold for clipping.')
    parser.add_argument('--measurements', type=str, nargs='+', default='tpcf', help='Type of measurements to analyze.')
    parser.add_argument('--save_dir', type=str, help='Directory to save the outlier results to a file.')
    parser.add_argument('--log_level', type=str, default='warning', help='Set logging level (e.g., DEBUG, INFO)')
    args = parser.parse_args()
    
    sim_type = args.sim_type
    sigma = args.sigma
    compress_args = {}.update(eval(args.compress_args)) # Convert stringified dict to actual dict
    
    logger = logging.getLogger(__file__.split('/')[-1])
    setup_logging(level=args.log_level)
    
    logger.info(f'Using {sim_type} boxes')
    logger.info(f'Using sigma clipping threshold: {sigma}')
    logger.info(f'Using method: {args.method}')
    
    paths = dict(
        measurements_dir = args.measurements_dir,
        param_dir = args.param_dir
    )
    
    outliers_dict_all = {} # To store outliers for all statistics if saving
    
    for stat_name in args.measurements:
        
        # For the h5 corruption method, we simply check which files cannot be read as valid h5 files, and consider those as outliers. 
        # This is a quick way to identify corrupted files without needing to load all the data into memory.
        if args.method == 'h5_corruption':
            n_outliers = 0
            outliers_dict = {}
            
            data_dir = Path(args.measurements_dir) / sim_type
            
            h5_files = list(data_dir.glob(f'*/*/*/{stat_name}*.h5')) # Assuming the structure is measurements_dir/sim_type/c000_ph000/seed0/hod000/stat_name*.h5
            
            for fn in h5_files:
                try:
                    lsstypes.read(fn)
                except OSError as e:
                    logger.warning(f'File {fn} is corrupted and cannot be read as a valid h5 file. Marking as outlier.')
                    
                    # Extract cosmo, phase, seed and hod from the file path
                    parts = fn.parts
                    cosmo = parts[-4].split('_')[0].lstrip('c') # e.g., 000
                    phase = parts[-4].split('_')[1].lstrip('ph') # e.g., 000
                    seed = parts[-3].lstrip('seed') # e.g., 0
                    hod = parts[-2].lstrip('hod') # e.g., 000
                    
                    # Update the outliers dictionary with the corrupted file information
                    nested_set(outliers_dict, [cosmo, phase, seed], hod, extend=True)
                    nested_set(outliers_dict_all, [cosmo, phase, seed], hod, extend=True)
                    
                    n_outliers += 1
            
            if args.save_dir and n_outliers > 0:
                outlier_dir = Path(args.save_dir)
                outlier_dir.mkdir(exist_ok=True)
                        
                outlier_fn = outlier_dir / f'{stat_name}_outliers_simtype-{sim_type}_h5_corruption.npy'
                np.save(outlier_fn, outliers_dict)
                logger.info(f'Saved {n_outliers} {stat_name} outliers to {outlier_fn}')
                
                outlier_fn_all = outlier_dir / f'all_outliers_simtype-{sim_type}_h5_corruption.npy'
                np.save(outlier_fn_all, outliers_dict_all)
                logger.info(f'Saved all outliers to {outlier_fn_all}')
        
        elif args.method == 'sigma_clip':
            cls = get_class_from_module(args.module, stat_name)
            
            # Update compression args for specific statistics if needed 
            if stat_name in ['ds_xiqg', 'ds_xiqq']:
                compress_args['quantiles'] = [0, 1, 2, 3, 4]
            
            outliers_dict = {}
            
            if sim_type == 'small':
                index, data = get_covariance(cls, paths=paths, **compress_args)
            elif sim_type == 'base':
                index, data = get_y(cls, paths=paths, **compress_args)
            else: 
                raise ValueError(f"sim_type must be one of ['base', 'small']")
            
            outliers = get_outliers(data, index, sigma=sigma, axis=0)
            n_outliers = len(outliers)
            
            if n_outliers > 0:
                print(f'Found {n_outliers} {stat_name} outliers at indices:', *outliers, sep='\n')
            else: 
                print(f'Found no {stat_name} outliers')
            
            if args.save_dir and n_outliers > 0: 
                # Structure outliers into a nested dictionary
                cosmo_idx = outliers[:,0]
                for cosmo in np.unique(cosmo_idx):
                    phase_mask = outliers[:,0] == cosmo
                    phases = outliers[phase_mask, 1]
                    for phase in np.unique(phases):
                        seed_mask = phases == phase
                        seeds = outliers[phase_mask][seed_mask, 2]
                        for seed in np.unique(seeds):
                            hod_mask = seeds == seed
                            hods = outliers[phase_mask][seed_mask][hod_mask, 3]
                            nested_set(outliers_dict, [str(cosmo), str(phase), str(seed)], hods.tolist(), extend=True)
                            
                            # Also update the all-statistics dictionary
                            nested_set(outliers_dict_all, [str(cosmo), str(phase), str(seed)], hods.tolist(), extend=True)
                            outliers_dict_all[str(cosmo)][str(phase)][str(seed)] = list(set(outliers_dict_all[str(cosmo)][str(phase)][str(seed)])) # Unique HODs indexes only
                            
                outlier_dir = Path(args.save_dir)
                outlier_dir.mkdir(exist_ok=True)
                
                outlier_fn = outlier_dir / f'{stat_name}_outliers_simtype-{sim_type}_sigma-{sigma}.npy'
                np.save(outlier_fn, outliers_dict)
                logger.info(f'Saved {n_outliers} {stat_name} outliers to {outlier_fn}')
                
                outlier_fn_all = outlier_dir / f'all_outliers_simtype-{sim_type}_sigma-{sigma}.npy'
                np.save(outlier_fn_all, outliers_dict_all)
                logger.info(f'Saved all outliers to {outlier_fn_all}')
        
        else:
            raise ValueError(f"Method must be one of ['h5_corruption', 'sigma_clip']")