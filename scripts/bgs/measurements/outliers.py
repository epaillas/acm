"""
Identify outliers in measurements using sigma clipping.
Note that this requires all the HOD folders to have all their measurements computed for the expected mocks (otherwise the compression functions will crash).

Usage:
    python outliers.py --sim_type small --measurements tpcf ds_xiqg ds_xigg --ells 0 2 --seed 0 --sigma 6.0 --log_level info
"""

import logging
import argparse
import numpy as np
from pathlib import Path
from astropy.stats import sigma_clip
from acm.utils.modules import get_class_from_module
from acm.utils.logging import setup_logging

def get_covariance(cls, **kwargs):
    """
    Compresses the covariance of the given statistic.
    
    Parameters
    ----------
    cls
        Class of the observable to load the data for.
    **kwargs
        Additional keyword arguments to pass to `compress_covariance`
        
    Returns
    -------
    phases: np.ndarray
        List of phase indexes associated to each mock.
    cov: np.ndarray
        Covariance array, of shape (n_sample, n_features)
    """
    obs = cls() # Instantiate the observable class
    data = obs.compress_covariance(**kwargs)
    phases = data.phase_idx.values
    cov = obs.flatten_output(data.covariance_y, flat_output_dims=2).values # to 2D numpy array
    return phases, cov

def get_y(cls, **kwargs):
    """
    Compresses the measureents of the given statistic.
     
    Parameters
    ----------
    cls
        Class of the observable to load the data for.
    **kwargs
        Additional keyword arguments to pass to `compress_data`
        
    Returns
    -------
    index: np.ndarray[tuple]
        List of cosmologies & phases indexes associated to each mock.
    y: np.ndarray
        Data array, of shape (n_sample, n_features)
    """    
    obs = cls()
    data = obs.compress_data(**kwargs)
    y = obs.flatten_output(data.y, flat_output_dims=2).values # to 2D numpy array
    
    # Get the indexes as tuples of (cosmo, hod)
    cosmologies = data.cosmo_idx.values
    index = []
    for cosmo_idx in cosmologies:
        hod_kwargs = dict(
            phase = kwargs.get('phase', 0),
            seed = kwargs.get('seed', 0),
            density_threshold=kwargs.get('density_threshold', None)
        )
        hods = obs.get_hod_from_files(cosmo_idx, **hod_kwargs)
        for hod in hods:
            index.append( (cosmo_idx, hod) )
    index = np.array(index)
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
    mask = clipped_data.mask.any(axis=1)  # True for outlier samples
    return index[mask]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify outliers in TPCF measurements using sigma clipping.")
    parser.add_argument('--module', type=str, default='acm.observables.bgs', help='Base module path for observables')
    parser.add_argument('--sim_type', type=str, choices=['small', 'base'], default='small', help='Type of simulation to analyze.')
    parser.add_argument('--ells', type=int, nargs='+', default=[0, 2], help='Multipoles to consider.')
    parser.add_argument('--seed', type=int, default=0, help='Seed number for measurements.')
    parser.add_argument('--sigma', type=float, default=6.0, help='Sigma threshold for clipping.')
    parser.add_argument('--measurements', type=str, nargs='+', choices=['tpcf', 'ds_xiqg', 'ds_xigg'], default='tpcf', help='Type of measurements to analyze.')
    parser.add_argument('--log_level', type=str, default='warning', help='Set logging level (e.g., DEBUG, INFO)')
    args = parser.parse_args()
    
    sim_type = args.sim_type
    ells = args.ells
    sigma = args.sigma
    
    logger = logging.getLogger(__file__.split('/')[-1])
    setup_logging(level=args.log_level)
    
    logger.info(f'Using {sim_type} boxes')
    logger.info(f'Using multipoles: {ells}')
    logger.info(f'Using sigma clipping threshold: {sigma}')
    
    for stat_name in args.measurements:
        cls = get_class_from_module(args.module, stat_name)
        
        if sim_type == 'small':
            index, data = get_covariance(cls, ells=ells)
        elif sim_type == 'base':
            index, data = get_y(cls, ells=ells)
        else: 
            raise ValueError(f"sim_type must be one of ['base', 'small']")
        
        outliers = get_outliers(data, index, sigma=sigma, axis=0)
        n_outliers = len(outliers)
        
        if n_outliers > 0:
            print(f'Found {n_outliers} {stat_name} outliers at indices:', *outliers, sep='\n')
        else: 
            print(f'Found no {stat_name} outliers')