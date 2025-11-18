"""
Identify outliers in measurements using sigma clipping.
Note that this requires all the HOD folders to have all their measurements computed (otherwise the shapes of measurements & masks won't match).

Usage:
    python outliers.py --sim_type small --measurements tpcf ds_xiqg ds_xigg --ells 0 2 --seed 0 --sigma 6.0 --log_level info
"""

import logging
import argparse
import numpy as np
from pathlib import Path
from pycorr import TwoPointEstimator
from astropy.stats import sigma_clip
from acm.utils.logging import setup_logging

def load_tpcf(dir: str|Path, ells: list = [0, 2], seed: int = 0):
    """Load the two-point correlation function from the specified directory."""
    dir = Path(dir)
    hod_fns = sorted(dir.glob(f'c000_ph*/seed{seed}/hod*'))
    data = []
    for hod in hod_fns:
        fns = sorted(hod.glob('tpcf_los_*.npy'))
        cf = sum([TwoPointEstimator.load(fn).normalize() for fn in fns])
        data.append(cf(ells=ells).flatten()) # flattened data vector
    return np.array(data) # Shape: (n_sample, n_features)

def load_ds_xiqg(dir: str|Path, ells: list = [0, 2], quantiles: list = [0, 1, 2, 3, 4], seed: int = 0):
    """Load the DensitySplit galaxy-galaxy correlation functions from the specified directory."""
    dir = Path(dir)
    hod_fns = sorted(dir.glob(f'c*_ph*/seed{seed}/hod*'))
    data = []
    for hod in hod_fns:
        fns = sorted(hod.glob('quantile_data_correlation_los_*.npy'))
        cf_q = [sum([np.load(fn, allow_pickle=True)[q].normalize() for fn in fns]) for q in quantiles]
        cf_q_flat = np.array([c(ells=ells).flatten() for c in cf_q]).flatten() # flattened data vector
        data.append(cf_q_flat) # list of flattened data vectors per quantile
    return np.array(data) # Shape: (n_sample, n_quantiles*n_features)

def load_ds_xigg(dir: str|Path, ells: list = [0, 2], quantiles: list = [0, 1, 2, 3, 4], seed: int = 0):
    """Load the DensitySplit galaxy-galaxy correlation functions from the specified directory."""
    dir = Path(dir)
    hod_fns = sorted(dir.glob(f'c*_ph*/seed{seed}/hod*'))
    data = []
    for hod in hod_fns:
        fns = sorted(hod.glob('quantile_correlation_los_*.npy'))
        cf_q = [sum([np.load(fn, allow_pickle=True)[q].normalize() for fn in fns]) for q in quantiles]
        cf_q_flat = np.array([c(ells=ells).flatten() for c in cf_q]).flatten() # flattened data vector
        data.append(cf_q_flat) # list of flattened data vectors per quantile
    return np.array(data) # Shape: (n_sample, n_quantiles*n_features)

def get_outliers_fn(dir: str|Path, data: np.ndarray, seed: int = 0, **kwargs):
    """Identify outlier indices in the data using sigma clipping."""
    clipped_data = sigma_clip(data, masked=True, **kwargs)
    mask = clipped_data.mask.any(axis=1)  # True for outlier samples
    fns = sorted(dir.glob(f'c000_ph*/seed{seed}/hod*'))
    fn_mocks = np.array(['/'.join(f.parts[-3:]) for f in fns])  # Extract mock identifiers from paths
    return fn_mocks[mask]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify outliers in TPCF measurements using sigma clipping.")
    parser.add_argument('--sim_type', type=str, choices=['small', 'base'], default='small', help='Type of simulation to analyze.')
    parser.add_argument('--ells', type=int, nargs='+', default=[0, 2], help='Multipoles to consider.')
    parser.add_argument('--seed', type=int, default=0, help='Seed number for measurements.')
    parser.add_argument('--sigma', type=float, default=6.0, help='Sigma threshold for clipping.')
    parser.add_argument('--measurements', type=str, nargs='+', choices=['tpcf', 'ds_xiqg', 'ds_xigg'], default='tpcf', help='Type of measurements to analyze.')
    parser.add_argument('--log_level', type=str, help='Set logging level (e.g., DEBUG, INFO)', default='warning')
    args = parser.parse_args()
    
    ells = tuple(args.ells)
    sigma = args.sigma
    
    logger = logging.getLogger(__file__.split('/')[-1])
    setup_logging(level=args.log_level)
    
    logger.info(f'Using multipoles: {ells}')
    logger.info(f'Using sigma clipping threshold: {sigma}')
    
    data_dir = Path(f'/pscratch/sd/s/sbouchar/acm/bgs/measurements') / args.sim_type
    
    if 'tpcf' in args.measurements:
        data_vector = load_tpcf(data_dir, ells=ells, seed=args.seed)
        logger.info(f"TPCF: loaded data with shape: {data_vector.shape}")
        outlier_fns = get_outliers_fn(data_dir, data_vector, seed=args.seed, sigma=sigma, axis=0)
        print(f"TPCF: Identified {len(outlier_fns)} outliers at indices: ", *outlier_fns, sep="\n")
        
    if 'ds_xiqg' in args.measurements:
        data_vector = load_ds_xiqg(data_dir, ells=ells, seed=args.seed)
        logger.info(f"DS xi_qg: loaded data with shape: {data_vector.shape}")
        outlier_fns = get_outliers_fn(data_dir, data_vector, seed=args.seed, sigma=sigma, axis=0)
        print(f"DS xi_qg: Identified {len(outlier_fns)} outliers at indices: ", *outlier_fns, sep="\n")
        
    if 'ds_xigg' in args.measurements:
        data_vector = load_ds_xigg(data_dir, ells=ells, seed=args.seed)
        logger.info(f"DS xi_gg: loaded data with shape: {data_vector.shape}")
        outlier_fns = get_outliers_fn(data_dir, data_vector, seed=args.seed, sigma=sigma, axis=0)
        print(f"DS xi_gg: Identified {len(outlier_fns)} outliers at indices: ", *outlier_fns, sep="\n")