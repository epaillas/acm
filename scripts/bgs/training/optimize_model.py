import torch
import logging
import argparse
from pathlib import Path
from astropy.stats import sigma_clip
from sunbird.data.transforms_array import LogTransform, ArcsinhTransform
from acm.model.optimize import StudyFCN
from acm.observables import Observable
from acm.utils.logging import setup_logging

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize the model hyperparameters for different observables.')
    parser.add_argument('--compressed_dir', type=str, required=True, help='Directory containing compressed data files')
    parser.add_argument('--statistics', type=str, nargs='+', default=['tpcf'], help='List of statistics to train and optimize the model for')
    parser.add_argument('--study_dir', type=str, required=True, help='Directory to save the optimization study')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the best models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_test', type=int, default=600, help='Number of test samples to hold out from training (only used if the data does not already have a train/test split)')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to run for each statistic')
    parser.add_argument('--transform', type=str, default=None, help='Data transform to apply (e.g., log, arcsinh)')
    parser.add_argument('--same_n_hidden', action='store_true', help='Whether to use the same number of hidden units')
    parser.add_argument('--log_level', type=str, default='warning', help='Set logging level (e.g., DEBUG, INFO)')
    parser.add_argument('--sigma', type=float, default=6.0, help='Sigma threshold for clipping outliers from training data. Set to 0 to disable.')
    args = parser.parse_args()
    
    logger = logging.getLogger(__file__.split('/')[-1])
    setup_logging(level=args.log_level)
    
    paths = dict(data_dir=args.compressed_dir)
    study_dir = Path(args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    
    idx_train = slice(args.n_test, None)  # Assuming first n_test are test data
    
    if args.transform == 'log':
        transform = LogTransform()
    elif args.transform == 'arcsinh':
        transform = ArcsinhTransform()
    elif args.transform is None:
        transform = None
    else:
        raise ValueError(f'Unknown transform: {args.transform}')
    
    for stat_name in args.statistics:
        study_fn = study_dir / f'{stat_name}.pkl'
        logger_fn = study_dir / f'{stat_name}.log'
        setup_logging(filename=logger_fn, level=args.log_level)
        
        logger.info(f'Starting optimization for {stat_name}')
        observable = Observable(stat_name=stat_name, paths=paths, numpy_output=True, flat_output_dims=2)
        
        checkpoint_dir = study_dir / f'{observable}'
        
        lhc_x = observable.get('x_train', observable.x[idx_train])
        lhc_y = observable.get('y_train', observable.y[idx_train])
        lhc_x_names = observable.x_names
        # covariance_matrix = observable.get_covariance_matrix() # Not used in mae loss
        
        # sigma clipping
        if args.sigma > 0:
            mask = sigma_clip(lhc_y, sigma=args.sigma, masked=True, axis=0).mask.any(axis=1)
            lhc_x = lhc_x[~mask]
            lhc_y = lhc_y[~mask]
            logger.info(f'Removed {mask.sum()} outliers from training data using sigma={args.sigma} clipping')
        
        StudyFCN(
            n_trials=args.n_trials,
            same_n_hidden=args.same_n_hidden,
            study_fn=study_fn,
            seed=args.seed,
            checkpoint_dir=checkpoint_dir,
            save_dir=args.save_dir,
            lhc_y=lhc_y,
            lhc_x=lhc_x,
            lhc_x_names=lhc_x_names,
            transform=transform,
            max_epochs=5_000,
            deterministic=True,
            devices=1,
        )