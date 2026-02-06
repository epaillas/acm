"""
Outdated script to train and optimize a model for given hyperparameters.
Use optimize_model.py instead.
"""

import torch
import logging
import argparse
from pathlib import Path
from astropy.stats import sigma_clip
from sunbird.emulators.train import train_fcn
from sunbird.data.transforms_array import LogTransform, ArcsinhTransform
from acm.observables import Observable
from acm.utils.logging import setup_logging

from pytorch_lightning import seed_everything

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize the model hyperparameters for different observables.')
    parser.add_argument('--compressed_dir', type=str, required=True, help='Directory containing compressed data files')
    parser.add_argument('--statistics', type=str, nargs='+', default=['tpcf'], help='List of statistics to train and optimize the model for')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to save the checkpoints')
    # parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the best models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_test', type=int, default=600, help='Number of test samples to hold out from training (only used if the data does not already have a train/test split)')
    # parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to run for each statistic')
    parser.add_argument('--transform', type=str, default=None, help='Data transform to apply (e.g., log, arcsinh)')
    # parser.add_argument('--same_n_hidden', action='store_true', help='Whether to use the same number of hidden units')
    parser.add_argument('--log_level', type=str, default='warning', help='Set logging level (e.g., DEBUG, INFO)')
    parser.add_argument('--sigma', type=float, default=6.0, help='Sigma threshold for clipping outliers from training data. Set to 0 to disable.')
    args = parser.parse_args()
    
    logger = logging.getLogger(__file__.split('/')[-1])
    setup_logging(level=args.log_level)
    
    seed_everything(args.seed, workers=True)
    
    paths = dict(data_dir=args.compressed_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
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
        logger_fn = checkpoint_dir / f'{stat_name}.log'
        setup_logging(filename=logger_fn, level=args.log_level)
        
        logger.info(f'Starting training for {stat_name}')
        observable = Observable(stat_name=stat_name, paths=paths, numpy_output=True, flat_output_dims=2)
        
        checkpoint_dir = checkpoint_dir / f'{observable}'
        
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
        
        val_loss = train_fcn(
            lhc_y=lhc_y,
            lhc_x=lhc_x,
            lhc_x_names=lhc_x_names,
            n_hidden = [512, 512, 512, 512],
            learning_rate = 1e-3,
            weight_decay = 0.,
            dropout_rate = 0.,
            transform=transform,
            max_epochs=5_000,
            deterministic=True,
            devices=1,
            checkpoint_dir=checkpoint_dir,
        )
        logger.info(f'Final validation loss for {stat_name}: {val_loss}')
        
        
# Usage
"""
python /global/homes/s/sbouchar/acm/scripts/bgs/training/train_model.py \
    --compressed_dir /pscratch/sd/s/sbouchar/acm/bgs-20/input_data \
    --checkpoint_dir /pscratch/sd/s/sbouchar/acm/bgs-20/trained_models/sigma_clipped/study \
    --transform arcsinh \
    --log_level info \
    --statistics tpcf
"""