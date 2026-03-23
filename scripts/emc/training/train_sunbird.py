import numpy as np
from pathlib import Path
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule
from sunbird.data.transforms_array import LogTransform, ArcsinhTransform
import torch
from acm.observables import Observable
from acm import setup_logging
import argparse

torch.set_float32_matmul_precision('high')

def _build_transform(transform_name):
    if transform_name is None:
        return None
    if transform_name == 'log':
        return LogTransform()
    if transform_name == 'arcsinh':
        return ArcsinhTransform()
    raise ValueError(f'Unknown transform: {transform_name}')


def TrainFCN(observable, learning_rate, n_hidden, dropout_rate, weight_decay, 
    model_dir=None, transform_input=None, transform_output=None, val_cosmo_fraction=0.1, seed=None):

    np.random.seed(seed)
    
    lhc_x = observable.x
    lhc_y = observable.y
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

    ncosmo, nhod = [len(observable.get_coordinate_list(name)) for name in ['cosmo_idx', 'hod_idx']]
    print(f'Number of cosmologies: {ncosmo}, Number of HODs: {nhod}')

    covariance_matrix = observable.get_covariance_matrix(volume_factor=64)
    print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

    input_transform = _build_transform(transform_input)
    output_transform = _build_transform(transform_output)

    if input_transform is not None:
        lhc_x = input_transform.transform(lhc_x)

    if output_transform is not None:
        lhc_y = output_transform.transform(lhc_y)
        
    ntot = len(lhc_y)
    n_test_cosmo = 6
    idx_train = list(range(nhod * n_test_cosmo, ntot))
    # idx_train = list(range(ntot))

    print(f'Using {len(idx_train)} samples for training')

    lhc_train_x = lhc_x[idx_train]
    lhc_train_y = lhc_y[idx_train]

    cosmo_values = np.array(observable.get_coordinate_list('cosmo_idx'))
    train_cosmos = cosmo_values[n_test_cosmo:]
    if len(train_cosmos) < 2:
        raise ValueError('Need at least 2 training cosmologies to build a cosmology-level validation split.')

    n_val_cosmo = max(1, int(len(train_cosmos) * val_cosmo_fraction))
    n_val_cosmo = min(n_val_cosmo, len(train_cosmos) - 1)
    val_cosmos_list = np.random.choice(train_cosmos, size=n_val_cosmo, replace=False)
    val_cosmos = set(val_cosmos_list)

    cosmo_labels_train = np.repeat(train_cosmos, nhod)
    val_idx = np.where(np.isin(cosmo_labels_train, list(val_cosmos)))[0].tolist()

    train_mean = np.mean(lhc_train_y, axis=0)
    train_std = np.std(lhc_train_y, axis=0)

    train_mean_x = np.mean(lhc_train_x, axis=0)
    train_std_x = np.std(lhc_train_x, axis=0)

    data = ArrayDataModule(x=torch.Tensor(lhc_train_x),
                        y=torch.Tensor(lhc_train_y), 
                        val_idx=val_idx, batch_size=128,
                        num_workers=0)
    data.setup()

    model = FCN(
            n_input=data.n_input,
            n_output=data.n_output,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate, 
            learning_rate=learning_rate,
            scheduler_patience=10,
            scheduler_factor=0.5,
            scheduler_threshold=1.e-6,
            weight_decay=weight_decay,
            act_fn='learned_sigmoid',
            loss='weighted_mae',
            training=True,
            mean_output=train_mean,
            std_output=train_std,
            mean_input=train_mean_x,
            std_input=train_std_x,
            transform_input=input_transform,
            transform_output=output_transform,
            standarize_output=True,
            covariance_matrix=covariance_matrix,
        )

    model_dir = './' if model_dir is None else model_dir
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    print(f'Saving model to {model_dir}')

    val_loss, model, early_stop_callback = train.fit(
        data=data, model=model,
        model_dir=model_dir,
        max_epochs=5000,
        devices=1,
        logger='tensorboard',
        log_dir=model_dir
    )
    return val_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train FCN for EMC observables.')
    parser.add_argument('--transform_input', type=str, choices=['log', 'arcsinh'], default=None, help='Transform to apply to inputs.')
    parser.add_argument('--transform_output', type=str, choices=['log', 'arcsinh'], default=None, help='Transform to apply to outputs.')
    parser.add_argument('-s', '--statistic', type=str, default='bispectrum', help='Statistic to train on.')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory to save the model.')
    parser.add_argument('--val_cosmo_fraction', type=float, default=0.1, help='Fraction of training cosmologies to hold out for validation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    setup_logging()

    paths = {
        'data_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.3/abacus/compressed/',
        'measurements_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.3/abacus/',
        'param_dir': None
    }

    observable = Observable(stat_name=args.statistic, paths=paths, numpy_output=True, flat_output_dims=2)

    if args.model_dir is not None:
        model_dir = args.model_dir
    else:
        model_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/models/v1.3/best/{args.statistic}/'

    TrainFCN(
        observable=observable,
        learning_rate=1.e-3,
        n_hidden=[512, 512, 512, 512],
        dropout_rate=0.,
        weight_decay=0,
        model_dir=model_dir,
        transform_input=args.transform_input,
        transform_output=args.transform_output,
        val_cosmo_fraction=args.val_cosmo_fraction,
        seed=args.seed,
    )
