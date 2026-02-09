import numpy as np
from pathlib import Path
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule
from sunbird.data.transforms_array import LogTransform, ArcsinhTransform
import torch
from acm.observables import Observable
import argparse

torch.set_float32_matmul_precision('high')

def TrainFCN(observable, learning_rate, n_hidden, dropout_rate, weight_decay, 
    model_dir=None, transform=None):

    # load the data
    lhc_x = observable.x
    lhc_y = observable.y
    # covariance_matrix = observable.get_covariance_matrix(divide_factor=64)
    # coordinates = observable.coordinates
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

    ncosmo, nhod = [len(observable.get_coordinate_list(name)) for name in ['cosmo_idx', 'hod_idx']]
    print(f'Number of cosmologies: {ncosmo}, Number of HODs: {nhod}')

    # covariance_matrix = observable.get_covariance_matrix(volume_factor=64)
    # print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

    if transform is not None:
        if transform == 'log':
            transform = LogTransform()
        elif transform == 'arcsinh':
            transform = ArcsinhTransform()
        else:
            raise ValueError(f'Unknown transform: {args.transform}')
        lhc_y = transform.transform(lhc_y)
        
    ntot = len(lhc_y)
    idx_train = list(range(nhod * 6, ntot))
    # idx_train = list(range(ntot))

    print(f'Using {len(idx_train)} samples for training')

    lhc_train_x = lhc_x[idx_train]
    lhc_train_y = lhc_y[idx_train]

    train_mean = np.mean(lhc_train_y, axis=0)
    train_std = np.std(lhc_train_y, axis=0)

    train_mean_x = np.mean(lhc_train_x, axis=0)
    train_std_x = np.std(lhc_train_x, axis=0)

    data = ArrayDataModule(x=torch.Tensor(lhc_train_x),
                        y=torch.Tensor(lhc_train_y), 
                        val_fraction=0.1, batch_size=128,
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
            # act_fn='SiLU',
            # loss='GaussianNLoglike',
            # loss='mae',
            training=True,
            mean_output=train_mean,
            std_output=train_std,
            mean_input=train_mean_x,
            std_input=train_std_x,
            output_transform=transform,
            standarize_output=True,
            # coordinates=coordinates,
            # covariance_matrix=covariance_matrix,
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
    parser.add_argument('--transform', type=str, choices=['log', 'arcsinh'], default=None, help='Transform to apply to outputs.')
    parser.add_argument('-s', '--statistic', type=str, default='bispectrum', help='Statistic to train on.')
    args = parser.parse_args()

    paths = {
        'data_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/compressed/',
        'measurements_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/',
        'param_dir': None
    }

    observable = Observable(stat_name=args.statistic, paths=paths, numpy_output=True, flat_output_dims=2)

    model_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/models/v1.2/best/{args.statistic}/'
    TrainFCN(
        observable=observable,
        learning_rate=1.e-3,
        n_hidden=[512, 512, 512, 512],
        dropout_rate=0.,
        weight_decay=0,
        model_dir=model_dir,
        transform=args.transform,
    )
