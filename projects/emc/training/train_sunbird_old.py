import numpy as np
from pathlib import Path
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule
from sunbird.data.data_utils import convert_to_summary
from sunbird.data.transforms_array import LogTransform, ArcsinhTransform
import torch
from acm.data.io_tools import *

torch.set_float32_matmul_precision('high')

def TrainFCN(statistic, learning_rate, n_hidden, dropout_rate, weight_decay):
    final_model = False
    apply_transform = True
    select_filters = {}
    slice_filters = {}

    lhc_x, lhc_y, coords = read_lhc(statistics=[statistic],
                                    select_filters=select_filters,
                                    slice_filters=slice_filters)
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

    covariance_matrix, n_sim = read_covariance(statistics=[statistic],
                                                select_filters=select_filters,
                                                slice_filters=slice_filters)
    print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

    if apply_transform:
        transform = ArcsinhTransform()
        # transform = LogTransform()
        lhc_y = transform.transform(lhc_y)
    else:
        transform = None
        

    nhod = int(len(lhc_y) / 85)
    ntot = len(lhc_y)

    if final_model:
        idx_train = list(range(ntot))
    else:
        idx_train = list(range(nhod * 6, ntot))


    print(f'Using {len(idx_train)} samples for training')

    lhc_train_x = lhc_x[idx_train]
    lhc_train_y = lhc_y[idx_train]

    train_mean = np.mean(lhc_train_y, axis=0)
    train_std = np.std(lhc_train_y, axis=0)

    train_mean_x = np.mean(lhc_train_x, axis=0)
    train_std_x = np.std(lhc_train_x, axis=0)

    data = ArrayDataModule(x=torch.Tensor(lhc_train_x),
                        y=torch.Tensor(lhc_train_y), 
                        val_fraction=0.2, batch_size=128,
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
            loss='mae',
            training=True,
            mean_output=train_mean,
            std_output=train_std,
            mean_input=train_mean_x,
            std_input=train_std_x,
            transform_output=transform,
            standarize_output=True,
            coordinates=coords,
            covariance_matrix=covariance_matrix,
        )

    model_dir = f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/{statistic}/cosmo+hod/optuna_log/'
    # model_dir = f'/pscratch/sd/e/epaillas/emc/trained_models/{statistic}/cosmo+hod/nov20/'
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    print(f'Saving model to {model_dir}')

    val_loss, model, early_stop_callback = train.fit(
        data=data, model=model,
        model_dir=model_dir,
        max_epochs=5000,
        devices=1,
    )
    return val_loss

if __name__ == '__main__':
    statistic = 'pdf_r20'
    TrainFCN(
        statistic=statistic,
        learning_rate=1.e-3,
        n_hidden=[512, 512, 512, 512],
        dropout_rate=0.,
        weight_decay=0,
    )
