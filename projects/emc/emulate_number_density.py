import numpy as np
# import pandas as pd
from pathlib import Path
from astropy.stats import sigma_clip
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule
# from pycorr import TwoPointCorrelationFunction
import torch


def read_lhc(statistic='quantile_data_correlation', n_hod=50_000, return_sep=False):
    if statistic == 'density_split':
        data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/number_density/z0.5/yuan23_prior/c000_ph000/seed0/'
        data_fn = Path(data_dir) / f'number_density_lhc.npy'
    else:
        data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/{statistic}/z0.5/yuan23_prior/c000_ph000/seed0/'
        data_fn = Path(data_dir) / f'{statistic}_lhc.npy'
    data = np.load(data_fn, allow_pickle=True).item()
    lhc_x = data['lhc_x'][:n_hod]
    lhc_y = data['lhc_y'][:n_hod]
    return lhc_x, lhc_y


statistic = 'number_density_downsampled'


lhc_x, lhc_y = read_lhc(statistic=statistic, n_hod=50_000)
lhc_y = lhc_y.reshape(len(lhc_y), 1)
print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

# # mask outliers
# mask = sigma_clip(lhc_y, sigma=6, axis=0, masked=True).mask
# mask = np.all(~mask, axis=1)
# lhc_x = lhc_x[mask]
# lhc_y = lhc_y[mask]
# print(f'After sigma clipping: {lhc_x.shape}, {lhc_y.shape}')

# if statistic == 'tpcf':
#     # apply a log transform to the output features
#     lhc_y = np.log10(lhc_y - np.min(lhc_y) + 1.e-6)

ntot = len(lhc_y)
nstep = int(ntot / 5)

for i in range(5):
    start_idx = i * nstep
    end_idx = (i + 1) * nstep
    print(f'Training leaveout {i}. Start index: {start_idx}, end index: {start_idx + nstep}')
    idx_train = list(range(0, start_idx)) + list(range(end_idx, ntot))
    idx_test = list(range(start_idx, end_idx))

    lhc_train_x = lhc_x[idx_train]
    lhc_train_y = lhc_y[idx_train]
    lhc_test_x = lhc_x[idx_test]
    lhc_test_y = lhc_y[idx_test]

    train_mean = np.mean(lhc_y, axis=0)
    train_std = np.std(lhc_y, axis=0)

    train_mean_x = np.mean(lhc_x, axis=0)
    train_std_x = np.std(lhc_x, axis=0)

    data = ArrayDataModule(x=torch.Tensor(lhc_train_x),
                        y=torch.Tensor(lhc_train_y), 
                        val_fraction=0.2, batch_size=128,
                        num_workers=0)
    data.setup()

    model = FCN(
            n_input=data.n_input,
            n_output=data.n_output,
            n_hidden=[512, 512],
            dropout_rate=0., 
            learning_rate=1.e-3,
            scheduler_patience=30,
            scheduler_factor=0.5,
            scheduler_threshold=1.e-6,
            weight_decay=0.5,
            act_fn='learned_sigmoid',
            # act_fn='SiLU',
            # loss='rmse',
            loss='mae',
            training=True,
            mean_output=train_mean,
            std_output=train_std,
            mean_input=train_mean_x,
            std_input=train_std_x,
        )
    
    model_dir = f'/pscratch/sd/e/epaillas/emc/trained_models/{statistic}/jun19_leaveout_{i}/'
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    val_loss, model, early_stop_callback = train.fit(
        data=data, model=model,
        model_dir=model_dir,
    )
