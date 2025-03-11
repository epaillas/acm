import numpy as np
import pandas as pd
from pathlib import Path
from astropy.stats import sigma_clip
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule
from pycorr import TwoPointCorrelationFunction
import torch


def read_lhc(statistic='wp', n_hod=5000):
    data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/wp/z0.5/yuan23_prior/c000_ph000/seed0/'
    lhc_y = []
    for hod in range(n_hod):
        data_fn = Path(data_dir) / f'{statistic}_hod{hod:03}.npy'
        data = TwoPointCorrelationFunction.load(data_fn)
        lhc_y.append(data.corr)
    lhc_y = np.array(lhc_y)

    data_dir = Path('/pscratch/sd/e/epaillas/emc')
    lhc_x = pd.read_csv(data_dir / 'hod_params/yuan23/hod_params_yuan23_c000.csv')
    lhc_x_names = list(lhc_x.columns)
    lhc_x_names = [name.replace(' ', '').replace('#', '') for name in lhc_x_names]
    lhc_x = lhc_x.values[:len(lhc_y),:]
    return lhc_x, lhc_y


lhc_x, lhc_y = read_lhc()
print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

# mask outliers
mask = sigma_clip(lhc_y, sigma=6, axis=0, masked=True).mask
mask = np.all(~mask, axis=1)
lhc_x = lhc_x[mask]
lhc_y = lhc_y[mask]
print(f'After sigma clipping: {lhc_x.shape}, {lhc_y.shape}')

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
                        val_fraction=0.2, batch_size=256,
                        num_workers=2)
    data.setup()

    model = FCN(
            n_input=data.n_input,
            n_output=data.n_output,
            n_hidden=[512, 512, 512, 512],
            dropout_rate=0., 
            learning_rate=1.e-3,
            scheduler_patience=30,
            scheduler_factor=0.5,
            scheduler_threshold=1.e-6,
            weight_decay=0.,
            act_fn='learned_sigmoid',
            # act_fn='SiLU',
            loss='rmse',
            training=True,
            mean_output=train_mean,
            std_output=train_std,
            mean_input=train_mean_x,
            std_input=train_std_x,
        )

    val_loss, model, early_stop_callback = train.fit(
        data=data, model=model,
        model_dir=f'/pscratch/sd/e/epaillas/emc/trained_models/wp/may21_leaveout_{i}/',
    )