import numpy as np
import pandas as pd
from pathlib import Path
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule
from pycorr import TwoPointCorrelationFunction
import torch


def read_lhc():
    data_dir = Path('/pscratch/sd/e/epaillas/emc')
    data_fn = Path(data_dir) / 'training_sets/tpcf/z0.5/yuan23_prior/cosmopower/tpcf.npy'
    lhc_y = np.load(data_fn, allow_pickle=True,).item()
    s = lhc_y['s']
    lhc_y = lhc_y['multipoles']
    lhc_x = pd.read_csv(data_dir / 'hod_params/yuan23/hod_params_yuan23_c000.csv')
    lhc_x_names = list(lhc_x.columns)
    lhc_x_names = [name.replace(' ', '').replace('#', '') for name in lhc_x_names]
    lhc_x = lhc_x.values[:len(lhc_y),:]
    return lhc_x, lhc_y

def read_covariance():
    data_dir = Path('/pscratch/sd/e/epaillas/emc')
    covariance_path = data_dir / 'covariance/tpcf/z0.5/yuan23_prior/'
    n_for_covariance = 1_000
    covariance_files = list(covariance_path.glob('tpcf_ph*.npy'))[:n_for_covariance]
    covariance_y = [
        TwoPointCorrelationFunction.load(file)[::4](ells=(0,2),).reshape(-1) for file in covariance_files
    ]
    prefactor = 1./8.
    return prefactor * np.cov(np.array(covariance_y).T)


nstep = 6000
for i in range(2, 5):
    start_idx = i * nstep
    end_idx = (i + 1) * nstep
    print(f'Training leaveout {i}. Start index: {start_idx}, end index: {start_idx + nstep}')
    idx_train = list(range(0, start_idx)) + list(range(end_idx, 30000))
    idx_test = list(range(start_idx, end_idx))

    lhc_x, lhc_y = read_lhc()
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
        model_dir=f'/pscratch/sd/e/epaillas/emc/trained_models/tpcf/may9_leaveout_{i}/',
    )