import numpy as np
from sunbird.emulators import FCN
from pathlib import Path
from sunbird.inference.hamiltonian import HMC
from sunbird.inference.priors import Yuan23, AbacusSummit
from sunbird.data.data_utils import convert_to_summary
import torch
from astropy.stats import sigma_clip


def summary_coords(statistic, data):
    if statistic == 'dsc_conf':
        return {
            'statistics': ['quantile_data_correlation', 'quantile_correlation'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            's': data['s'],
        }
    elif statistic == 'tpcf':
        return {
            'multipoles': [0, 2],
            's': data['s'],
        }

def lhc_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/{statistic}/cosmo+hod/z0.5/yuan23_prior/ph000/seed0/'
    return Path(data_dir) / f'{statistic}_lhc.npy'

def covariance_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/{statistic}/cosmo+hod/z0.5/yuan23_prior/ph000/seed0/'
    return Path(data_dir) / f'{statistic}_lhc.npy'

def filter_lhc(lhc_y, coords, filters):
    dimensions = list(coords.keys())
    dimensions.insert(0, 'mock_idx')
    coords['mock_idx'] = np.arange(lhc_y.shape[0])
    lhc_y = lhc_y.reshape([len(coords[d]) for d in dimensions])
    lhc_y = convert_to_summary(data=lhc_y, dimensions=dimensions, coords=coords)
    # return lhc_y.sel(**filters).values.reshape(lhc_y.shape[0], -1)
    fil = [getattr(getattr(lhc_y, key), 'isin')(value) for key, value in filters.items()]
    for i, cond in enumerate(fil):
        mask = mask & cond if i > 0 else fil[0]
    mask = lhc_y.where(mask & (lhc_y.s > smin)).to_masked_array().mask
    return lhc_y.values[~mask].reshape(lhc_y.shape[0], -1), mask[0]

def read_lhc(statistic='dsc_conf', filters={}):
    data_fn = lhc_fnames(statistic)
    data = np.load(data_fn, allow_pickle=True).item()
    coords = summary_coords(statistic, data)
    lhc_x = data['lhc_x']
    lhc_x_names = data['lhc_x_names']
    lhc_y = data['lhc_y']
    if filters:
        lhc_y, mask = filter_lhc(lhc_y, coords, filters)
    return lhc_x, lhc_y, lhc_x_names, mask

def read_model(statistic):
    if statistic == 'wp':
        checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/wp/cosmo+hod/jul10_trans/last-v30.ckpt'
    if statistic == 'pk':
        checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/pk/jun25_leaveout_0/last.ckpt'
    elif statistic == 'tpcf':
        # checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/tpcf/cosmo+hod/jul9/last.ckpt'
        checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/tpcf/cosmo+hod/jul24/last.ckpt'
    elif statistic == 'dsc_conf':
        # checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/density_split/jun2_9_leaveout_0/last.ckpt'
        checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_conf/cosmo+hod/jul25/last.ckpt'
    elif statistic == 'dsc_fourier':
        checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_fourier/jun24_leaveout_0/last-v1.ckpt'
    elif statistic == 'wst':
        checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/wst/jun27_leaveout_0/last.ckpt'
    elif statistic == 'minkowski':
        checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/minkowski/Minkowski-best-model-epoch=276-val_loss=0.02366.ckpt'
    model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
    model.eval()
    return model

def read_covariance(statistic, filters={}):
    data_fn = covariance_fnames(statistic)
    data = np.load(data_fn, allow_pickle=True).item()
    coords = summary_coords(statistic, data)
    y = data['cov_y']
    if filters:
        y, mask = filter_lhc(y, coords, filters)
    prefactor = 1 / 64
    cov = prefactor * np.cov(y, rowvar=False)
    return cov, len(y)

def get_emulator_error(lhc_test_y, pred_test_y):
    return np.median(np.abs(lhc_test_y - pred_test_y), axis=0)

def get_covariance_correction(n_s, n_d, n_theta=None, correction_method='percival'):
    if correction_method == 'percival':
        B = (n_s - n_d - 2) / ((n_s - n_d - 1)*(n_s - n_d - 4))
        return (n_s - 1)*(1 + B*(n_d - n_theta))/(n_s - n_d + n_theta - 1)
    elif correction_method == 'hartlap':
        return (n_s - 1)/(n_s - n_d - 2)

getdist_ranges = {
    'omega_b': [0.0207, 0.0243],
    'omega_cdm': [0.1032, 0.140],
    'sigma8_m': [0.678, 0.938],
    'n_s': [0.9012, 1.025],
    'nrun': [-0.038, 0.038],
    'N_ur': [1.188, 2.889],
    'w0_fld': [-1.22, -0.726],
    'wa_fld': [-0.628, 0.621],
    'logM_cut': [12.5, 13.7],
    'logM_1': [13.6, 15.1],
    'sigma': [-2.99, 0.96],
    'alpha': [0.3, 1.48],
    'kappa': [0., 0.99],
    'alpha_c': [0., 0.61],
    'alpha_s': [0.58, 1.49],
    's': [-0.98, 1.],
    'A_cen': [-0.99, 0.93],
    'A_sat': [-1., 1.],
    'B_cen': [-0.67, 0.2],
    'B_sat': [-0.97, 0.99],
}

# fixed_parameters = {
#     'A_cen': 0.0,
#     'A_sat': 0.0,
#     's': 0.0,
#     'w0_fld': -1.0,
#     'wa_fld': 0.0,
#     'nrun': 0.0,
#     'N_ur': 2.0328,
#     'omega_b': 0.02237,
#     'alpha_c': 0.0,
#     'alpha_s': 1.0,
# }

priors = {**AbacusSummit().prior, **Yuan23().prior}
filters = {'multipoles': [0, 2], 'statistics': ['quantile_data_correlation']}
fixed_params = ['A_cen', 'A_sat', 'w0_fld', 'wa_fld', 'nrun', 'N_ur',]


smins = [0, 5, 10, 20, 40, 60, 80]
for smin in smins:
    # for statistic in ['dsc_conf', 'dsc_fourier']:
    for statistic in ['dsc_conf']:

        covariance_matrix, n_sim = read_covariance(statistic=statistic, filters=filters)
        print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

        # load the data
        lhc_x, lhc_y, lhc_x_names, model_filters = read_lhc(statistic=statistic, filters=filters)
        print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

        idxs = [30]

        for idx_fit in idxs:
            print(f'Fitting HOD {idx_fit}')

            fixed_params_dict = {key: lhc_x[idx_fit, lhc_x_names.index(key)]
                                for key in fixed_params}

            # load the model
            model = read_model(statistic=statistic)
            nn_model, nn_params = model.to_jax()
            with torch.no_grad():
                pred_y = model.get_prediction(torch.Tensor(lhc_x))
                pred_y = pred_y.numpy().reshape(pred_y.shape[0], *model_filters.shape)
                pred_y = pred_y[:, ~model_filters]
            lhc_test_y = lhc_y[:600]
            pred_test_y = pred_y[:600]
            emulator_error = get_emulator_error(lhc_test_y, pred_test_y)
            covariance_matrix += np.diag(emulator_error**2)


        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots()
        #     ax.errorbar(np.arange(len(lhc_test_y[idx_fit])),
        #                 lhc_test_y[idx_fit],
        #                 np.sqrt(np.diag(covariance_matrix)),
                        # marker='o', ls='', ms=2.0)
            # ax.plot(pred_test_y[idx_fit], ls='-')
            # plt.show()

            # apply correction to the covariance matrix
            correction = get_covariance_correction(
                n_s=n_sim,
                n_d=len(covariance_matrix),
                n_theta=len(lhc_x_names) - len(fixed_params),
                correction_method='percival',
            )
            print(f'Number of simulations: {n_sim}')
            print(f'Number of data points: {len(covariance_matrix)}')
            print(f'Number of parameters: {len(lhc_x_names) - len(fixed_params)}')
            print(f'Covariance correction factor: {correction}')
            covariance_matrix *= correction

            precision_matrix = np.linalg.inv(covariance_matrix)

            hmc = HMC(
                observation=lhc_test_y[idx_fit],
                # observation=pred_y[idx_fit],
                precision_matrix=precision_matrix,
                nn_theory_model=nn_model,
                nn_parameters=nn_params,
                fixed_parameters=fixed_params_dict,
                priors=priors,
                model_filters=model_filters,
                # model_filters=filters,
            )

            posterior = hmc()
            posterior_array = np.stack(list(posterior.values()), axis=0)

            # remove fixed parameters from the posterior
            idx = [list(posterior.keys()).index(param) for param in fixed_params]
            posterior_array = np.delete(posterior_array, idx, axis=0)
            names = np.delete(list(posterior.keys()), idx)


            cout = {
                'samples': posterior_array.T,
                'weights': np.ones(posterior_array.shape[-1]),
                'names': names,
                'ranges': getdist_ranges,
            }

            save_dir = f'/pscratch/sd/e/epaillas/emc/posteriors/minimal/{statistic}_cross/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_dir) / f'posterior_cosmo+hod_idx{idx_fit}_smin{smin}.npy'
            np.save(save_fn, cout)
