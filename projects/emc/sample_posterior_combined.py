import scipy.stats as st
import numpy as np
from acm.observables import BaseObservable
from sunbird.emulators import FCN
from pycorr import TwoPointCorrelationFunction
from pathlib import Path
from numpyro import distributions as dist
from sunbird.inference.hamiltonian import HMC
from sunbird.inference.priors import Yuan23, AbacusSummit
import pandas
import glob
import torch
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import sys


def read_lhc(statistics):
    lhc_y_all = []
    for statistic in statistics: 
        data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/{statistic}/cosmo+hod/z0.5/yuan23_prior/ph000/seed0/'
        data_fn = Path(data_dir) / f'{statistic}_lhc.npy'
        data = np.load(data_fn, allow_pickle=True).item()
        lhc_x = data['lhc_x']
        lhc_y = data['lhc_y']
        lhc_x_names = data['lhc_x_names']
        lhc_y_all.append(lhc_y)
    lhc_y = np.concatenate(lhc_y_all, axis=1)
    return lhc_x, lhc_y, lhc_x_names

def read_model(statistics):
    models = []
    for statistic in statistics:
        if statistic == 'wp':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/wp/cosmo+hod/jul10_trans/last-v30.ckpt'
        if statistic == 'pk':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/pk/jun25_leaveout_0/last.ckpt'
        elif statistic == 'tpcf':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/tpcf/cosmo+hod/jul9/last.ckpt'
        elif statistic == 'dsc_conf':
            # checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/density_split/jun2_9_leaveout_0/last.ckpt'
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_conf/jun27_leaveout_0/last-v1.ckpt'
        elif statistic == 'dsc_fourier':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_fourier/jun24_leaveout_0/last-v1.ckpt'
        elif statistic == 'wst':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/wst/jun27_leaveout_0/last.ckpt'
        elif statistic == 'minkowski':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/minkowski/Minkowski-best-model-epoch=276-val_loss=0.02366.ckpt'
        model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
        model.eval()
        models.append(model)
    return models


def read_covariance(statistics):
    y_all = []
    for statistic in statistics:
        if statistic == 'minkowski':
            data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/minkowski/z0.5/yuan23_prior/c000_ph000/seed0/'
            data_fn = Path(data_dir) / f'minkowski_lhc.npy'
            y = np.load(data_fn, allow_pickle=True).item()['abacus_small_y']
        elif statistic in ['dsc_conf', 'w_p']:
            data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/{statistic}/z0.5/yuan23_prior/c000_ph000/seed0/'
            data_fn = Path(data_dir) / f'{statistic}_lhc.npy'
            y = np.load(data_fn, allow_pickle=True).item()['cov_y']
        else:
            data_dir = f'/pscratch/sd/e/epaillas/emc/covariance_sets/{statistic}/z0.5/yuan23_prior'
            data_fn = Path(data_dir) / f'{statistic}_cov.npy'
            y = np.load(data_fn, allow_pickle=True)
        y_all.append(y)
    y = np.concatenate(y_all, axis=1)
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

getdist_priors = {
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

fixed_parameters = {
    'A_cen': 0.0,
    'A_sat': 0.0,
    's': 0.0,
    'w0_fld': -1.0,
    'wa_fld': 0.0,
    'nrun': 0.0,
    'N_ur': 2.0328,
    'omega_b': 0.02237,
}

priors = {**AbacusSummit().prior, **Yuan23().prior}

statistics = ['wp', 'tpcf']

for idx_fit in range(10):

    covariance_matrix, n_sim = read_covariance(statistics=statistics)
    print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

    lhc_x, lhc_y, lhc_x_names = read_lhc(statistics=statistics)
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

    models = read_model(statistics=statistics)
    nn_model = [model.to_jax()[0] for model in models]
    nn_params = [model.to_jax()[1] for model in models]

    with torch.no_grad():
        pred_y = [model.get_prediction(torch.Tensor(lhc_x)).numpy() for model in models]
        pred_y = np.concatenate(pred_y, axis=1)
        # pred_y = pred_y.numpy()

    lhc_test_y = lhc_y[:600]
    pred_test_y = pred_y[:600]
    emulator_error = get_emulator_error(lhc_test_y, pred_test_y)

    covariance_matrix += np.diag(emulator_error**2)

    correction = get_covariance_correction(
        n_s=n_sim,
        n_d=len(covariance_matrix),
        n_theta=len(lhc_x_names),
        correction_method='percival',
    )
    print(f'Number of simulations: {n_sim}')
    print(f'Number of data points: {len(covariance_matrix)}')
    print(f'Number of parameters: {len(lhc_x_names)}')
    print(f'Covariance correction factor: {correction}')

    covariance_matrix *= correction
    precision_matrix = np.linalg.inv(covariance_matrix)

    hmc = HMC(
        observation=lhc_test_y[idx_fit],
        # observation=pred_y[idx_fit],
        precision_matrix=precision_matrix,
        nn_theory_model=nn_model,
        nn_parameters=nn_params,
        fixed_parameters=fixed_parameters,
        priors=priors,
    )

    posterior = hmc()
    posterior_array = np.stack(list(posterior.values()), axis=0)

    # remove fixed parameters from the posterior
    fixed_parameters_keys = list(fixed_parameters.keys())
    idx = [list(posterior.keys()).index(key) for key in fixed_parameters_keys]
    posterior_array = np.delete(posterior_array, idx, axis=0)
    names = np.delete(list(posterior.keys()), idx)


    cout = {
        'samples': posterior_array.T,
        'weights': np.ones(posterior_array.shape[-1]),
        'names': names,
        'ranges': getdist_priors,
    }

    np.save(f'posterior_cosmo+hod_wp_tpcf_idx{idx_fit}.npy', cout)



    # # chain_getdist = MCSamples(
    # #         samples=posterior_array.T,
    # #         weights=np.ones(posterior_array.shape[-1]),
    # #         names=list(posterior.keys()),
    # #         ranges=getdist_priors,
    # #     )

    # # g = plots.get_subplot_plotter()
    # # g.settings.constrained_layout = True
    # # g.settings.axis_marker_lw = 1.0
    # # g.settings.axis_marker_ls = ":"
    # # g.settings.title_limit_labels = False
    # # g.settings.axis_marker_color = "k"
    # # g.settings.legend_colored_text = True
    # # g.settings.figure_legend_frame = False
    # # g.settings.linewidth_contour = 1.0
    # # g.settings.legend_fontsize = 22
    # # g.settings.axes_fontsize = 16
    # # g.settings.axes_labelsize = 20
    # # g.settings.axis_tick_x_rotation = 45
    # # g.settings.axis_tick_max_labels = 6

    # # g.triangle_plot(
    # #     roots=[chain_getdist],
    # #     filled=True,
    # #     markers=dict(zip(lhc_x_names, lhc_x[idx_fit]))
    # # )
    # # plt.savefig(f'posterior_{statistic}_hod{idx_fit}.pdf', bbox_inches='tight')
    # # # plt.show()