import numpy as np
from pathlib import Path
from sunbird.inference.hamiltonian import HMCSampler
from sunbird.inference.priors import Yuan23, AbacusSummit
import torch
import numpyro
from numpyro.infer import init_to_mean
import matplotlib.pyplot as plt
import sys
from acm.data.io_tools import *


def get_covariance_correction(n_s, n_d, n_theta=None, correction_method='percival'):
    if correction_method == 'percival':
        B = (n_s - n_d - 2) / ((n_s - n_d - 1)*(n_s - n_d - 4))
        return (n_s - 1)*(1 + B*(n_d - n_theta))/(n_s - n_d + n_theta - 1)
    elif correction_method == 'hartlap':
        return (n_s - 1)/(n_s - n_d - 2)

def get_priors(cosmo=True, hod=True):
    stats_module = 'numpyro.distributions'
    priors, ranges, labels = {}, {}, {}
    if cosmo:
        priors.update(AbacusSummit(stats_module).priors)
        ranges.update(AbacusSummit(stats_module).ranges)
        labels.update(AbacusSummit(stats_module).labels)
    if hod:
        priors.update(Yuan23(stats_module).priors)
        ranges.update(Yuan23(stats_module).ranges)
        labels.update(Yuan23(stats_module).labels)
    return priors, ranges, labels

def get_save_fn(statistic, mock_idx, kmin, kmax, smin, smax):
    save_dir = f'/pscratch/sd/e/epaillas/emc/posteriors/hmc/{statistic}/oct8'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    scales_str = ''
    if slice_filters:
        for key, value in slice_filters.items():
            scales_str += f'_{key}{value[0]:.1f}_{key}{value[1]:.1f}'
    return Path(save_dir) / f'chain_idx{mock_idx}{scales_str}.npy'


priors, ranges, labels = get_priors(cosmo=True, hod=True)
# select_filters = {'multipoles': [0, 2], 'statistics': ['quantile_data_correlation']}
select_filters = {'cosmo_idx': 0, 'hod_idx': 30}
fixed_params = ['w0_fld', 'wa_fld', 'nrun', 'N_ur', 'A_cen', 'A_sat']
add_emulator_error = True
# statistics = ['wp', 'dsc_conf', 'tpcf']
statistics = ['pk', 'number_density']

# smins = [0]
num_chains = 1
smax = 152
kmin = 0.0
# smins = [0, 5, 10, 20, 40, 60, 80, 100]
smins = [12.5]
kmaxs = [0.2]
for smin in smins:
    for kmax in kmaxs:
        slice_filters = {'k': [kmin, kmax]}
        # slice_filters = {}

        covariance_matrix, n_sim = read_covariance(statistics=statistics,
                                                   select_filters=select_filters,
                                                   slice_filters=slice_filters)
        print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

        # load the data
        lhc_x, lhc_y, lhc_x_names, model_filters = read_lhc(statistics=statistics,
                                                            select_filters=select_filters,
                                                            slice_filters=slice_filters,
                                                            return_mask=True)
        print(f'Loaded LHC x with shape: {lhc_x.shape}')
        print(f'Loaded LHC y with shape {lhc_y.shape}')

        # lhc_test_y = lhc_y[:600]

        # idxs = [30, 199, 330, 438]
        # idxs = [30]

    #     print(f'Fitting HOD {mock_idx}')

        fixed_params_dict = {key: lhc_x[lhc_x_names.index(key)]
                            for key in fixed_params}
    #     # fixed_params_dict['N_ur'] = 3.046
        print(f'Fixed parameters: {fixed_params_dict}')

        # load the model
        models = read_model(statistics=statistics)
        nn_model = [model.to_jax()[0] for model in models]
        nn_params = [model.to_jax()[1] for model in models]

        if add_emulator_error:
            emulator_error = read_emulator_error(statistics, select_filters=select_filters,
                                                slice_filters=slice_filters)
            print(f'Loaded emulator error with shape: {emulator_error.shape}')
            print(emulator_error.shape)
            covariance_matrix += np.diag(emulator_error**2)

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
        # covariance_matrix *= correction

        precision_matrix = np.linalg.inv(covariance_matrix)

    #     # hmc = HMCSampler(
    #     #     observation=lhc_test_y[mock_idx],
    #     #     # observation=pred_y[mock_idx],
    #     #     precision_matrix=precision_matrix,
    #     #     nn_theory_model=nn_model,
    #     #     nn_parameters=nn_params,
    #     #     fixed_parameters=fixed_params_dict,
    #     #     priors=priors,
    #     #     ranges=ranges,
    #     #     labels=labels,
    #     #     model_filters=model_filters,
    #     # )
    #     # numpyro.set_host_device_count(num_chains)

    #     # save_fn = get_save_fn(statistic='+'.join(statistics),
    #     #                       kmin=kmin, kmax=kmax,
    #     #                       smin=smin, smax=smax,
    #     #                       mock_idx=mock_idx)

    #     # posterior = hmc(num_warmup=4000, num_samples=4000, dense_mass=True,
    #     # # posterior = hmc(num_warmup=500, num_samples=2000, dense_mass=True,
    #     #                 target_accept_prob=0.95,
    #     #                 num_chains=num_chains, save_fn=save_fn)
