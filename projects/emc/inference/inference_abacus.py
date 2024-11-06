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
import argparse


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

def get_save_fn(save_dir,):
    save_dir.mkdir(parents=True, exist_ok=True)
    slice_str = ""
    select_str = ""
    if slice_filters:
        for key, value in slice_filters.items():
            slice_str += f"_{key}{value[0]:.2f}-{value[1]:.2f}"
    if select_filters:
        for key, value in select_filters.items():
            select_str += "_" + f"{key}{value}".replace("_", "-")
    return Path(save_dir) / f"chain{select_str}{slice_str}.npy"

def get_posterior(
    statistics,
    select_filters,
    slice_filters,
    save_dir,
    fixed_params,
    add_emulator_error,
    model_dir_dict,

):
    covariance_matrix, n_sim = read_covariance(
        statistics=statistics,
        select_filters=select_filters,
        slice_filters=slice_filters,
    )
    print(f"Loaded covariance matrix with shape: {covariance_matrix.shape}")

    print('select filters = ', select_filters)
    print('slice filters = ', slice_filters)
    # load the data
    data_x, data_y, data_x_names, model_filters = read_lhc(
        statistics=statistics,
        select_filters=select_filters,
        slice_filters=slice_filters,
        return_mask=True,
    )
    print(f"Loaded LHC x with shape: {data_x.shape}")
    print(f"Loaded LHC y with shape {data_y.shape}")

    if fixed_params is not None:
        fixed_params = {key: data_x[data_x_names.index(key)] for key in fixed_params}
        #     # fixed_params_dict['N_ur'] = 3.046
        print(f"Fixed parameters: {fixed_params}")
        n_theta = len(data_x_names) - len(fixed_params)
    else:
        print(f"No fixed parameters")
        fixed_params = {}
        n_theta = len(data_x_names)

    # load the model
    models = read_model(statistics=statistics, model_dir_dict=model_dir_dict,)
    nn_model = [model.to_jax()[0] for model in models]
    nn_params = [model.to_jax()[1] for model in models]

    if add_emulator_error:
        emulator_error = read_emulator_error(
            statistics, select_filters=select_filters, slice_filters=slice_filters
        )
        print(f"Loaded emulator error with shape: {emulator_error.shape}")
        covariance_matrix += np.diag(emulator_error**2)

    # apply correction to the covariance matrix
    correction = get_covariance_correction(
        n_s=n_sim,
        n_d=len(covariance_matrix),
        n_theta=n_theta,
        correction_method="percival",
    )
    print(f"Number of simulations: {n_sim}")
    print(f"Number of data points: {len(covariance_matrix)}")
    print(f"Number of parameters: {n_theta}")
    print(f"Covariance correction factor: {correction}")
    covariance_matrix *= correction

    precision_matrix = np.linalg.inv(covariance_matrix)

    hmc = HMCSampler(
        observation=data_y,
        precision_matrix=precision_matrix,
        nn_theory_model=nn_model,
        nn_parameters=nn_params,
        fixed_parameters=fixed_params,
        priors=priors,
        ranges=ranges,
        labels=labels,
        model_filters=model_filters,
    )
    numpyro.set_host_device_count(num_chains)

    save_fn = get_save_fn(save_dir=save_dir / "+".join(statistics))

    metadata = {
        # 'select_filters': select_filters,
        # 'slice_filters': slice_filters,
        # 'n_sim': n_sim,
        # 'correction': correction,
        # 'emulator_error': emulator_error,
        "covariance_matrix": covariance_matrix,
        # 'precision_matrix': precision_matrix,
        "fixed_params": fixed_params,
        "true_params": dict(zip(data_x_names, data_x)),
        "data": data_y,
        # 'model_filters': model_filters,
    }

    return hmc(
        num_warmup=250,
        num_samples=1000,
        dense_mass=True,
        #target_accept_prob=0.95,
        num_chains=num_chains,
        save_fn=save_fn,
        metadata=metadata,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmo_idx", type=int, default=0)
    parser.add_argument("--hod_idx", type=int, default=30)
    parser.add_argument('--fixed_params', nargs='+', default=None, help='List of fixed parameters.')
    parser.add_argument('--no_emulator_error', action='store_false', help='Flag to disable adding emulator error.')
    parser.add_argument('--statistics', nargs='+', default=["pk", "number_density"], help='List of statistics to use.')
    parser.add_argument('--model_dirs', nargs='+', default=None, help='List of statistics to use.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the output.')

    args = parser.parse_args()


    priors, ranges, labels = get_priors(cosmo=True, hod=True)
    select_filters = {"cosmo_idx": args.cosmo_idx, "hod_idx": args.hod_idx}
    statistics = args.statistics
    add_emulator_error = args.no_emulator_error
    if args.model_dirs is not None:
        model_dir_dict = {stat: Path(model_dir) for stat, model_dir in zip(statistics, args.model_dirs)}
    else:
        model_dir_dict = None

    num_chains = 1
    slice_filters = None
    posterior = get_posterior(
        statistics=statistics,
        select_filters=select_filters,
        slice_filters=slice_filters,
        add_emulator_error=add_emulator_error,
        fixed_params = args.fixed_params,
        save_dir = Path(args.save_dir) / "+".join(args.statistics),
        model_dir_dict=model_dir_dict,
    )

