import numpy as np
from sunbird.inference.pocomc import PocoMCSampler
from sunbird.inference.priors import Yuan23, AbacusSummit
import argparse
from acm.data.io_tools import *
import torch
from sunbird import setup_logging


def get_covariance_correction(n_s, n_d, n_theta=None, correction_method='percival'):
    if correction_method == 'percival':
        B = (n_s - n_d - 2) / ((n_s - n_d - 1)*(n_s - n_d - 4))
        return (n_s - 1)*(1 + B*(n_d - n_theta))/(n_s - n_d + n_theta - 1)
    elif correction_method == 'hartlap':
        return (n_s - 1)/(n_s - n_d - 2)

def get_priors(cosmo=True, hod=True):
    stats_module = 'scipy.stats'
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

def save_handle():
    save_dir = '/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus'
    save_dir = Path(save_dir) / f'c{args.cosmo_idx:03}_hod{args.hod_idx:03}/test/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    slice_str = ''
    select_str = ''
    statistic = '+'.join(statistics)
    if slice_filters:
        for key, value in slice_filters.items():
            slice_str += f'_{key}{value[0]:.2f}-{value[1]:.2f}'
    return Path(save_dir) / f'{statistic}{select_str}{slice_str}'


parser = argparse.ArgumentParser()
parser.add_argument("--cosmo_idx", type=int, default=0)
parser.add_argument("--hod_idx", type=int, default=30)

args = parser.parse_args()
setup_logging()

# set up the inference
priors, ranges, labels = get_priors(cosmo=True, hod=True)
select_filters = {'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx,
    'statistics': ['quantile_data_power'], 'quantiles': [0, 4]}
slice_filters = {}
fixed_params = ['w0_fld', 'wa_fld', 'N_ur', 'nrun']
add_emulator_error = True
statistics = ['wst', 'number_density']


print(f'Fitting {statistics} with cosmo_idx={args.cosmo_idx} and hod_idx={args.hod_idx}')

# load the covariance matrix
covariance_matrix, n_sim = read_covariance(statistics=statistics,
                                            select_filters=select_filters,
                                            slice_filters=slice_filters)
print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

# load the data
data_x, data_y, data_x_names, model_filters = read_lhc(statistics=statistics,
                                                    select_filters=select_filters,
                                                    slice_filters=slice_filters,
                                                    return_mask=True)
print(f'Loaded LHC x with shape: {data_x.shape}')
print(f'Loaded LHC y with shape {data_y.shape}')

fixed_params = {key: data_x[data_x_names.index(key)]
                    for key in fixed_params}

# load the model
models = read_model(statistics=statistics)
nn_model = [model.to_jax()[0] for model in models]
nn_params = [model.to_jax()[1] for model in models]

with torch.no_grad():
    pred_y = [model.get_prediction(torch.Tensor(data_x), filters=model_filter).numpy()
                for model, model_filter in zip(models, model_filters)]
    pred_y = np.concatenate(pred_y)

if add_emulator_error:
    emulator_error = read_emulator_error(statistics, select_filters=select_filters,
                                        slice_filters=slice_filters)
    print(f'Loaded emulator error with shape: {emulator_error.shape}')
    covariance_matrix += np.diag(emulator_error**2)

# apply correction to the covariance matrix
correction = get_covariance_correction(
    n_s=n_sim,
    n_d=len(covariance_matrix),
    n_theta=len(data_x_names) - len(fixed_params),
    correction_method='percival',
)
print(f'Number of simulations: {n_sim}')
print(f'Number of data points: {len(covariance_matrix)}')
print(f'Number of parameters: {len(data_x_names) - len(fixed_params)}')
print(f'Covariance correction factor: {correction}')
covariance_matrix *= correction
precision_matrix = np.linalg.inv(covariance_matrix)

sampler = PocoMCSampler(
    observation=data_y,
    precision_matrix=precision_matrix,
    theory_model=models,
    fixed_parameters=fixed_params,
    priors=priors,
    ranges=ranges,
    labels=labels,
    model_filters=model_filters,
)

sampler(vectorize=True)

# plot and save results
markers = {key: data_x[data_x_names.index(key)] for key in data_x_names if key not in fixed_params}

sampler.plot_bestfit(save_fn=f'{save_handle()}_model_maxl.pdf', model='maxl')
sampler.plot_bestfit(save_fn=f'{save_handle()}_model_mean.pdf', model='mean')
sampler.plot_triangle(save_fn=f'{save_handle()}_triangle.pdf', thin=128,
                    markers=markers, title_limit=1)
sampler.plot_trace(save_fn=f'{save_handle()}_trace.pdf', thin=128)
sampler.save_chain(f'{save_handle()}_chain.npy', metadata={'markers': markers})
sampler.save_table(f'{save_handle()}_stats.txt')