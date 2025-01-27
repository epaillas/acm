import numpy as np
from sunbird.inference.pocomc import PocoMCSampler
from sunbird.inference.priors import Yuan23, AbacusSummit
import argparse
from acm.data.io_tools import *
from sunbird import setup_logging
import matplotlib.pyplot as plt


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
    save_dir = '/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/diffsky'
    save_dir = Path(save_dir) / f'base_lcdm/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    slice_str = ''
    select_str = ''
    statistic = '+'.join(statistics)
    if slice_filters:
        for key, value in slice_filters.items():
            slice_str += f'_{key}{value[0]:.2f}-{value[1]:.2f}'
    # if select_filters:
    #     for key, value in select_filters.items():
    #         if key in ['cosmo_idx', 'hod_idx']:
    #             select_str += '_' + f'{key}{value}'.replace('_', '-').replace("'", "")
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
fixed_params = ['w0_fld', 'wa_fld', 'N_ur', 'nrun']
add_emulator_error = True
# statistics = ['wst', 'minkowski', 'dsc_pk', 'pk', 'number_density']
statistics = ['wst', 'number_density']
kmin, kmax = 0.0, 0.5
# slice_filters = {'k': (kmin, kmax)}
slice_filters = {}

# load the covariance matrix
covariance_matrix, n_sim = read_covariance(statistics=statistics,
                                            select_filters=select_filters,
                                            slice_filters=slice_filters,
                                            volume_factor=8)
print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

# load diffsky
diffsky_y, model_filters = read_diffsky(statistics=statistics,
                                        select_filters=select_filters,
                                        slice_filters=slice_filters,
                                        return_mask=True)
print(f'Loaded diffsky with shape: {diffsky_y.shape}')

# load abacus lhc
lhc_x, lhc_y, lhc_x_names, model_filters = read_lhc(statistics=statistics,
                                                    select_filters=select_filters,
                                                    slice_filters=slice_filters,
                                                    return_mask=True)
print(f'Loaded LHC x with shape: {lhc_x.shape}')
print(f'Loaded LHC y with shape {lhc_y.shape}')

fixed_params = {key: lhc_x[lhc_x_names.index(key)]
                    for key in fixed_params}

# load the model
models = read_model(statistics=statistics)
nn_model = [model.to_jax()[0] for model in models]
nn_params = [model.to_jax()[1] for model in models]

if add_emulator_error:
    emulator_error = read_emulator_error(statistics, select_filters=select_filters,
                                         slice_filters=slice_filters)
    print(f'Loaded emulator error with shape: {emulator_error.shape}')
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
covariance_matrix *= correction
precision_matrix = np.linalg.inv(covariance_matrix)

sampler = PocoMCSampler(
    observation=diffsky_y,
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
markers = {
    'Omega_m': 0.3089,
    'omega_cdm': 0.1188,
    'omega_b': 0.02230,
    'h': 0.6774,
    'n_s': 0.9667,
    'sigma8_m': 0.8147
}
sampler.plot_bestfit(save_fn=f'{save_handle()}_model_maxl.pdf', model='maxl')
sampler.plot_bestfit(save_fn=f'{save_handle()}_model_mean.pdf', model='mean')
sampler.plot_triangle(save_fn=f'{save_handle()}_triangle.pdf', thin=128,
                      markers=markers, title_limit=1)
sampler.plot_trace(save_fn=f'{save_handle()}_trace.pdf', thin=128)
sampler.save_chain(f'{save_handle()}_chain.npy', metadata={'markers': markers})
sampler.save_table(f'{save_handle()}_stats.txt')