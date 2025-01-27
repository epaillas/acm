import numpy as np
from sunbird.inference.emcee import EmceeSampler
from sunbird.inference.priors import Yuan23, AbacusSummit
import argparse
from acm.data.io_tools import *
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


parser = argparse.ArgumentParser()
parser.add_argument("--statistic", type=str, default='pk')
parser.add_argument("--cosmo_idx", type=int, default=0)
parser.add_argument("--hod_idx", type=int, default=30)

args = parser.parse_args()
setup_logging()

# set up the inference
priors, ranges, labels = get_priors(cosmo=True, hod=True)
select_filters = {'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx,
}
fixed_params = ['w0_fld', 'wa_fld', 'N_ur', 'nrun', 'omega_b']
add_emulator_error = True
statistics = ['dsc_fourier', 'wst', 'number_density']
kmin, kmax = 0.0, 0.5
slice_filters = {'k': [kmin, kmax]}

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

# run the sampler
sampler = EmceeSampler(
    observation=data_y,
    precision_matrix=precision_matrix,
    theory_model=models,
    fixed_parameters=fixed_params,
    priors=priors,
    ranges=ranges,
    labels=labels,
    model_filters=model_filters,
)
sampler(nwalkers=128, niter=50_000, vectorize=True, burnin=200,
        moves=[emcee.moves.DEMove(), emcee.moves.KDEMove()])

# plot and save results
sampler.triangle_plot(save_fn='emcee_triangle.pdf', thin=128,
                      params=['omega_cdm', 'sigma8_m', 'n_s'],
                      markers={'omega_cdm': 0.12, 'sigma8_m': 0.807952, 'n_s': 0.9649})
sampler.trace_plot(save_fn='emcee_trace.pdf', thin=128)
sampler.save_chain(save_fn='emcee_chain.npy', thin=128)