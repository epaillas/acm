import numpy as np
from sunbird.inference.minuit import MinuitProfiler
from sunbird.inference.priors import Yuan23, AbacusSummit
import argparse
import acm.observables.emc as emc
from acm.data.io_tools import *
from sunbird import setup_logging


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
    save_dir = '/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/feb11/'
    save_dir = Path(save_dir) / f'c{args.cosmo_idx:03}_hod{args.hod_idx:03}/lcdm/projection_effects/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    slice_str = ''
    select_str = ''
    # statistic = '+'.join(statistics)
    statistic = statistics
    slice_filters = observable.slice_filters
    # if slice_filters:
    #     for key, value in slice_filters.items():
    #         slice_str += f'_{key}{value[0]:.2f}-{value[1]:.2f}'
    # return Path(save_dir) / f'{statistic}_ell2'
    return Path(save_dir) / f'{statistic}'


parser = argparse.ArgumentParser()
parser.add_argument("--cosmo_idx", type=int, default=0)
parser.add_argument("--hod_idx", type=int, default=30)

args = parser.parse_args()
setup_logging()

# set up the inference
priors, ranges, labels = get_priors(cosmo=True, hod=True)
# fixed_params = ['w0_fld', 'wa_fld', 'nrun', 'N_ur', 'sigma', 'kappa', 'alpha', 's', 'A_cen', 'A_sat', 'B_cen', 'B_sat', 'alpha_s', 'alpha_c']
fixed_params = ['w0_fld', 'wa_fld', 'nrun', 'N_ur']
add_emulator_error = True

# load observables with their custom filters
observable = emc.GalaxyPowerSpectrumMultipoles(
        select_filters={
            'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx,
        },
        slice_filters={
        }
    )

statistics = observable.stat_name
print(f'Fitting {statistics} with cosmo_idx={args.cosmo_idx} and hod_idx={args.hod_idx}')

# load the data
data_x = observable.lhc_x
data_x_names = observable.lhc_x_names
data_y = observable.lhc_y
print(f'Loaded LHC x with shape: {data_x.shape}')
print(f'Loaded LHC y with shape {data_y.shape}')

# model prediction at the true cosmology
pred_y = observable.get_model_prediction(data_x)
print(f'Loaded model prediction with shape: {pred_y.shape}')

# load the covariance matrix
covariance_matrix = observable.get_covariance_matrix(divide_factor=64)
print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

# load emulator error
if add_emulator_error:
    emulator_error = observable.get_emulator_error()
    covariance_matrix += np.diag(emulator_error**2)

# get the debiased inverse
correction = observable.get_covariance_correction(
    n_s=len(observable.small_box_y),
    n_d=len(covariance_matrix),
    n_theta=len(data_x_names) - len(fixed_params),
    method='percival',
)
precision_matrix = np.linalg.inv(correction * covariance_matrix)

fixed_params = {key: data_x[data_x_names.index(key)]
                    for key in fixed_params}

# load the model
models = observable.model
model_coordinates = observable.coords_model

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/feb11/c000_hod030/lcdm/projection_effects'
data_fn = Path(data_dir) / f"pk_chain.npy"
data = np.load(data_fn, allow_pickle=True).item()
maxlike = data['samples'][data['log_likelihood'].argmax()]
# turn into dictionary
maxlike = {data['names'][i]: maxlike[i] for i in range(len(data['names']))}
mean = data['samples'].mean(axis=0)
mean = {data['names'][i]: mean[i] for i in range(len(data['names']))}

std = data['samples'].std(axis=0)
std = {data['names'][i]: std[i] for i in range(len(data['names']))}

# start_ranges = {key: [mean[key] - 1*std[key], mean[key] + 1*std[key]] for key in mean}
start_ranges = {key: [maxlike[key] - 0.5*std[key], maxlike[key] + 0.5*std[key]] for key in maxlike}
print(start_ranges)

# run the profiler
minuit = MinuitProfiler(
    observation=pred_y,
    precision_matrix=precision_matrix,
    theory_model=models,
    fixed_params=fixed_params,
    priors=priors,
    ranges=ranges,
    labels=labels,
    # start_ranges=start_ranges,
    slice_filters=observable.slice_filters,
    select_filters=observable.select_filters,
    coordinates=model_coordinates,
)

profiles = minuit.minimize(nstart=50, autodiff=True)

print(profiles[0])

np.save(f'{save_handle()}_profile.npy', profiles)
