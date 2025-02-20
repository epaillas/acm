import acm.projects.bgs as project
from acm.data.io_tools import get_covariance_correction
from acm.observables import BaseCombinedObservable as CombinedObservable
from sunbird.inference.pocomc import PocoMCSampler

import numpy as np
from pathlib import Path
from sunbird import setup_logging
setup_logging()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cosmo_idx", type=int, default=0)
parser.add_argument("--hod_idx", type=int, default=96)
args = parser.parse_args()

# Set up the inference (NOTE : hardcoded values !)
fixed_parameters = ['omega_b', 'w0_fld', 'wa_fld', 'nrun', 'N_ur', 'B_cen', 'B_sat']
add_emulator_error = True

save_dir = '/pscratch/sd/s/sbouchar/acm/bgs/chains/' 
save_dir = Path(save_dir) / f'c{args.cosmo_idx:03}_hod{args.hod_idx:03}/'

# Load observables with their custom filters
observable = CombinedObservable([
    project.GalaxyCorrelationFunctionMultipoles(
        select_filters={
            'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx,
        },
        slice_filters={
        }
    ),
    project.DensitySplitCorrelationFunctionMultipoles(
        select_filters={
            'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx,
        },
        slice_filters={
        }
    ),
])

#%% From this point, everything is automated !

# Define utility functions
def get_save_handle(save_dir: str|Path, observable: CombinedObservable):
    """
    Creates a handle for saving the results of the inference that includes the statistics and filters used.

    Parameters
    ----------
    save_dir : str
        Directory where the results will be saved.
    observable : CombinedObservable
        The observable object used for the inference.

    Returns
    -------
    Path
        The handle for saving the results, to be completed with the file extension.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    statistics = observable.stat_name
    slice_filters = observable.slice_filters
    
    statistic_handles = []
    for i, statistic in enumerate(statistics):
        statistic_handles.append(statistic)
        if slice_filters[i]:
            for key, value in slice_filters[i].items():
                statistic_handles[-1] += f'_{key}_{value[0]:.2f}-{value[1]:.2f}'
        # TODO : add select filters to the handle ?
    statistic = '+'.join(statistic_handles)
    return Path(save_dir) / f'{statistic}'

statistics = observable.stat_name
print(f'Fitting {statistics} with cosmo_idx={args.cosmo_idx} and hod_idx={args.hod_idx}')

# Load the data
data_x = observable.lhc_x
data_x_names = observable.lhc_x_names
data_y = observable.lhc_y
print(f'Loaded LHC x with shape: {data_x.shape}')
print(f'Loaded LHC y with shape {data_y.shape}')

# Order of the parameters needs to be the same as the data_x_names (also removes unused keys)
priors, ranges, labels = project.get_priors(cosmo=True, hod=True)
priors = {key: priors[key] for key in observable.lhc_x_names}

# Load the covariance matrix
covariance_matrix = observable.get_covariance_matrix(volume_factor=64)
print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

# Load emulator error
if add_emulator_error:
    emulator_error = observable.emulator_error
    covariance_matrix += np.diag(emulator_error**2)

# Get the debiased inverse
correction = get_covariance_correction(
    n_s=len(observable.covariance_y),
    n_d=len(covariance_matrix),
    n_theta=len(data_x_names) - len(fixed_parameters),
    method='percival',
)
precision_matrix = np.linalg.inv(correction * covariance_matrix)

fixed_parameters = {key: data_x[data_x_names.index(key)]
                    for key in fixed_parameters}

# Load the model
models = observable.model

# Sample the posterior
sampler = PocoMCSampler(
    observation=data_y,
    precision_matrix=precision_matrix,
    theory_model=models,
    fixed_parameters=fixed_parameters,
    priors=priors,
    ranges=ranges,
    labels=labels,
    # NOTE : requires to comment lines 123-133 of the PocoMC class (othetwise it expects list and dicts that are empty by default :/ )
)

# sampler(vectorize=True, n_total=10_000)
sampler(vectorize=True, n_total=4096)

# Plot and save results
markers = {key: data_x[data_x_names.index(key)] for key in data_x_names if key not in fixed_parameters}

save_handle = get_save_handle(save_dir, observable)
sampler.plot_triangle(save_fn=f'{save_handle}_triangle.pdf', thin=128,
                    markers=markers, title_limit=1)
sampler.plot_trace(save_fn=f'{save_handle}_trace.pdf', thin=128)
sampler.save_chain(f'{save_handle}_chain.npy', metadata={'markers': markers, 'fixed_parameters': fixed_parameters})
sampler.save_table(f'{save_handle}_stats.txt')
sampler.plot_bestfit(save_fn=f'{save_handle}_model_maxl.pdf', model='maxl')
sampler.plot_bestfit(save_fn=f'{save_handle}_model_mean.pdf', model='mean')