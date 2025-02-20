import acm.projects.bgs as bgs
from acm.projects.bgs import get_priors # TODO : change this if restrained to Yuan23 priors later ?
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

# set up the inference (NOTE : hardcoded values !)
fixed_parameters = ['omega_b', 'w0_fld', 'wa_fld', 'nrun', 'N_ur', 'B_cen', 'B_sat']
add_emulator_error = True

save_dir = '/pscratch/sd/s/sbouchar/acm/bgs/chains/' 

# load observables with their custom filters
observable = CombinedObservable([
    bgs.GalaxyCorrelationFunctionMultipoles(
        select_filters={
            'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx,
        },
        slice_filters={
        }
    ),
    bgs.DensitySplitCorrelationFunctionMultipoles(
        select_filters={
            'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx,
        },
        slice_filters={
        }
    ),
])

# define utility functions
def get_covariance_correction(n_s, n_d, n_theta=None, method='percival'):
        """
        Correction factor to debias de inverse covariance matrix.

        Args:
            n_s (int): Number of simulations.
            n_d (int): Number of bins of the data vector.
            n_theta (int): Number of free parameters.
            method (str): Method to compute the correction factor.

        Returns:
            float: Correction factor
        """
        if method == 'percival':
            B = (n_s - n_d - 2) / ((n_s - n_d - 1)*(n_s - n_d - 4))
            return (n_s - 1)*(1 + B*(n_d - n_theta))/(n_s - n_d + n_theta - 1)
        elif method == 'hartlap':
            return (n_s - 1)/(n_s - n_d - 2)

def save_handle():
    # NOTE : bad practice to use global variables (TODO : change later)
    save_dir = save_dir 
    statistics = observable.stat_name
    slice_filters = observable.slice_filters
    
    save_dir = Path(save_dir) / f'c{args.cosmo_idx:03}_hod{args.hod_idx:03}/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
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

# load the data
data_x = observable.lhc_x
data_x_names = observable.lhc_x_names
data_y = observable.lhc_y
print(f'Loaded LHC x with shape: {data_x.shape}')
print(f'Loaded LHC y with shape {data_y.shape}')

# Order of the parameters needs to be the same as the data_x_names (also removes unused keys)
priors, ranges, labels = get_priors(cosmo=True, hod=True)
priors = {key: priors[key] for key in observable.lhc_x_names}

# load the covariance matrix
covariance_matrix = observable.get_covariance_matrix(volume_factor=64)
print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

# load emulator error
if add_emulator_error:
    emulator_error = observable.emulator_error
    covariance_matrix += np.diag(emulator_error**2)

# get the debiased inverse
correction = get_covariance_correction(
    n_s=len(observable.covariance_y),
    n_d=len(covariance_matrix),
    n_theta=len(data_x_names) - len(fixed_parameters),
    method='percival',
)
precision_matrix = np.linalg.inv(correction * covariance_matrix)

fixed_parameters = {key: data_x[data_x_names.index(key)]
                    for key in fixed_parameters}

# load the model
models = observable.model

# sample the posterior
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

# plot and save results
markers = {key: data_x[data_x_names.index(key)] for key in data_x_names if key not in fixed_parameters}

sampler.plot_triangle(save_fn=f'{save_handle()}_triangle.pdf', thin=128,
                    markers=markers, title_limit=1)
sampler.plot_trace(save_fn=f'{save_handle()}_trace.pdf', thin=128)
sampler.save_chain(f'{save_handle()}_chain.npy', metadata={'markers': markers, 'fixed_parameters': fixed_parameters})
sampler.save_table(f'{save_handle()}_stats.txt')
sampler.plot_bestfit(save_fn=f'{save_handle()}_model_maxl.pdf', model='maxl')
sampler.plot_bestfit(save_fn=f'{save_handle()}_model_mean.pdf', model='mean')