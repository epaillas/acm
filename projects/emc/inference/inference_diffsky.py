from sunbird.inference.pocomc import PocoMCSampler
from sunbird.inference.priors import Yuan23, AbacusSummit
from sunbird import setup_logging

import acm.observables.emc as emc

import numpy as np
import argparse
from pathlib import Path


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


phase_idx = 1
sampling = 'mass_conc'

setup_logging()

# set up the inference
priors, ranges, labels = get_priors(cosmo=True, hod=True)
fixed_params = ['w0_fld', 'wa_fld', 'nrun', 'N_ur']
add_emulator_error = True

# load observables with their custom filters
observable = emc.CombinedObservable([
    emc.GalaxyNumberDensity(
        select_filters={
        },
    ),
    emc.CorrectedGalaxyProjectedCorrelationFunction(
        select_filters={
        },
    ),
    emc.GalaxyCorrelationFunctionMultipoles(
        select_filters={
        },
    ),
    # emc.WaveletScatteringTransform(
    #     select_filters={
    #     },
    # ),
])

statistics = observable.stat_name
print(f'Fitting {statistics} on diffsky')

# load the data
data_x = observable.lhc_x
data_x_names = observable.lhc_x_names
data_y = observable.diffsky_y(phase_idx=phase_idx, sampling=sampling)
print(f'Loaded LHC x with shape: {data_x.shape}')
print(f'Loaded Diffsky y with shape {data_y.shape}')

# load the covariance matrix
covariance_matrix = observable.get_covariance_matrix(divide_factor=8)
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

# sample the posterior
sampler = PocoMCSampler(
    observation=data_y,
    precision_matrix=precision_matrix,
    theory_model=models,
    fixed_parameters=fixed_params,
    priors=priors,
    ranges=ranges,
    labels=labels,
    slice_filters=observable.slice_filters,
    select_filters=observable.select_filters,
    coordinates=model_coordinates,
    ellipsoid=True,
)

# sampler(vectorize=True, n_total=10_000)
sampler(vectorize=True, n_total=4096)

# plot and save results
markers = {
    'Omega_m': 0.3089,
    'omega_cdm': 0.1188,
    'omega_b': 0.02230,
    'h': 0.6774,
    'n_s': 0.9667,
    'sigma8_m': 0.8147
}

# plot and save results
statistics = '+'.join(statistics)

save_dir = '/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/diffsky/mar24/'
save_dir = Path(save_dir) / f'galsampled_67120_fixedAmp_{phase_idx:03}_{sampling}_v0.3/LCDM/'
Path(save_dir).mkdir(parents=True, exist_ok=True)

sampler.plot_triangle(save_fn=save_dir / f'chain_{statistics}_triangle.pdf', thin=128,
                      markers=markers, title_limit=1)
sampler.plot_trace(save_fn=save_dir / f'chain_{statistics}_trace.pdf', thin=128)
sampler.save_chain(save_fn=save_dir / f'chain_{statistics}.npy', metadata={'markers': markers})
sampler.save_table(save_fn=save_dir / f'chain_{statistics}_stats.txt')
sampler.plot_bestfit(save_fn=save_dir / f'chain_{statistics}_bestfit.png', model='maxl')
sampler.plot_bestfit(save_fn=save_dir / f'chain_{statistics}_mean.png', model='mean')
