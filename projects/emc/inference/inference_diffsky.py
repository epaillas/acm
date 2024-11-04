import numpy as np
from sunbird.emulators import FCN
from pathlib import Path
from sunbird.inference.hamiltonian import HMCSampler
from sunbird.inference.priors import Yuan23, AbacusSummit
from sunbird.data.data_utils import convert_to_summary
import torch
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
        
def get_save_fn(statistic):
    save_dir = f'/pscratch/sd/e/epaillas/emc/posteriors/diffsky/oct2/{statistic}/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    scales_str = ''
    if slice_filters:
        for key, value in slice_filters.items():
            scales_str += f'_{key}{value[0]:.1f}_{key}{value[1]:.1f}'
    return Path(save_dir) / f'chain{scales_str}_z{redshift}_fixedAmp_{phase_idx:03}_{galsample}_v{version}.npy'

fixed_params = {
    'A_cen': 0.0,
    'A_sat': 0.0,
    # 's': 0.0,
    'w0_fld': -1.0,
    'wa_fld': 0.0,
    'nrun': 0.0,
    'N_ur': 2.0328,
    # 'N_ur': 2.0268,  # Planck 2015
    # 'omega_b': 0.02230,
    # 'alpha_c': 0.0,
    # 'alpha_s': 1.0,
    # 'n_s': 0.9667
}   

# data vector settings
redshift = 0.5
phase_idx = 2
galsample = 'mass_conc'
version = 0.3


priors, ranges, labels = get_priors(cosmo=True, hod=True)
select_filters = {}
smins = [0]
add_emulator_error = True
statistics = ['pk']

num_chains = 1
smax = 152
kmin = 0.0
# smins = [0, 5, 10, 20, 40, 60, 80, 100]
smins = [12.5]
kmaxs = [0.5]

for smin in smins:
    for kmax in kmaxs:
        # slice_filters = {'k': [kmin, kmax]}
        slice_filters = {}

        covariance_matrix, n_sim = read_covariance(statistics=statistics,
                                                   select_filters=select_filters,
                                                   slice_filters=slice_filters,
                                                   volume_factor=8)
        print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

        diffsky_y, model_filters = read_diffsky(statistics=statistics,
                                                select_filters=select_filters,
                                                slice_filters=slice_filters,
                                                return_mask=True)
        print(f'Loaded diffsky with shape: {diffsky_y.shape}')

        # load the LHC
        lhc_x, lhc_y, lhc_x_names = read_lhc(statistics=statistics,
                                             select_filters=select_filters,
                                             slice_filters=slice_filters,
                                             return_mask=False)
        print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

        print(f'Fitting Diffsky')

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

        hmc = HMCSampler(
            observation=diffsky_y,
            precision_matrix=precision_matrix,
            nn_theory_model=nn_model,
            nn_parameters=nn_params,
            fixed_parameters=fixed_params,
            priors=priors,
            ranges=ranges,
            labels=labels,
            model_filters=model_filters,
        )

        save_fn = get_save_fn(statistic='+'.join(statistics))

        posterior = hmc(num_warmup=4000, num_samples=4000, dense_mass=True,
                        target_accept_prob=0.95,
                        num_chains=num_chains, save_fn=save_fn)