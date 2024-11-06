import numpy as np
from pathlib import Path
from sunbird.inference.dynesty import DynamicDynestySampler
from sunbird.inference.priors import Yuan23, AbacusSummit
import sys
sys.path.insert(0, '..')
from io_tools import *


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

def get_save_fn(statistic, mock_idx, kmin, kmax, smin, smax):
    save_dir = f'/pscratch/sd/e/epaillas/emc/posteriors/nested/{statistic}/test'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    scales_str = ''
    if any([stat in fourier_stats for stat in statistic.split('+')]):
        scales_str += f'_kmin{kmin}_kmax{kmax}'
    if any([stat in conf_stats for stat in statistic.split('+')]):
        scales_str += f'_smin{smin:.1f}_smax{smax:.1f}'
    return Path(save_dir) / f'chain_idx{mock_idx}{scales_str}.csv'


priors, ranges, labels = get_priors(cosmo=True, hod=True)
select_filters = {'multipoles': [0, 2], 'statistics': ['quantile_data_correlation']}
fixed_params = ['w0_fld', 'wa_fld', 'nrun', 'N_ur',]
add_emulator_error = True
# statistics = ['wp', 'dsc_conf', 'tpcf']
statistics = ['tpcf']


# smins = [0]
smax = 152
kmin = 0.0
# smins = [0, 5, 10, 20, 40, 60, 80, 100]
smins = [0, 5, 10, 20, 40, 60, 80, 100]
# smins = [0.0]
kmaxs = [1.0]
# kmaxs = [0.2]
for smin in smins:
    for kmax in kmaxs:
        slice_filters = {'s': [smin, smax], 'k': [kmin, kmax]}

        covariance_matrix, n_sim = read_covariance(statistics=statistics,
                                                select_filters=select_filters,
                                                slice_filters=slice_filters)
        print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

        # load the data
        lhc_x, lhc_y, lhc_x_names, model_filters = read_lhc(statistics=statistics,
                                                            select_filters=select_filters,
                                                            slice_filters=slice_filters,
                                                            return_mask=True)
        print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

        lhc_test_y = lhc_y[:600]

        # idxs = [30, 199, 330, 438]
        idxs = [30]

        for mock_idx in idxs:
            print(f'Fitting HOD {mock_idx}')

            fixed_params_dict = {key: lhc_x[mock_idx, lhc_x_names.index(key)]
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
            
            sampler = DynamicDynestySampler(
                observation=lhc_test_y[mock_idx],
                # observation=pred_y[mock_idx],
                precision_matrix=precision_matrix,
                theory_model=models[0],
                fixed_parameters=fixed_params_dict,
                priors=priors,
                model_filters=model_filters[0],
                # model_filters=filters,
            )

            save_fn = get_save_fn(statistic='+'.join(statistics),
                                  kmin=kmin, kmax=kmax,
                                  smin=smin, smax=smax,
                                  mock_idx=mock_idx)

            sampler(nlive=1000, dlogz=0.1, save_fn=save_fn)



