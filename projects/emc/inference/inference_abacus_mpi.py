import numpy as np
from sunbird.inference.pocomc import PocoMCSampler
from sunbird.inference.priors import Yuan23, AbacusSummit
from pyrecon import mpi
from acm.data.io_tools import *
from sunbird import setup_logging
import os

os.environ["OMP_NUM_THREADS"] = "1"


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
    save_dir = Path(save_dir) / f'c{cosmo_idx:03}_hod{hod_idx:03}/base_lcdm/'
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



mpicomm = mpi.COMM_WORLD
setup_logging(level='WARNING')

cosmo_idx = 0
hods = list(range(0, 100))

n_hods = len(hods)

for hod_idx in hods:
    # spread hods across MPI ranks
    if hod_idx % mpicomm.size != mpicomm.rank:
        continue
    print(f'Rank {mpicomm.rank} is fitting HOD {hod_idx}')

    # set up the inference
    priors, ranges, labels = get_priors(cosmo=True, hod=True)
    select_filters = {'cosmo_idx': cosmo_idx, 'hod_idx': hod_idx,
        'statistics': ['quantile_data_power'], 'quantiles': [0, 4]}
    fixed_params = ['w0_fld', 'wa_fld', 'N_ur', 'nrun']
    add_emulator_error = True
    statistics = ['pk', 'number_density']
    kmin, kmax = 0.0, 0.5
    slice_filters = {'k': (kmin, kmax)}

    # load the covariance matrix
    covariance_matrix, n_sim = read_covariance(statistics=statistics,
                                                select_filters=select_filters,
                                                slice_filters=slice_filters)
    if mpicomm.rank == 0:
        print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

    # load the data
    data_x, data_y, data_x_names, model_filters = read_lhc(statistics=statistics,
                                                        select_filters=select_filters,
                                                        slice_filters=slice_filters,
                                                        return_mask=True)
    if mpicomm.rank == 0:
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
        if mpicomm.rank == 0:
            print(f'Loaded emulator error with shape: {emulator_error.shape}')
        covariance_matrix += np.diag(emulator_error**2)

    # apply correction to the covariance matrix
    correction = get_covariance_correction(
        n_s=n_sim,
        n_d=len(covariance_matrix),
        n_theta=len(data_x_names) - len(fixed_params),
        correction_method='percival',
    )
    if mpicomm.rank == 0:
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
    sampler.triangle_plot(save_fn=f'{save_handle()}_triangle.pdf', thin=128,
                        markers=markers, title_limit=1)
    sampler.trace_plot(save_fn=f'{save_handle()}_trace.pdf', thin=128)
    print(f'Saving {save_handle()}_chain.npy')
    sampler.save_chain(f'{save_handle()}_chain.npy', metadata={'markers': markers})