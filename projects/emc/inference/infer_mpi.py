from sunbird.inference.pocomc import PocoMCSampler
from sunbird.inference.priors import Yuan23, AbacusSummit
from sunbird import setup_logging

import acm.observables.emc as emc

from desilike import mpi

from pathlib import Path
import numpy as np
import os


os.environ["OMP_NUM_THREADS"] = "1"


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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run cosmological parameter inference')
    
    parser.add_argument('--cosmo_indices', type=int, nargs='+', 
                      default=[0, 1, 2, 3, 110, 111],
                      help='List of cosmology indices to process')
    
    parser.add_argument('--hod_start', type=int, default=0,
                      help='Starting HOD index')
    parser.add_argument('--hod_end', type=int, default=99,
                      help='Ending HOD index (inclusive)')
    
    parser.add_argument('--lambda_cdm', action='store_true',
                      help='Use Lambda CDM model')
    
    parser.add_argument('--statistics', nargs='+', 
                      choices=['number_density', 'wp', 
                               'tpcf', 'bk', 'pk', 'wst', 'dt_voids',],
                      default=['number_density', 'wp', 
                               'tpcf', 'bk'],
                      help='Statistics to use in the analysis')
    parser.add_argument('--save_dir', type=str, default= '/global/cfs/cdirs/desicollab/users/cuesta/emc_chains')
    return parser.parse_args()

def get_observables(
    statistics, cosmo_idx, hod_idx
):
    observables = []
    for stat in statistics:
        obs = stat_map[stat]
        obs.select_mocks['cosmo_idx'] = cosmo_idx
        obs.select_mocks['hod_idx'] = hod_idx
        observables.append(obs)
    return emc.CombinedObservable(observables)

if __name__ == "__main__":
    args = parse_args()
    
    lambda_cdm = args.lambda_cdm
    cosmo_indices = args.cosmo_indices
    hods = list(range(args.hod_start, args.hod_end + 1))
    n_hods = len(hods)

    stat_map = {
        'number_density': emc.GalaxyNumberDensity(
            select_mocks={'cosmo_idx': None, 'hod_idx': None}),
        'wp': emc.GalaxyProjectedCorrelationFunction(
            select_mocks={'cosmo_idx': None, 'hod_idx': None}),
        'tpcf': emc.GalaxyCorrelationFunctionMultipoles(
            select_mocks={'cosmo_idx': None, 'hod_idx': None,},
            select_filters={'multipoles': [0, 2]},
        ),
        'pk': emc.GalaxyPowerSpectrumMultipoles(
            select_mocks={'cosmo_idx': None, 'hod_idx': None},
        ),
        'bk': emc.GalaxyBispectrumMultipoles(
            select_mocks={'cosmo_idx': None, 'hod_idx': None},
        ),
        'wst': emc.WaveletScatteringTransform(
            select_mocks={'cosmo_idx': None, 'hod_idx': None},
        ),
        'dt_voids': emc.DTVoidGalaxyCorrelationFunctionMultipoles(
            select_mocks={'cosmo_idx': None, 'hod_idx': None},
        ),
    }

    mpicomm = mpi.COMM_WORLD
    if mpicomm.rank == 0:
        setup_logging()

    if lambda_cdm:
        fixed_params = ['w0_fld', 'wa_fld', 'nrun', 'N_ur']
    else:
        fixed_params = None
    add_emulator_error = True
    priors, ranges, labels = get_priors(cosmo=True, hod=True)

    for cosmo_idx in cosmo_indices:
        for hod_idx in hods:
            # spread hods across MPI ranks
            if hod_idx % mpicomm.size != mpicomm.rank:
                continue
            print(f'Rank {mpicomm.rank} is fitting HOD {hod_idx}')

            observable = get_observables(args.statistics, cosmo_idx, hod_idx)
            statistics = observable.stat_name

            # load the data
            data_x = observable.lhc_x
            data_x_names = observable.lhc_x_names
            data_y = observable.lhc_y

            # load the covariance matrix
            covariance_matrix = observable.get_covariance_matrix(divide_factor=64)

            # load emulator error
            if add_emulator_error:
                emulator_error = observable.get_emulator_error()
                covariance_matrix += np.diag(emulator_error**2)

            # get the debiased inverse
            correction = observable.get_covariance_correction(
                n_s=len(observable.small_box_y),
                n_d=len(covariance_matrix),
                n_theta=len(data_x_names) - len(fixed_params) if fixed_params is not None else len(data_x_names),
                method='percival',
            )
            precision_matrix = np.linalg.inv(correction * covariance_matrix)

            if fixed_params is not None:
                fixed_params = {key: data_x[data_x_names.index(key)]
                                for key in fixed_params}

            # load the model
            models = observable.model

            if mpicomm.rank == 0:
                print(f'Rank 0 fitting {statistics} with cosmo_idx={cosmo_idx} and hod_idx={hod_idx}')
                print(f'Loaded data_x with shape: {data_x.shape}')
                print(f'Loaded data_y with shape {data_y.shape}')
                print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

            # sample the posterior
            sampler = PocoMCSampler(
                observation=data_y,
                precision_matrix=precision_matrix,
                theory_model=models,
                fixed_parameters=fixed_params,
                priors=priors,
                ranges=ranges,
                labels=labels,
                ellipsoid=True,
            )

            sampler(vectorize=True, n_total=4096, progress=False)

            # plot and save results
            markers = {key: data_x[data_x_names.index(key)] for key in data_x_names if (fixed_params is not None) and (key not in fixed_params)}
            statistics = '+'.join(statistics)

            if lambda_cdm:
                save_dir = Path(args.save_dir) / f'c{cosmo_idx:03}_hod{hod_idx:03}/LCDM/'
            else:
                save_dir = Path(args.save_dir) / f'c{cosmo_idx:03}_hod{hod_idx:03}/w0waCDM/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            sampler.plot_triangle(save_fn=save_dir / f'chain_{statistics}_triangle.pdf', thin=128,
                                markers=markers, title_limit=1)
            sampler.plot_trace(save_fn=save_dir / f'chain_{statistics}_trace.pdf', thin=128)
            sampler.save_chain(save_fn=save_dir / f'chain_{statistics}.npy', metadata={'markers': markers, 'zeff': 0.5})
            sampler.save_table(save_fn=save_dir / f'chain_{statistics}_stats.txt')
            sampler.plot_bestfit(save_fn=save_dir / f'chain_{statistics}_bestfit.png', model='maxl')
            sampler.plot_bestfit(save_fn=save_dir / f'chain_{statistics}_mean.png', model='mean')