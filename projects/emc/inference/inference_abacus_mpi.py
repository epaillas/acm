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


mpicomm = mpi.COMM_WORLD
if mpicomm.rank == 0:
    setup_logging()

for cosmo_idx in [2, 0, 1, 3]:
    hods = list(range(0, 100))

    n_hods = len(hods)

    for hod_idx in hods:
        # spread hods across MPI ranks
        if hod_idx % mpicomm.size != mpicomm.rank:
            continue
        print(f'Rank {mpicomm.rank} is fitting HOD {hod_idx}')

        # set up the inference
        priors, ranges, labels = get_priors(cosmo=True, hod=True)
        fixed_params = ['w0_fld', 'wa_fld', 'nrun', 'N_ur']
        # , 'sigma', 'kappa', 'alpha', 's', 'A_cen', 'A_sat', 'B_cen', 'B_sat', 'alpha_s', 'alpha_c']
        add_emulator_error = True

        # load observables with their custom filters
        observable = emc.CombinedObservable([
            emc.GalaxyNumberDensity(
                select_filters={
                    'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx,
                },
            ),
            # emc.GalaxyProjectedCorrelationFunction(
            #     select_filters={
            #         'cosmo_idx': cosmo_idx, 'hod_idx': hod_idx,
            #     },
            #     slice_filters={
            #     }
            # ),
            emc.GalaxyPowerSpectrumMultipoles(
                select_filters={
                    'cosmo_idx': cosmo_idx, 'hod_idx': hod_idx,
                },
                slice_filters={
                }
            ),
            emc.GalaxyBispectrumMultipoles(
                select_filters={
                    'cosmo_idx': cosmo_idx, 'hod_idx': hod_idx,
                },
                slice_filters={
                }
            ),
            # emc.WaveletScatteringTransform(
            #     select_filters={
            #         'cosmo_idx': cosmo_idx, 'hod_idx': hod_idx,
            #     },
            #     slice_filters={
            #     }
            # )
            # emc.DTVoidGalaxyCorrelationFunctionMultipoles(
            #     select_filters={
            #         'cosmo_idx': cosmo_idx, 'hod_idx': hod_idx,
            #     },
            #     slice_filters={
            #     }
            # )
        ])

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
            n_theta=len(data_x_names) - len(fixed_params),
            method='percival',
        )
        precision_matrix = np.linalg.inv(correction * covariance_matrix)

        fixed_params = {key: data_x[data_x_names.index(key)]
                            for key in fixed_params}

        # load the model
        models = observable.model
        model_coordinates = observable.coords_model

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
            slice_filters=observable.slice_filters,
            select_filters=observable.select_filters,
            coordinates=model_coordinates,
            ellipsoid=True,
        )

        sampler(vectorize=True, n_total=4096, progress=False)

        # plot and save results
        markers = {key: data_x[data_x_names.index(key)] for key in data_x_names if key not in fixed_params}
        statistics = '+'.join(statistics)

        save_dir = '/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/mar24/'
        save_dir = Path(save_dir) / f'c{cosmo_idx:03}_hod{hod_idx:03}/LCDM/'
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        sampler.plot_triangle(save_fn=save_dir / f'chain_{statistics}_triangle.pdf', thin=128,
                            markers=markers, title_limit=1)
        sampler.plot_trace(save_fn=save_dir / f'chain_{statistics}_trace.pdf', thin=128)
        sampler.save_chain(save_fn=save_dir / f'chain_{statistics}.npy', metadata={'markers': markers})
        sampler.save_table(save_fn=save_dir / f'chain_{statistics}_stats.txt')
        sampler.plot_bestfit(save_fn=save_dir / f'chain_{statistics}_bestfit.png', model='maxl')
        sampler.plot_bestfit(save_fn=save_dir / f'chain_{statistics}_mean.png', model='mean')
