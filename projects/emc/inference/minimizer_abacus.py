import numpy as np
from sunbird.inference.bobyqa import BobyqaProfiler
from sunbird.inference.priors import Yuan23, AbacusSummit
import argparse
from pathlib import Path
import acm.observables.emc as emc
from sunbird import setup_logging
from desilike import mpi


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


# Initialize MPI
mpicomm = mpi.COMM_WORLD
rank = mpicomm.Get_rank()
size = mpicomm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument("--cosmo_idx", type=int, default=0)
parser.add_argument("--hod_idx", type=int, default=30)

args = parser.parse_args()
if rank == 0:
    setup_logging()

# set up the inference
priors, ranges, labels = get_priors(cosmo=True, hod=True)
fixed_params = ['w0_fld', 'wa_fld', 'nrun', 'N_ur']
add_emulator_error = True

# load observables with their custom filters
observable = emc.GalaxyCorrelationFunctionMultipoles(
        select_filters={
            'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx,
        },
        slice_filters={
        }
    )

statistics = observable.stat_name

# load the data
data_x = observable.lhc_x
data_x_names = observable.lhc_x_names
data_y = observable.lhc_y

# load the covariance matrix
covariance_matrix = observable.get_covariance_matrix(divide_factor=64)

if rank == 0:
    print(f'Fitting {statistics} with cosmo_idx={args.cosmo_idx} and hod_idx={args.hod_idx}')
    print(f'Loaded data_x with shape: {data_x.shape}')
    print(f'Loaded data_y with shape {data_y.shape}')
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

start = [np.random.uniform(*ranges[key]) for key in ranges if key not in fixed_params]

# run the profiler
bobyqa = BobyqaProfiler(
    observation=data_y,
    precision_matrix=precision_matrix,
    theory_model=models,
    fixed_params=fixed_params,
    priors=priors,
    ranges=ranges,
    labels=labels,
    start=start,
    slice_filters=observable.slice_filters,
    select_filters=observable.select_filters,
    coordinates=model_coordinates,
    mpicomm=mpicomm,
)

save_dir = '/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/feb14/'
save_dir = Path(save_dir) / f'c{args.cosmo_idx:03}_hod{args.hod_idx:03}/lcdm/'
Path(save_dir).mkdir(parents=True, exist_ok=True)

bobyqa.minimize(save_fn=save_dir / f'profiles_{statistics}.npy')
bobyqa.save_table(save_fn=save_dir / f'profiles_{statistics}_stats.txt')
bobyqa.plot_bestfit(save_fn=save_dir / f'profiles_{statistics}_bestfit.png')