from acm.hod import BoxHOD
from acm.utils import setup_logging
from pathlib import Path
import pandas
import numpy

setup_logging()

def get_hod_params(cosmo_idx=0):
    """Some example HOD parameters."""
    hod_dir = Path(f'/pscratch/sd/e/epaillas/emc/hod_params/yuan23/cosmo_split')
    hod_fn = hod_dir / f'hod_params_yuan23_c{cosmo_idx:03}.csv'
    df = pandas.read_csv(hod_fn, delimiter=',')
    df.columns = df.columns.str.strip()
    df.columns = list(df.columns.str.strip('# ').values)
    return df.to_dict('list')


boxsize = 500
hod_idx = 30
phases = list(range(3000, 5000))
seed_idx = 0
cosmo_idx = 0

hod_params = get_hod_params(cosmo_idx=cosmo_idx)

for phase_idx in phases:
    print(f'Processing phase {phase_idx}...')
    try:
        # load abacusHOD class
        abacus = BoxHOD(varied_params=hod_params.keys(),
                        sim_type='small', redshift=0.5,
                        cosmo_idx=cosmo_idx, phase_idx=phase_idx)

        hod = {key: hod_params[key][hod_idx] for key in hod_params.keys()}
        hod_dict = abacus.run(hod, nthreads=64, tracer_density_mean=5e-4)
        nden = (len(hod_dict['LRG']['X']) / boxsize ** 3)

        save_dir = '/pscratch/sd/e/epaillas/emc/covariance_sets/number_density/z0.5/yuan23_prior/'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_fn = Path(save_dir) / f'number_density_ph{phase_idx:03}_hod{hod_idx:03}.npy'
        numpy.save(save_fn, nden)
    except:
        print(f'Phase {phase_idx} is missing.')
        continue

