from acm.hod import AbacusHOD
from acm.utils import setup_logging
import numpy as np
from pathlib import Path
import pandas

setup_logging()

def get_hod_params():
    """Some example HOD parameters."""
    hod_dir = Path(f'/pscratch/sd/e/epaillas/emc/hod_params/yuan23/')
    hod_fn = hod_dir / f'hod_params_yuan23_c000.csv'
    df = pandas.read_csv(hod_fn, delimiter=',')
    df.columns = df.columns.str.strip()
    df.columns = list(df.columns.str.strip('# ').values)
    return df.to_dict('list')

n_hod = 30000  # number of hods to sample

# read example HOD parameters
hod_params = get_hod_params()

# load abacusHOD class
abacus = AbacusHOD(varied_params=hod_params.keys(),
                   sim_type='base', redshift=0.5,
                   cosmo_idx=0, phase_idx=0)

for i in range(2430, n_hod):
    save_dir = f'/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior/c000_ph000'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / f'hod{i:03}.fits'

    hod = {key: hod_params[key][i] for key in hod_params.keys()}
    positions_dict = abacus.run(hod, nthreads=16, save_fn=save_fn, tracer_density_mean=5e-4)
