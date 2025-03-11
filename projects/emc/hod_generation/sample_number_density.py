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


boxsize = 2000
hods = list(range(500))
phase_idx = 0
seed_idx = 0
# cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
cosmos = list(range(139, 127)) + list(range(130, 182))



for cosmo_idx in cosmos:
    print(f'cosmo_idx: {cosmo_idx}')
    
    hod_params = get_hod_params(cosmo_idx=cosmo_idx)

    # load abacusHOD class
    abacus = BoxHOD(varied_params=hod_params.keys(),
                    sim_type='base', redshift=0.5,
                    cosmo_idx=cosmo_idx, phase_idx=phase_idx)

    nden = []

    # sample HODs and save to disk
    for hod_idx in hods:
        hod = {key: hod_params[key][hod_idx] for key in hod_params.keys()}
        hod_dict = abacus.run(hod, nthreads=64, tracer_density_mean=5e-4)
        nden.append(len(hod_dict['LRG']['X']) / boxsize ** 3)

    save_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/number_density/cosmo+hod/z0.5/yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx}/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / f'nden_downsampled.npy'
    numpy.save(save_fn, nden)
