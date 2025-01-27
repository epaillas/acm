from acm.hod import BoxHOD
from acm.utils import setup_logging
from pathlib import Path
import pandas

setup_logging()

def get_hod_params(cosmo_idx=0):
    """Some example HOD parameters."""
    hod_dir = Path(f'/pscratch/sd/e/epaillas/emc/hod_params/yuan23/cosmo_split')
    hod_fn = hod_dir / f'hod_params_yuan23_c{cosmo_idx:03}.csv'
    df = pandas.read_csv(hod_fn, delimiter=',')
    df.columns = df.columns.str.strip()
    df.columns = list(df.columns.str.strip('# ').values)
    return df.to_dict('list')

# hods = [466]
hods = list(range(30, 31))
phases = list(range(1))
seeds = list(range(1))
# cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
cosmos = list(range(1))

redshift = 0.5

# tracer_density_mean = 5e-4
tracer_density_mean = None

for cosmo_idx in cosmos:
    hod_params = get_hod_params(cosmo_idx=cosmo_idx)
    for phase_idx in phases:

        # load abacusHOD class
        abacus = BoxHOD(varied_params=hod_params.keys(),
                        sim_type='base', redshift=redshift,
                        cosmo_idx=cosmo_idx, phase_idx=phase_idx)

        # sample HODs and save to disk
        for hod_idx in hods:
            hod = {key: hod_params[key][hod_idx] for key in hod_params.keys()}
            for seed in seeds:
                save_dir = f'/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z{redshift:.1f}/yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed}/'
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                save_fn = Path(save_dir) / f'hod{hod_idx:03}_raw.fits'
                hod_dict = abacus.run(hod, nthreads=64, save_fn=save_fn, tracer_density_mean=tracer_density_mean, seed=seed)
