from acm.hod import AbacusHOD
from acm.utils import setup_logging
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

# hod_idx = 9971
# hod_idx = 466
hod_idx = 6486
phases = list(range(3000, 5000))

hod_params = get_hod_params()  # read example HOD parameters
hod = {key: hod_params[key][hod_idx] for key in hod_params.keys()}

# sample HODs and save to disk
for phase_idx in phases:
    try:
        abacus = AbacusHOD(varied_params=hod_params.keys(),
                        sim_type='small', redshift=0.5,
                        cosmo_idx=0, phase_idx=phase_idx)
        save_dir = f'/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior/small/hod{hod_idx:03}/'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_fn = Path(save_dir) / f'ph{phase_idx:03}_hod{hod_idx:03}.fits'
        positions_dict = abacus.run(hod, nthreads=64, save_fn=save_fn, tracer_density_mean=5e-4)
    except:
        print(f'Phase {phase_idx} is missing.')
        continue
