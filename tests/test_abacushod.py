from acm.hod import AbacusHOD
from acm.utils import setup_logging
import numpy as np
from pathlib import Path
from pycorr import TwoPointCorrelationFunction
import matplotlib.pyplot as plt
import pandas 

def get_hod_params(nrows=None):
    hod_dir = Path(f'/pscratch/sd/e/epaillas/emc/hod_params/yuan23/')
    hod_fn = hod_dir / f'hod_params_yuan23_c000.csv'
    df = pandas.read_csv(hod_fn, delimiter=',')
    df.columns = df.columns.str.strip()
    df.columns = list(df.columns.str.strip('# ').values)
    return df.to_dict('list')


setup_logging()

hod_params = get_hod_params()
hod_params['logM1'] = hod_params.pop('logM_1')
hod_params['Acent'] = hod_params.pop('A_cen')
hod_params['Asat'] = hod_params.pop('A_sat')
hod_params['Bcent'] = hod_params.pop('B_cen')
hod_params['Bsat'] = hod_params.pop('B_sat')

abacus = AbacusHOD(varied_params=hod_params.keys(),
                   sim_type='base', redshift=0.5,
                   cosmo_idx=0, phase_idx=0)


fig, ax = plt.subplots()

for i in range(50):

    hod = {key: hod_params[key][i] for key in hod_params.keys()}

    positions_dict = abacus.run(hod, nthreads=256)
    data_positions = np.c_[
        positions_dict['X'],
        positions_dict['Y'],
        positions_dict['Z_RSD']
    ]
    sedges = np.linspace(0, 150, 100)
    muedges = np.linspace(-1, 1, 241)
    result = TwoPointCorrelationFunction(data_positions1=data_positions, boxsize=abacus.boxsize,
                                        mode='smu', edges=(sedges, muedges), los='z',
                                        nthreads=256, position_type='pos')
    s, multipoles = result(ells=(0, 2, 4), return_sep=True)
    ax.plot(s, s**2*multipoles[1])

plt.show()