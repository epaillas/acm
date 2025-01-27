from acm.hod import LightconeHOD
from acm.utils import setup_logging
from pyrecon.utils import sky_to_cartesian
import numpy as np
from pathlib import Path
import pandas
from cosmoprimo.fiducial import AbacusSummit

setup_logging()

def get_hod_params(nrows=None):
    """Some example HOD parameters."""
    hod_dir = Path(f'/pscratch/sd/e/epaillas/emc/hod_params/yuan23/')
    hod_fn = hod_dir / f'hod_params_yuan23_c000.csv'
    df = pandas.read_csv(hod_fn, delimiter=',')
    df.columns = df.columns.str.strip()
    df.columns = list(df.columns.str.strip('# ').values)
    return df.to_dict('list')

def get_clustering_positions(data):
    mask = (data['Z'] > zmin) & (data['Z'] < zmax)
    ra = data['RA'][mask]
    dec = data['DEC'][mask]
    dist = distance(data['Z'][mask])
    pos = sky_to_cartesian(ra=ra, dec=dec, dist=dist)
    weights = np.ones(len(pos))
    return pos

def get_twopoint_clustering(save_fn=False):
    print('Computing clustering.')
    from pycorr import setup_logging, TwoPointCorrelationFunction
    setup_logging()
    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)
    return TwoPointCorrelationFunction(
        data_positions1=data_positions, randoms_positions1=randoms_positions,
        position_type='pos', edges=edges, mode='smu', gpu=False, nthreads=128, estimator='landyszalay',
    )


zmin, zmax = 0.5, 0.55

# read example HOD parameters
hod_params = get_hod_params()

# load abacusHOD class
abacus = LightconeHOD(varied_params=hod_params.keys(),
                zrange=(zmin, zmax), cosmo_idx=0, phase_idx=0)


hod = {key: hod_params[key][30] for key in hod_params.keys()}

hod_dict, randoms_dict = abacus.run(hod, nthreads=128, full_sky=True,
                                    make_randoms=True, apply_nz=True, alpha_rand=5)


cosmo = AbacusSummit(0)
distance = cosmo.comoving_radial_distance

data_positions = get_clustering_positions(hod_dict['LRG'])
randoms_positions = get_clustering_positions(randoms_dict['LRG'])

tpcf = get_twopoint_clustering()
tpcf.save('xinyi_example_tpcf_z0.5-0.55.npy')

