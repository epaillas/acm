from acm.estimators.galaxy_clustering.cumulants import DensityFieldCumulants
import fitsio
from pathlib import Path
import numpy as np
from cosmoprimo.fiducial import AbacusSummit

def get_hod_positions(input_fn, los='z'):
    hod = fitsio.read(input_fn)
    pos = np.c_[hod['X'], hod['Y'], hod['Z']]
    hubble = 100 * cosmo.efunc(redshift)
    scale_factor = 1 / (1 + redshift)
    if los == 'x':
        pos[:, 0] += hod['VX'] / (hubble * scale_factor)
    elif los == 'y':
        pos[:, 1] += hod['VY'] / (hubble * scale_factor)
    elif los == 'z':
        pos[:, 2] += hod['VZ'] / (hubble * scale_factor)
    return pos

cosmo = AbacusSummit(0)
redshift = 0.5


hod_dir = f'/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior/c000_ph000/seed0/'
hod_fn = Path(hod_dir) / f'hod010.fits'
data_positions = get_hod_positions(hod_fn)

    
dc = DensityFieldCumulants(boxsize=2000, boxcenter=2000/2, cellsize=5.0)
lda = np.arange(-10, 11, 1)  # cumulant orders (lambdas)
dc.assign_data(positions=data_positions, wrap=True)
dc.set_density_contrast(smoothing_radius=10)
cgf = dc.compute_cumulants(lda)

cout = {'lda': lda, 'cgf': cgf}
np.save('cumulants.npy', cout)
