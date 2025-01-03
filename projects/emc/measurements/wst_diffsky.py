#First load some basic packages

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
#from pycorr import TwoPointCorrelationFunction, setup_logging
from astropy.table import Table
from cosmoprimo.cosmology import Cosmology

import matplotlib.pyplot as plt
import numpy as np
import kymatio
import sklearn
import torch
#import MAS_library as MASL
#import Pk_library as PKL
from kymatio import Scattering2D
from kymatio.numpy import Scattering2D
from kymatio.sklearn import Scattering2D
from kymatio.torch import HarmonicScattering3D
#from kymatio.numpy import HarmonicScattering3D
import math
from matplotlib import gridspec
from matplotlib import cm
from nbodykit.lab import *
from astropy.io import fits


#Define functions to get Hubble factor for a given cosmology and z
import os
import multiprocessing
from joblib import Parallel, delayed
import time

from nbodykit import setup_logging, style
plt.style.use(style.notebook)
setup_logging()
#adot
def adot(a, Om0):
 return np.sqrt(Om0/a+(1-Om0)*a*a)
def H(a, Om0):
 return adot(a, Om0)/a
import fitsio


gridlittle = 100 #200 #200 #300 #200 #200
D3d = gridlittle#256
J3d = 4#8 # number of scales: 2^8 = 256 => scattering output will be scalar
L3d = 4 # number of thetas
print (D3d,J3d,L3d)

#from kymatio.scattering3d.backend.torch_backend \    import compute_integrals
    
#integral_powers = [2.0] #[1.0] #[0.5, 1.0, 2.0, 3.0]
integral_powers = [0.8]
sigma = 0.8

S = HarmonicScattering3D(J=J3d, shape=(D3d, D3d, D3d), L=L3d, sigma_0=sigma, integral_powers=integral_powers ,
                                      max_order=2)
                                      
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print (device)
S.to(device)


# define cosmology
boxsize = 1000 #2000
redshift = 0.5
cosmo = Cosmology(Omega_m=0.3089, h=0.6774, n_s=0.9667,
                  sigma8=0.8147, engine='class')  # UNIT cosmology
hubble = 100 * cosmo.efunc(redshift)
scale_factor = 1 / (1 + redshift)

#Load one of the diffsky mocks

import h5py
import numpy as np

f1 = h5py.File('/global/cfs/cdirs/desicollab/users/gbeltzmo/C3EMC/UNIT/galsampled_diffsky_mock_67120_fixedAmp_001_mass_conc_v0.3.hdf5')
pos = f1['data']['pos']#.shape
vel = f1['data']['vel']
is_lrg = f1['data']["diffsky_isLRG"]

posLRG = pos[is_lrg==True]
velLRG = vel[is_lrg==True]

#Add RSD along z-axis in this example
posLRG[:, 2] = (posLRG[:, 2] + velLRG[:, 2]/(hubble * scale_factor))%1000

#Save positions to intermediate txt file. #commented out below
#np.savetxt('/pscratch/sd/g/gvalogia/PosRSDz_galsampled_diffsky_mock_67120_fixedAmp_001_mass_conc_v0.3.txt', posLRG, fmt='%.7e')

#Load positions, calculate density field and then extract WST data vector from given diffsky field

#Can loop over phases. Now only for phase no 1 for this example.
for i in range(1):
    print (i)
    # read the data
    namesRSDder =['x', 'y', 'z']
    locals()['fder'+str(i)] = CSVCatalog('/pscratch/sd/g/gvalogia/PosRSDz_galsampled_diffsky_mock_67120_fixedAmp_00'+str(i+1)+'_mass_conc_v0.3.txt', namesRSDder)

    # combine x, y, z to Position, and add boxsize
    locals()['fder'+str(i)]['Position'] = locals()['fder'+str(i)]['x'][:, None] * [1, 0, 0] + locals()['fder'+str(i)]['y'][:, None] * [0, 1, 0] + locals()['fder'+str(i)]['z'][:, None] * [0, 0, 1]
    locals()['fder'+str(i)].attrs['BoxSize'] = 1000.0
    locals()['meshder'+str(i)] = locals()['fder'+str(i)].to_mesh(Nmesh=100, BoxSize=1000.0, window='tsc')#, compensated=True)
    locals()['oneplusdeltader'+str(i)] = locals()['meshder'+str(i)].paint(mode='real')
    #locals()['deltaRSDfid'+str(i)] = locals()['oneplusdelta'+str(i)].value.flatten()
    locals()['deltader'+str(i)] = locals()['oneplusdeltader'+str(i)] - 1.0

    full_density_batch = torch.from_numpy(np.asarray(locals()['deltader'+str(i)]))
    full_density_batch = full_density_batch.to(device).float()
    full_density_batch = full_density_batch.contiguous()
    smat_orders_1_and_2 = S(full_density_batch)
    smat = np.absolute(smat_orders_1_and_2.cpu().numpy()[:,:,0])

    #smatavg = np.mean(smat, axis=1)
    smatavg = smat.flatten()
    #mean s0 eval.

    s0 = np.sum(np.absolute(locals()['deltader'+str(i)])**0.80)
    #stack s0 + s12
    smatavg = np.hstack((s0, smatavg))
    #Save in case you want to, which we do. Now commented out below.
    #np.savetxt('/global/homes/g/gvalogia/MockChallenge/WST_RSDz_galsampled_diffsky_mock_67120_fixedAmp_00'+str(i+1)+'_mass_conc_v0.3.txt', np.real(smatavg), fmt='%.7e')

    del smatavg,smat










