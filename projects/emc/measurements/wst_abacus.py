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


gridlittle = 200 #200 #200 #300 #200 #200
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

txt_file = open("/pscratch/sd/g/gvalogia/wst/log_kron/tasks.disbatch_copy", "r")

content_list = txt_file.read().splitlines()
cosmos = np.loadtxt("/pscratch/sd/g/gvalogia/wst/Emulator_cosmo.txt")

#New loop to evaluate WST for 85 abacus cosmologies produced by Enrique, for 350 HOD configurations each
n_HOD = 350
begin = 0
#for i in range(0):
for idx, cosmoid in enumerate(content_list[begin:]):
    st = cosmoid.split('_')[2]
    print (idx, st, 100*cosmos[begin+idx,2])
    #Load hubble factor needed for RSD
    hubble = 100*cosmos[begin+idx,2]
    redshift = 0.50
    scale_factor = 1 / (1 + redshift)
    for i in range(n_HOD):
        if (i<10):
            input_fn = '/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z0.5/yuan23_prior/'+str(st)+'_ph000/seed0/hod00'+str(i)+'.fits'
        elif (i<100):
            input_fn = '/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z0.5/yuan23_prior/'+str(st)+'_ph000/seed0/hod0'+str(i)+'.fits'
        else:
            input_fn = '/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z0.5/yuan23_prior/'+str(st)+'_ph000/seed0/hod'+str(i)+'.fits'
        hod = fitsio.read(input_fn)
        #Load real-space cartesian positions x,y,z from hod mock.
        x = hod['X'] + 1000.0
        y = hod['Y'] + 1000.0
        z = hod['Z'] + 1000.0
        #Add RSD, along z direction in this case
        z_RSD = (z + hod['VZ']/(hubble * scale_factor))%2000
        #And then save positions to file for post-processing
        #Posdat = (np.vstack((x,y,z))).T #for real-space
        Posdat = (np.vstack((x,y,z_RSD))).T #for RSD
        numpy.savetxt('/pscratch/sd/g/gvalogia/csv-exampletempder.txt', Posdat, fmt='%.7e')
        namesRSDder =['x', 'y', 'z']

        # read the data
        locals()['fder'+str(i)] = CSVCatalog('/pscratch/sd/g/gvalogia/csv-exampletempder.txt', namesRSDder)

        # combine x, y, z to Position, and add boxsize
        locals()['fder'+str(i)]['Position'] = locals()['fder'+str(i)]['x'][:, None] * [1, 0, 0] + locals()['fder'+str(i)]['y'][:, None] * [0, 1, 0] + locals()['fder'+str(i)]['z'][:, None] * [0, 0, 1]
        locals()['fder'+str(i)].attrs['BoxSize'] = 2000.0
        locals()['meshder'+str(i)] = locals()['fder'+str(i)].to_mesh(Nmesh=200, BoxSize=2000.0, window='tsc')#, compensated=True)
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
        #np.savetxt('/global/homes/g/gvalogia/MockChallenge/s012Abacsig08q08_LRG_z0.500_'+str(st)+'_ph000_hod'+str(i)+'.txt', np.real(smatavg), fmt='%.7e')

        del smatavg,smat
