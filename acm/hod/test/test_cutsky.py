from acm.hod import HODLatinHypercube
from acm.hod import CutskyHOD
from acm import setup_logging

from sunbird.inference.priors import Yuan23

import warnings
warnings.filterwarnings("ignore")

setup_logging()

# infer parameter ranges from the priors adopted in arXiv:2203.11963
ranges = Yuan23().ranges

# sample and distribute on a Latin hypercube
lhc = HODLatinHypercube(ranges)
lhc.sample(n=1)

# pick one set of HOD parameters from the LHC
hod_params = lhc.params
hod_params = {key: hod_params[key][0] for key in hod_params.keys()}

lc = CutskyHOD(varied_params=hod_params.keys())

hod = lc.run(hod_params, nthreads=1, seed=0, generate_randoms=False, alpha_randoms=5,
        randoms_seed=42, replications=True)

print(hod)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(hod['RA'], hod['DEC'], s=0.1)
plt.savefig('cutsky_footprint.png', dpi=300, bbox_inches='tight')