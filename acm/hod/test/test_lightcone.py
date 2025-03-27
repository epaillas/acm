from acm.hod import HODLatinHypercube
from acm.hod import LightconeHOD
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

lc = LightconeHOD(varied_params=hod_params.keys(), zrange=[0.1, 0.5])

hod = lc.run(hod_params, nthreads=1, tracer='LRG', make_randoms=False, add_weights=False,
       seed=None, save_fn=None, full_sky=False, alpha_rand=1, apply_radial_mask=False)

z = hod['LRG']['Z']

print(z.min(), z.max())