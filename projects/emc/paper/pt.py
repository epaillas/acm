import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo.fiducial import DESI

cosmo = DESI()
edges = np.linspace(0., 0.3, 31)
k = (edges[:-1] + edges[1:]) / 2.
nmodes = 4. * np.pi / 3. * (edges[1:]**3 - edges[:-1]**3)
z = 1.
fo = cosmo.get_fourier()
pk = fo.pk_interpolator(of='delta_cb')(k, z=z)
b = 2.
f = fo.sigma8_z(z, of='theta_cb') / fo.sigma8_z(z, of='delta_cb')
shotnoise = 1 / 5e-4
volume = 1e10

ells = (0, 2, 4)
poles = []
poles.append((b**2 + 2. / 3. * f * b + 1. / 5. * f**2) * pk + shotnoise)
poles.append((4. / 3. * f * b + 4. / 7. * f**2) * pk)
poles.append(8. / 35 * f**2 * pk)
poles = np.array(poles, dtype='f8')

from pypower import PowerSpectrumStatistics
mean = PowerSpectrumStatistics(edges, k, poles, nmodes=nmodes, ells=ells, shotnoise_nonorm=shotnoise, statistic='multipole')
cov = [2. * (2. * np.pi)**3 / (2 * ell + 1) / (volume * nmodes) * poles[0]**2 for ell in ells]
cov = np.diag(np.concatenate(cov, axis=0))

rng = np.random.RandomState(seed=42)
mocks = []
for i in range(1000):
    tmp = mean.deepcopy()
    tmp.power_nonorm.flat[...] = rng.multivariate_normal(mean.power_nonorm.ravel(), cov)
    mocks.append(tmp)
data, mocks = mocks[0], mocks[1:]

print(data.shape, cov.shape)
print(data)

# from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles
# from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
# from desilike.likelihoods import ObservablesGaussianLikelihood
# from desilike.parameter import ParameterCollection
# from desilike import setup_logging


# template = DirectPowerSpectrumTemplate(z=0.5, fiducial='DESI')
# for param in ['omega_b', 'n_s']: template.params[param].update(fixed=True)
# theory = KaiserTracerPowerSpectrumMultipoles(template=template)
# theory.params['b1'].update(value=2.)
# observable = TracerPowerSpectrumMultipolesObservable(data=data, covariance=mocks,
#                                                      klim={0: [0.02, 0.2], 2: [0.02, 0.2]}, # fit monopole and quadrupole, between 0.02 and 0.2 h/Mpc
#                                                      theory=theory)
# likelihood = ObservablesGaussianLikelihood(observables=[observable])



# # NOTE: if we wanted to fit xi instead:
# # theory = KaiserTracerCorrelationFunctionMultipoles(template=template)
# # observable = ObservedTracerCorrelationFunction(data=data, covariance=mocks,
# #                                                slim={0: [40., 160], 2: [40., 160]}, # fit monopole and quadrupole, between 0.02 and 0.2 h/Mpc
# #                                                theory=theory)
# # The rest would be the same

# setup_logging()
# likelihood()  # just to initialize

# from desilike.emulators import Emulator, EmulatedCalculator, TaylorEmulatorEngine

# emulator = Emulator(theory, engine=TaylorEmulatorEngine(order={'*': 2, 'sn0': 1}))  # order 2 except for sn0 (order 1 is enough)
# emulator.set_samples()
# emulator.fit()
# emulator.plot(name='power')

# import os
# base_dir = '_tests'
# kaiser_emulator_fn = os.path.join(base_dir, 'kaiser_emulator.npy')
# emulator.save(kaiser_emulator_fn)

# from desilike.samplers import ZeusSampler

# # Let's just update the observable's theory, no need to redefine the observable & likelihood
# # (Internally the code will reinitialize all calculators that depend on observable)
# observable.init.update(theory=emulator.to_calculator())

# sampler = ZeusSampler(likelihood, save_fn='_tests/chain_fs_direct_*.npy', seed=42)
# sampler.run(check={'max_eigen_gr': 0.1})