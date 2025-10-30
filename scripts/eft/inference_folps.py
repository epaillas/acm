from desilike.theories.galaxy_clustering import (DirectPowerSpectrumTemplate,
                                                 FOLPSAXTracerPowerSpectrumMultipoles,
                                                 FOLPSAXTracerCorrelationFunctionMultipoles)
from desilike.observables.galaxy_clustering import (TracerPowerSpectrumMultipolesObservable,
                                                    TracerCorrelationFunctionMultipolesObservable)
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.likelihoods.base import BaseLikelihood
from desilike.theories import Cosmoprimo
from desilike import setup_logging
from cosmoprimo.fiducial import DESI

from acm.observables import emc

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import jax.numpy as jnp


class ThetaStarLikelihood(BaseLikelihood):
    """
    Likelihood to incorporate a constraint on the angular size
    of the BAO at recombination.
    """
    def initialize(self, cosmo=None):
        self.cosmo = cosmo
        self.data = 1.04092
        self.covariance = 0.00031**2
        super().initialize(name='theta_MC_100')

    def calculate(self):
        self.loglikelihood = -0.5 * (self.cosmo['theta_MC_100'] - self.data)**2 / self.covariance

def read_emc_data(summary='power'):
    """
    Read data from the Emulator Mock Challenge to use as a test.
    """
    if summary == 'correlation':
        emc_dataset = emc.GalaxyCorrelationFunctionMultipoles(
            train=True,
            select_mocks={'cosmo_idx': 0, 'hod_idx': 30},
            select_coordinates={'multipoles': [0, 2]},
            slice_coordinates={'s': [30, 150]}
        )
        x = emc_dataset.separation
        x = x[(x >= 30) & (x <= 150)]
    elif summary == 'power':
        emc_dataset = emc.GalaxyPowerSpectrumMultipoles(
            train=True,
            select_mocks={'cosmo_idx': 0, 'hod_idx': 30},
            select_coordinates={'multipoles': [0, 2]},
            slice_coordinates={'k': [0, 0.2]}
        )
        x = emc_dataset.separation
        x = x[x < 0.2]
        # k = emc_dataset.separation[:int(len(data)/2)]
    data = emc_dataset.lhc_y
    cov = emc_dataset.get_covariance_matrix(divide_factor=64)
    error = np.sqrt(np.diag(cov))
    
    print(x)
    return x, data, cov

def setup_likelihood():
    """
    Setup the desilike likelihood for a full-shape power spectrum
    fit with a perturbation theory model.
    """
    cosmo = Cosmoprimo(fiducial='DESI', engine='class', lensing=True)

    if 'base' in cosmo_model:
        cosmo.params['omega_b'].update(prior={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037})
        # cosmo.params['omega_b'].update(prior={'dist': 'uniform', 'limits': [0.0207, 0.0243]})
        cosmo.params['omega_cdm'].update(prior={'dist': 'uniform', 'limits': [0.1032, 0.140]})
        cosmo.params['logA'].update(prior={'dist': 'uniform', 'limits': [2.0, 4.0]})
        cosmo.params['h'].update(prior={'dist': 'uniform', 'limits': [0.1, 1.0]})
        # cosmo.params['n_s'].update(prior={'dist': 'uniform', 'limits': [0.9012, 1.025]})

        # derived parameters
        cosmo.init.params['sigma8_m'] = dict(derived=True, latex=r'$\sigma_8$', fixed=False)
        cosmo.init.params['Omega0_m'] = dict(derived=True, latex=r'$\Omega_{\rm m}$', fixed=False)
        
        # fixed parameters
        cosmo.params['tau_reio'].update(fixed=True)
        cosmo.params['n_s'].update(fixed=True)

    if 'w0' in cosmo_model:
        cosmo.init.params['w0_fld'].update(prior={'limits': [-3.0, 1.0]}, latex='w_0', fixed=False) 
    if 'wa' in cosmo_model:
        cosmo.init.params['wa_fld'].update(prior={'limits': [-3., 2.]}, latex='w_a', fixed=False)


    template = DirectPowerSpectrumTemplate(z=0.5, cosmo=cosmo)
    if summary == 'correlation':
        theory = FOLPSAXTracerCorrelationFunctionMultipoles(template=template, ells=(0, 2),
                                                            s=s, tracer='LRG',
                                                            prior_basis='physical')
        observable = TracerCorrelationFunctionMultipolesObservable(data=data, covariance=cov,
                                                                s=s, slim={0: (s.min(), s.max()),
                                                                           2: (s.min(), s.max())},    
                                                                ells=[0, 2], theory=theory)
    elif summary == 'power':
        theory = FOLPSAXTracerPowerSpectrumMultipoles(template=template, ells=(0, 2), k=k,
                                                      tracer='LRG', prior_basis='physical')
        observable = TracerPowerSpectrumMultipolesObservable(data=data, covariance=cov,
                                                            k=k, klim={0: (k.min(), k.max()),
                                                                       2: (k.min(), k.max())},
                                                         ells=[0, 2], theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])

    if thetas_constraint:
        cosmo_capse = Cosmoprimo(fiducial='DESI', engine='capse', lensing=True)
        likelihood += ThetaStarLikelihood(cosmo=cosmo_capse)

    return theory, observable, likelihood

def setup_emulator():
    """
    Setup full-shape emulator.
    """
    from desilike.emulators import Emulator, EmulatedCalculator, TaylorEmulatorEngine

    emulator_fn = Path(f'emulators/emulator_folps_cosmo-{cosmo_model}_{summary}-test.npy')
    if emulator_fn.exists():
        emulator = Emulator.load(emulator_fn)
    else:
        # emulator = Emulator(theory, engine=TaylorEmulatorEngine(order={'*': 4, 'sn0': 1}))
        emulator = Emulator(theory, engine=TaylorEmulatorEngine(order={'*': 3}))
        emulator.set_samples()
        emulator.fit()
        emulator.save(emulator_fn)

    observable.init.update(theory=emulator.to_calculator())
    for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*']):
        if param.varied: param.update(derived='.auto')
    likelihood.all_params['w'] = {'derived': '{w0_fld} + {wa_fld}', 'prior': {'limits': [-100., 0.]}, 'drop': True} 
    return emulator

def run_mcmc():
    """
    Posterior sampling.
    """
    from desilike.samplers import EmceeSampler
    from desilike.samples import plotting

    save_fn = [f'chains/chain_folps_{handle_str}_{i}-test.npy' for i in range(nchains)]

    sampler = EmceeSampler(likelihood, nwalkers=40, save_fn=save_fn, seed=42)
    chains = sampler.run(min_iterations=5_000, max_iterations=50_000, check={'max_eigen_gr': 0.01})

    return chains

if __name__ == "__main__":
    """
    Main function to run the inference.
    """
    setup_logging()
    nchains = 4
    cosmo_model = 'base'
    thetas_constraint = False
    summary = 'correlation'

    # handle string for saving and loading files
    if thetas_constraint:
        handle_str = f'cosmo-{cosmo_model}_{summary}_thetas'
    else:
        handle_str = f'cosmo-{cosmo_model}_{summary}'

    handle_str += 'test'

    # read data
    if summary == 'correlation':
        s, data, cov = read_emc_data(summary='correlation')
    elif summary == 'power':
        k, data, cov = read_emc_data()
    else:
        raise ValueError(f"Observable '{summary}' is not supported.")

    # likelihood
    theory, observable, likelihood = setup_likelihood()

    # Taylor emulator
    emulator = setup_emulator()

    # run MCMC
    chains = run_mcmc()