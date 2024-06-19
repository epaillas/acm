import os
from pathlib import Path
import yaml
import numpy as np
from abacusnbody.hod import abacus_hod
from cosmoprimo.fiducial import AbacusSummit
from astropy.io import fits
from astropy.table import Table
import logging
import warnings
import sys
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class AbacusHOD:
    def __init__(self, varied_params, config_file=None, cosmo_idx=0, phase_idx=0,
        sim_type='base', redshift=0.5):
        self.logger = logging.getLogger('AbacusHOD')
        self.cosmo_idx = cosmo_idx
        self.phase_idx = phase_idx
        if sim_type not in ['base', 'small']:
            raise ValueError('Invalid sim_type. Must be either "base" or "small".')
        self.sim_type = sim_type
        self.boxsize = 2000 if sim_type == 'base' else 500
        self.redshift = redshift
        if config_file is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = Path(config_dir) /  'abacushod.yaml'
        config = yaml.safe_load(open(config_file))
        self.setup(config)
        self.check_params(varied_params)

    def setup(self, config):
        sim_params = config['sim_params']
        sim_dir, subsample_dir = self.abacus_simdirs()
        sim_params['sim_dir'] = sim_dir
        sim_params['subsample_dir'] = subsample_dir
        sim_params['sim_name'] = self.abacus_simname()
        sim_params['z_mock'] = self.redshift
        HOD_params = config['HOD_params']
        self.ball = abacus_hod.AbacusHOD(sim_params, HOD_params)
        self.ball.params['Lbox'] = self.boxsize
        self.cosmo = AbacusSummit(self.cosmo_idx)
        self.az = 1 / (1 + self.redshift)
        self.hubble = 100 * self.cosmo.efunc(self.redshift)
        self.logger.info(f'Processing {self.abacus_simname()} at z = {self.redshift}')

    def abacus_simdirs(self):
        if self.sim_type == 'small':
            sim_dir = '/global/cfs/cdirs/desi/cosmosim/Abacus/small/'
            subsample_dir = '/pscratch/sd/e/epaillas/summit_subsamples/boxes/small/'
        else:
            sim_dir = '/global/cfs/cdirs/desi/cosmosim/Abacus/'
            subsample_dir = '/pscratch/sd/s/sihany/summit_subsamples_cleaned_desi'
        return sim_dir, subsample_dir

    def abacus_simname(self):
        return f'AbacusSummit_{self.sim_type}_c{self.cosmo_idx:03}_ph{self.phase_idx:03}'

    def check_params(self, params):
        params = list(params)
        params = self.param_mapping(params)
        for param in params:
            if param not in self.ball.tracers['LRG'].keys():
                raise ValueError(f'Invalid parameter: {param}. Valid list '
                                 f'of parameters include: {list(self.ball.tracers["LRG"].keys())}')
        self.logger.info(f'Varied parameters: {params}.')
        self.varied_params = params
        default = {key: value for key, value in self.ball.tracers['LRG'].items() if key not in params}
        self.logger.info(f'Default parameters: {default}.')

    def run(self, hod_params, nthreads=1, tracer='LRG', tracer_density_mean=None,
        seed=None, save_fn=None, add_rsd=False):
        if seed == 0: seed = None
        if tracer not in ['LRG']:
            raise ValueError('Only LRGs are currently supported.')
        hod_params = self.param_mapping(hod_params)
        if set(hod_params.keys()) != set(self.varied_params):
            raise ValueError('Invalid HOD parameters. Must match the varied parameters.')
        for key in hod_params.keys():
            if key == 'sigma' and tracer == 'LRG':
                self.ball.tracers[tracer][key] = 10**hod_params[key]
            else:
                self.ball.tracers[tracer][key] = hod_params[key]
        self.ball.tracers[tracer]['ic'] = 1
        ngal_dict = self.ball.compute_ngal(Nthread=nthreads)[0]
        n_tracers= ngal_dict[tracer]
        if tracer_density_mean is not None:
            self.ball.tracers[tracer]['ic'] = min(
                1, tracer_density_mean * self.boxsize ** 3 / n_tracers
            )
        hod_dict = self.ball.run_hod(self.ball.tracers, self.ball.want_rsd, Nthread=nthreads, reseed=seed)
        # positions_dict = self.get_positions(hod_dict, tracer)
        self.format_catalog(hod_dict, save_fn, tracer, add_rsd)
        return hod_dict

    def format_catalog(self, hod_dict, save_fn=False, tracer='LRG', add_rsd=False):
        Ncent = hod_dict[tracer]['Ncent']
        hod_dict[tracer].pop('Ncent', None)
        is_central = np.zeros(len(hod_dict[tracer]['x']))
        is_central[:Ncent] += 1
        hod_dict[tracer]['is_cent'] = is_central
        # hod_dict[tracer]['nden'] = len(hod_dict[tracer]['x']) / self.boxsize**3
        hod_dict[tracer] = {k.upper():v  for k, v in hod_dict[tracer].items()}
        if add_rsd:
            hod_dict = self._add_rsd(hod_dict, tracer)
        if save_fn:
            table = Table(hod_dict[tracer])
            header = fits.Header({'N_cent': Ncent, 'gal_type': tracer, **self.ball.tracers[tracer]})
            myfits = fits.BinTableHDU(data=table, header=header)
            myfits.writeto(save_fn, overwrite=True)
            self.logger.info(f'Saving {save_fn}.')

    def param_mapping(self, hod_params: dict | list):
        """
        Map custom HOD parameters to Abacus HOD parameters. 

        Parameters
        ----------
        hod_params : dict or list
            Dictionary or list of HOD parameters.

        Returns
        -------
        dict or list
            Dictionary or list of AbacusHOD parameters.
        
        Raises
        ------
        ValueError
            If the type of hod_params is not dict or list.
        """
        
        # Add custom keys here if needed. 
        # Be careful to the one-to-one position mapping in the list !!
        abacus_keys = ['logM1', 'Acent', 'Asat', 'Bcent', 'Bsat']
        custom_keys = ['logM_1', 'A_cen', 'A_sat', 'B_cen', 'B_sat']
        
        # Check if custom keys are used
        if any(key in hod_params for key in custom_keys): # Same syntax for dict and list :)
            for abacus_key, custom_key in zip(abacus_keys, custom_keys): 
                if custom_key in hod_params: # Just in case not all custom keys are used
                    # Replace custom keys with Abacus keys
                    if type(hod_params) is dict:
                        hod_params[abacus_key] = hod_params.pop(custom_key)
                    elif type(hod_params) is list:
                        hod_params[hod_params.index(custom_key)] = abacus_key
                    else:
                        raise ValueError('Invalid type for hod_params. Must be either dict or list.')
                    
        return hod_params

    def _add_rsd(self, hod_dict, tracer='LRG'):
        data = hod_dict[tracer]
        x = data['X'] + self.boxsize / 2
        y = data['Y'] + self.boxsize / 2
        z = data['Z'] + self.boxsize / 2
        vx = data['VX']
        vy = data['VY']
        vz = data['VZ']
        x_rsd = (x + vx / (self.hubble * self.az)) % self.boxsize
        y_rsd = (y + vy / (self.hubble * self.az)) % self.boxsize
        z_rsd = (z + vz / (self.hubble * self.az)) % self.boxsize
        hod_dict[tracer]['X_RSD'] = x_rsd
        hod_dict[tracer]['Y_RSD'] = y_rsd
        hod_dict[tracer]['Z_RSD'] = z_rsd
        return hod_dict

# def run_hod(p, param_mapping, param_tracer, data_params, Ball, nthreads):
#     for key in param_mapping.keys():
#         mapping_idx = param_mapping[key]
#         tracer = param_tracer[key]
#         if key == 'sigma' and tracer == 'LRG':
#             Ball.tracers[tracer][key] = 10**p[mapping_idx]
#         else:
#             Ball.tracers[tracer][key] = p[mapping_idx]
#     Ball.tracers['LRG']['ic'] = 1
#     ngal_dict = Ball.compute_ngal(Nthread=nthreads)[0]
#     N_lrg = ngal_dict['LRG']
#     Ball.tracers['LRG']['ic'] = min(1, data_params['tracer_density_mean']['LRG']*Ball.params['Lbox']**3/N_lrg)
#     mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread=nthreads)
#     return mock_dict



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--start_hod", type=int, default=0)
#     parser.add_argument("--n_hod", type=int, default=1)
#     parser.add_argument("--start_cosmo", type=int, default=0)
#     parser.add_argument("--n_cosmo", type=int, default=1)
#     parser.add_argument("--start_phase", type=int, default=0)
#     parser.add_argument("--n_phase", type=int, default=1)

#     args = parser.parse_args()
#     start_hod = args.start_hod
#     n_hod = args.n_hod
#     start_cosmo = args.start_cosmo
#     n_cosmo = args.n_cosmo
#     start_phase = args.start_phase
#     n_phase = args.n_phase

#     setup_logging(level='WARNING')
#     boxsize = 2000
#     redshift = 0.5

#     # HOD configuration
#     hod_prior = 'yuan23'
#     config_dir = './'
#     config_fn = Path(config_dir, f'abacushod_config.yaml')
#     config = yaml.safe_load(open(config_fn))

#     for cosmo in range(start_cosmo, start_cosmo + n_cosmo):
#         mock_cosmo = AbacusSummit(cosmo)
#         az = 1 / (1 + redshift)
#         hubble = 100 * mock_cosmo.efunc(redshift)

#         hods_dir = Path(f'/pscratch/sd/e/epaillas/emc/hod_params/{hod_prior}/')
#         hods_fn = hods_dir / f'hod_params_{hod_prior}_c{cosmo:03}.csv'
#         hod_params = np.genfromtxt(hods_fn, skip_header=1, delimiter=',')

#         for phase in range(start_phase, start_phase + n_phase):
#             sim_fn = f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}'
#             config['sim_params']['sim_name'] = sim_fn
#             newBall, param_mapping, param_tracer, data_params = setup_hod(config)

#             fig, ax = plt.subplots()
#             for hod in range(start_hod, start_hod + n_hod):
#                 print(f'c{cosmo:03} ph{phase:03} hod{hod}')

#                 hod_dict = run_hod(hod_params[hod], param_mapping, param_tracer,
#                               data_params, newBall, nthreads=256)

#                 x, y, z, x_rsd, y_rsd, z_rsd = read_positions(hod_dict)

#                 data_positions = {
#                     'x': x, 'y': y, 'z': z,
#                     'x_rsd': x_rsd, 'y_rsd': y_rsd, 'z_rsd': z_rsd,
#                 }
#                 output_dir = f'/pscratch/sd/e/epaillas/emc/hods/z0.5/{hod_prior}_prior2/c{cosmo:03}_ph{phase:03}'
#                 Path(output_dir).mkdir(parents=True, exist_ok=True)
#                 output_fn = Path(output_dir) / f'hod{hod:03}.npy'
#                 # np.save(output_fn, data_positions)

