import os
from pathlib import Path
import yaml
import numpy as np
from abacusnbody.hod import abacus_hod
from cosmoprimo.fiducial import AbacusSummit
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
        self.data_params = config['data_params']
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
        params = self.param_mapping(params)
        params = list(params)
        for param in params:
            if param not in self.ball.tracers['LRG'].keys():
                raise ValueError(f'Invalid parameter: {param}. Valid list '
                                 f'of parameters include: {list(self.ball.tracers["LRG"].keys())}')
        self.logger.info(f'Varied parameters: {params}.')
        default = {key: value for key, value in self.ball.tracers['LRG'].items() if key not in params}
        self.logger.info(f'Default parameters: {default}.')

    def run(self, hod_params, nthreads=1, tracer_type='LRG'):
        if tracer_type not in ['LRG']:
            raise ValueError('Only LRGs are currently supported.')
        hod_params = self.param_mapping(hod_params)
        for key in hod_params.keys():
            if key == 'sigma' and tracer_type == 'LRG':
                self.ball.tracers[tracer_type][key] = 10**hod_params[key]
            else:
                self.ball.tracers[tracer_type][key] = hod_params[key]
        self.ball.tracers[tracer_type]['ic'] = 1
        ngal_dict = self.ball.compute_ngal(Nthread=nthreads)[0]
        N_lrg = ngal_dict[tracer_type]
        self.ball.tracers[tracer_type]['ic'] = min(
            1, self.data_params['tracer_density_mean'][tracer_type] * self.ball.params['Lbox']**3/N_lrg
        )
        self.hod_dict = self.ball.run_hod(self.ball.tracers, self.ball.want_rsd, Nthread=nthreads)
        return self.hod_positions(self.hod_dict, tracer_type)

    def param_mapping(self, hod_params: dict):
        """
        Map custom HOD parameters to Abacus HOD parameters.

        Parameters
        ----------
        hod_params : dict
            Dictionary of HOD parameters.

        Returns
        -------
        dict
            Dictionary of Abacus HOD parameters, if return_keys is False.

        Raises
        ------
        ValueError
            If HOD parameters are not provided as a dictionary.
        """
        
        # Add custom keys here if needed. 
        # Be careful to the one-to-one position mapping in the list !!
        abacus_keys = ['logM1', 'Acent', 'Asat', 'Bcent', 'Bsat']
        custom_keys = ['logM_1', 'A_cen', 'A_sat', 'B_cen', 'B_sat']
        
        if type(hod_params) is not dict:
            raise ValueError('HOD parameters must be provided as a dictionary.')
            
        if any(key in hod_params for key in custom_keys): # Check if custom keys are used
            for abacus_key, custom_key in zip(abacus_keys, custom_keys): 
                if custom_key in hod_params: # Just in case not all custom keys are used
                    hod_params[abacus_key] = hod_params.pop(custom_key) # Replace custom keys with Abacus keys
        
        return hod_params

    def hod_positions(self, hod_dict, tracer_type='LRG'):
        data = hod_dict[tracer_type]
        x = data['x'] + self.boxsize / 2
        y = data['y'] + self.boxsize / 2
        z = data['z'] + self.boxsize / 2
        vx = data['vx']
        vy = data['vy']
        vz = data['vz']
        x_rsd = (x + vx / (self.hubble * self.az)) % self.boxsize
        y_rsd = (y + vy / (self.hubble * self.az)) % self.boxsize
        z_rsd = (z + vz / (self.hubble * self.az)) % self.boxsize
        positions = {
            'X': x, 'Y': y, 'Z': z,
            'X_RSD': x_rsd, 'Y_RSD': y_rsd, 'Z_RSD': z_rsd,
        }
        return positions

# def run_hod(p, param_mapping, param_tracer, data_params, Ball, nthreads):
#     for key in param_mapping.keys():
#         mapping_idx = param_mapping[key]
#         tracer_type = param_tracer[key]
#         if key == 'sigma' and tracer_type == 'LRG':
#             Ball.tracers[tracer_type][key] = 10**p[mapping_idx]
#         else:
#             Ball.tracers[tracer_type][key] = p[mapping_idx]
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

