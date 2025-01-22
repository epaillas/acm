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

from acm.data.paths import LRG_Abacus_DM as DM_DICT

class BoxHOD:
    def __init__(
        self,
        varied_params, 
        config_file=None, 
        cosmo_idx=0, 
        phase_idx=0,
        sim_type='base', 
        redshift=0.5,
        DM_DICT=DM_DICT):
        # TODO : document this !!
        
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
            config_file = Path(config_dir) /  'box.yaml'
        config = yaml.safe_load(open(config_file))
        self.setup(config, DM_DICT)
        self.check_params(varied_params)

    def setup(self, config, DM_DICT): # Will override most of the config file !
        sim_params = config['sim_params']
        sim_dir, subsample_dir = self.abacus_simdirs(DM_DICT) 
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

    def abacus_simdirs(self, DM_DICT):
        sim_dir = DM_DICT[self.sim_type]['sim_dir']
        subsample_dir = DM_DICT[self.sim_type]['subsample_dir']
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

class LightconeHOD:
    def __init__(self, varied_params, config_file=None, cosmo_idx=0, phase_idx=0,
        zrange=[0.4, 0.8]):
        self.logger = logging.getLogger('LightconeHOD')
        self.cosmo_idx = cosmo_idx
        self.phase_idx = phase_idx
        self.sim_type = 'base'
        self.zrange = zrange
        self.boxsize = 2000
        if config_file is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = Path(config_dir) /  'lightcone.yaml'
        config = yaml.safe_load(open(config_file))
        self.setup(config)
        self.check_params(varied_params)

    @property
    def snap_redshifts(self):
        return [0.400, 0.450, 0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950, 1.025, 1.100]

    def snap_in_zrange(self):
        snap_min = np.abs(np.array(self.snap_redshifts) - self.zrange[0]).argmin()
        snap_max = np.abs(np.array(self.snap_redshifts) - self.zrange[1]).argmin()
        snaps = self.snap_redshifts[snap_min:snap_max+2]  # Include an extra snapshot at high-z to avoid edge effects
        self.logger.info(f'Lightcone composed of snapshots at z: {snaps}.')
        return snaps
        # return [z for z in self.snap_redshifts if z >= self.zrange[0] and z <= self.zrange[1]]

    def abacus_simdirs(self):
        sim_dir = '/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/halo_light_cones/'
        subsample_dir = '/pscratch/sd/e/epaillas/summit_subsamples/lightcones/'
        return sim_dir, subsample_dir

    def abacus_simname(self):
        return f'AbacusSummit_{self.sim_type}_c{self.cosmo_idx:03}_ph{self.phase_idx:03}'

    def setup(self, config):
        sim_params = config['sim_params']
        sim_dir, subsample_dir = self.abacus_simdirs()
        sim_params['sim_dir'] = sim_dir
        sim_params['subsample_dir'] = subsample_dir
        sim_params['sim_name'] = self.abacus_simname()
        HOD_params = config['HOD_params']
        self.balls = []
        for znap in self.snap_in_zrange():
            sim_params['z_mock'] = znap
            self.balls += [abacus_hod.AbacusHOD(sim_params, HOD_params)]
        self.cosmo = AbacusSummit(self.cosmo_idx)
        # self.az = 1 / (1 + self.redshift)
        # self.hubble = 100 * self.cosmo.efunc(self.redshift)
        self.logger.info(f'Processing {self.abacus_simname()} at zrange = {self.zrange}')

    def check_params(self, params):
        params = list(params)
        params = self.param_mapping(params)
        for param in params:
            if param not in self.balls[0].tracers['LRG'].keys():
                raise ValueError(f'Invalid parameter: {param}. Valid list '
                                 f'of parameters include: {list(self.balls[0].tracers["LRG"].keys())}')
        self.logger.info(f'Varied parameters: {params}.')
        self.varied_params = params
        default = {key: value for key, value in self.balls[0].tracers['LRG'].items() if key not in params}
        self.logger.info(f'Default parameters: {default}.')

    def run(self, hod_params, nthreads=1, tracer='LRG', make_randoms=False, add_weights=False,
        seed=None, save_fn=None, full_sky=False, alpha_rand=1, apply_nz=False):
        if seed == 0: seed = None
        if tracer not in ['LRG']:
            raise ValueError('Only LRGs are currently supported.')
        hod_params = self.param_mapping(hod_params)
        if set(hod_params.keys()) != set(self.varied_params):
            raise ValueError('Invalid HOD parameters. Must match the varied parameters.')
        for i, ball in enumerate(self.balls):
            for key in hod_params.keys():
                if key == 'sigma' and tracer == 'LRG':
                    ball.tracers[tracer][key] = 10**hod_params[key]
                else:
                    ball.tracers[tracer][key] = hod_params[key]
            ball.tracers[tracer]['ic'] = 1
            if i == 0:
                hod_dict = ball.run_hod(ball.tracers, ball.want_rsd, Nthread=nthreads, reseed=seed)
            else:
                hod_dict_i = ball.run_hod(ball.tracers, ball.want_rsd, Nthread=nthreads, reseed=seed)
                for key in hod_dict_i[tracer].keys():
                    if key == 'Ncent': 
                        hod_dict[tracer][key] += hod_dict_i[tracer][key]
                    else:
                        hod_dict[tracer][key] = np.concatenate([hod_dict[tracer][key], hod_dict_i[tracer][key]])
            # positions_dict = self.get_positions(hod_dict, tracer)
        self.format_catalog(hod_dict, save_fn, tracer, full_sky, apply_nz)
        if make_randoms:
            zmin = hod_dict[tracer]['Z'].min()
            zmax = hod_dict[tracer]['Z'].max()
            nbar = self.get_data_nbar(hod_dict, tracer, full_sky)
            randoms_dict = self._make_randoms(nbar=nbar, zmin=zmin, zmax=zmax, apply_nz=apply_nz,
                                              full_sky=full_sky, alpha=alpha_rand, tracer=tracer)
            return hod_dict, randoms_dict
        return hod_dict

    def recenter_box(self, hod_dict, tracer='LRG'):
        data = hod_dict[tracer]
        data['X'] += 990
        data['Y'] += 990
        data['Z'] += 990

    def remove_outbounds(self, hod_dict, tracer='LRG'):
        mask = (hod_dict[tracer]['X'] >= 0) & (hod_dict[tracer]['Y'] >= 0) & (hod_dict[tracer]['Z'] >= 0)
        for key in hod_dict[tracer].keys():
            hod_dict[tracer][key] = hod_dict[tracer][key][mask]

    def apply_zcut(self, hod_dict, zmin, zmax, tracer='LRG'):
        self.logger.info(f'Applying redshift cut: {zmin} < z < {zmax}.')
        mask = (hod_dict[tracer]['Z'] >= zmin) & (hod_dict[tracer]['Z'] <= zmax)
        for key in hod_dict[tracer].keys():
            hod_dict[tracer][key] = hod_dict[tracer][key][mask]

    def get_sky_coordinates(self, hod_dict, tracer='LRG'):
        from mockfactory import cartesian_to_sky, DistanceToRedshift
        data = hod_dict[tracer]
        dist, ra, dec = cartesian_to_sky(np.c_[data['X'], data['Y'], data['Z']])
        d2z = DistanceToRedshift(self.cosmo.comoving_radial_distance)
        redshift = d2z(dist)
        hod_dict[tracer]['RA'] = ra
        hod_dict[tracer]['DEC'] = dec
        hod_dict[tracer]['Redshift'] = redshift

    def make_full_sky(self, hod_dict, tracer='LRG'):
        data = hod_dict[tracer]
        x = data['X']
        y = data['Y']
        z = data['Z']
        pos = np.c_[x, y, z]
        pos = np.concatenate(
            [pos,
             np.c_[-x, y, z],
             np.c_[x, -y, z],
                np.c_[x, y, -z],
                np.c_[-x, -y, z],
                np.c_[-x, y, -z],
                np.c_[x, -y, -z],
                np.c_[-x, -y, -z]
            ]
        )
        hod_dict[tracer]['X'] = pos[:, 0]
        hod_dict[tracer]['Y'] = pos[:, 1]
        hod_dict[tracer]['Z'] = pos[:, 2]
        for key in ['MASS', 'ID', 'IS_CENT']:
            if key in hod_dict[tracer]:
                hod_dict[tracer][key] = np.tile(hod_dict[tracer][key], 8)

    def get_data_nbar(self, hod_dict, tracer='LRG', full_sky=False):
        """
        Compute the number density of the data catalog, which is defined
        by an octant of the spherical shell delimited by the redshift cuts.
        """
        data = hod_dict[tracer]
        dmin, dmax = self.cosmo.comoving_radial_distance(self.zrange)
        volume = 4/3 * np.pi * (dmax**3 - dmin**3)
        correction = 1 if full_sky else 8  # divide by 8 if only using a sky octant
        nbar = len(data['Z']) / (volume / correction)
        return nbar

    def format_catalog(self, hod_dict, save_fn=False, tracer='LRG', full_sky=False, apply_nz=False):
        Ncent = hod_dict[tracer]['Ncent']
        hod_dict[tracer].pop('Ncent', None)
        is_central = np.zeros(len(hod_dict[tracer]['x']))
        is_central[:Ncent] += 1
        hod_dict[tracer]['is_cent'] = is_central
        hod_dict[tracer] = {k.upper():v  for k, v in hod_dict[tracer].items()}
        self.recenter_box(hod_dict, tracer)
        self.remove_outbounds(hod_dict, tracer)
        if full_sky: self.make_full_sky(hod_dict)
        self.get_sky_coordinates(hod_dict)
        self.drop_cartesian(hod_dict)
        self.apply_zcut(hod_dict, self.zrange[0], self.zrange[1])
        data_nbar = self.get_data_nbar(hod_dict, tracer, full_sky)
        self.logger.info(f'Raw data nbar: {data_nbar}' )
        if apply_nz:
            nz_filename = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/{tracer}_NGC_nz.txt'
            self.apply_radial_mask(hod_dict, nz_filename, norm=1/data_nbar)
            self.logger.info(f'Downsampled data nbar: {self.get_data_nbar(hod_dict, tracer, full_sky)}' )
        if save_fn:
            table = Table(hod_dict[tracer])
            header = fits.Header({'N_cent': Ncent, 'gal_type': tracer, **self.ball.tracers[tracer]})
            myfits = fits.BinTableHDU(data=table, header=header)
            myfits.writeto(save_fn, overwrite=True)
            self.logger.info(f'Saving {save_fn}.')

    def drop_cartesian(self, hod_dict, tracer='LRG'):
        hod_dict[tracer].pop('X')
        hod_dict[tracer].pop('Y')
        hod_dict[tracer].pop('Z')
        if 'VX' in hod_dict[tracer]: hod_dict[tracer].pop('VX')
        if 'VY' in hod_dict[tracer]: hod_dict[tracer].pop('VY')
        if 'VZ' in hod_dict[tracer]: hod_dict[tracer].pop('VZ')
        hod_dict[tracer]['Z'] = hod_dict[tracer].pop('Redshift')

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

    def _make_randoms(self, nbar, zmin, zmax, alpha=1, full_sky=False, apply_nz=False, tracer='LRG'):
        from mockfactory import RandomBoxCatalog
        self.logger.info(f'Generating random catalog.')
        # nbar = 6e-4  # hardcoded for LRG
        pos = RandomBoxCatalog(
            boxsize=self.boxsize, boxcenter=self.boxsize/2, nbar=nbar*alpha, seed=42
        )['Position']
        randoms = {tracer: {'X': pos[:, 0], 'Y': pos[:, 1], 'Z': pos[:, 2]}}
        if full_sky: self.make_full_sky(randoms)
        self.get_sky_coordinates(randoms)
        self.drop_cartesian(randoms)
        zmask = (randoms[tracer]['Z'] >= zmin) & (randoms[tracer]['Z'] <= zmax)
        for key in randoms[tracer].keys():
            randoms[tracer][key] = randoms[tracer][key][zmask]
        if apply_nz:
            nz_filename = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/{tracer}_NGC_nz.txt'
            self.apply_radial_mask(randoms, nz_filename)
        return randoms

    
    def apply_radial_mask(self, hod_dict, nz_filename, norm=None, tracer='LRG'):
        # example from https://github.com/cosmodesi/mockfactory/
        from mockfactory import TabulatedRadialMask
        self.logger.info(f'Applying radial mask from {nz_filename}.')
        # Load nz
        zbin_mid, n_z = np.genfromtxt(nz_filename, skip_header=3, usecols=(0, 3)).T
        # if norm is not None and 1/norm <= n_z.max():
        #     norm = None
        mask_radial = TabulatedRadialMask(z=zbin_mid, nbar=n_z, interp_order=2, norm=norm)
        for key in hod_dict[tracer].keys():
            hod_dict[tracer][key] = hod_dict[tracer][key][mask_radial(hod_dict[tracer]['Z'], seed=42)]


# class CutskyHOD:
#     """
#     Patch together cubic boxes to form a pseudo-lightcone.
#     """
#     def __init__(self, varied_params, config_file=None, cosmo_idx=0, phase_idx=0,
#         zrange=[0.4, 0.8], snapshots=[0.5]):
#         self.logger = logging.getLogger('CutskyHOD')
#         self.cosmo_idx = cosmo_idx
#         self.phase_idx = phase_idx
#         self.sim_type = 'base'
#         self.zrange = zrange
#         self.boxsize = 2000
#         if config_file is None:
#             config_dir = os.path.dirname(os.path.abspath(__file__))
#             config_file = Path(config_dir) /  'box.yaml'
#         config = yaml.safe_load(open(config_file))
#         self.setup(config)
#         self.check_params(varied_params)

#     def abacus_simdirs(self):
#         if self.sim_type == 'small':
#             sim_dir = '/global/cfs/cdirs/desi/cosmosim/Abacus/small/'
#             subsample_dir = '/pscratch/sd/e/epaillas/summit_subsamples/boxes/small/'
#         else:
#             sim_dir = '/global/cfs/cdirs/desi/cosmosim/Abacus/'
#             subsample_dir = '/pscratch/sd/e/epaillas/summit_subsamples/boxes/base/'
#         return sim_dir, subsample_dir

#     def abacus_simname(self):
#         return f'AbacusSummit_{self.sim_type}_c{self.cosmo_idx:03}_ph{self.phase_idx:03}'

#     def setup(self, config):
#         sim_params = config['sim_params']
#         sim_dir, subsample_dir = self.abacus_simdirs()
#         sim_params['sim_dir'] = sim_dir
#         sim_params['subsample_dir'] = subsample_dir
#         sim_params['sim_name'] = self.abacus_simname()
#         HOD_params = config['HOD_params']
#         self.balls = []
#         for znap in self.snapshots:
#             self.logger.info(f'Processing {self.abacus_simname()} at z = {self.redshift}')
#             sim_params['z_mock'] = znap
#             ball = abacus_hod.AbacusHOD(sim_params, HOD_params)
#             ball.params['Lbox'] = self.boxsize
#             self.balls += [ball]
#         self.cosmo = AbacusSummit(self.cosmo_idx)
#         self.az = 1 / (1 + self.redshift)
#         self.hubble = 100 * self.cosmo.efunc(self.redshift)

#     def check_params(self, params):
#         params = list(params)
#         params = self.param_mapping(params)
#         for param in params:
#             if param not in self.ball.tracers['LRG'].keys():
#                 raise ValueError(f'Invalid parameter: {param}. Valid list '
#                                  f'of parameters include: {list(self.ball.tracers["LRG"].keys())}')
#         self.logger.info(f'Varied parameters: {params}.')
#         self.varied_params = params
#         default = {key: value for key, value in self.ball.tracers['LRG'].items() if key not in params}
#         self.logger.info(f'Default parameters: {default}.')

#     def run(self, hod_params, nthreads=1, tracer='LRG', make_randoms=False, add_weights=False,
#         seed=None, save_fn=None, full_sky=False, alpha_rand=1, apply_nz=False):
#         if seed == 0: seed = None
#         if tracer not in ['LRG']:
#             raise ValueError('Only LRGs are currently supported.')
#         hod_params = self.param_mapping(hod_params)
#         if set(hod_params.keys()) != set(self.varied_params):
#             raise ValueError('Invalid HOD parameters. Must match the varied parameters.')
#         for i, ball in enumerate(self.balls):
#             for key in hod_params.keys():
#                 if key == 'sigma' and tracer == 'LRG':
#                     ball.tracers[tracer][key] = 10**hod_params[key]
#                 else:
#                     ball.tracers[tracer][key] = hod_params[key]
#             ball.tracers[tracer]['ic'] = 1
#             if i == 0:
#                 hod_dict = ball.run_hod(ball.tracers, ball.want_rsd, Nthread=nthreads, reseed=seed)
#             else:
#                 hod_dict_i = ball.run_hod(ball.tracers, ball.want_rsd, Nthread=nthreads, reseed=seed)
#                 for key in hod_dict_i[tracer].keys():
#                     if key == 'Ncent': 
#                         hod_dict[tracer][key] += hod_dict_i[tracer][key]
#                     else:
#                         hod_dict[tracer][key] = np.concatenate([hod_dict[tracer][key], hod_dict_i[tracer][key]])
#         self.format_catalog(hod_dict, save_fn, tracer, full_sky, apply_nz)
#         return hod_dict

#     def get_box_replications(pos, vel, mappings=[-1, 0, 1]):
#         rep_pos = []
#         rep_vel = []
#         for i in mappings:
#             for j in mappings:
#                 for k in mappings:
#                     rep_pos.append(pos + [boxsize * idx for idx in [i, j, k]])
#                     rep_vel.append(vel)
#         rep_pos = np.concatenate(rep_pos)
#         rep_vel = np.concatenate(rep_vel)
#         return rep_pos, rep_vel

#     def photometric_region_center(region):
#         if region == 'N':
#             ra, dec = 192.3, 56.0
#         elif region in ['N+DN', 'N+SNGC']:
#             ra, dec = 192, 35
#         elif region in ['DN', 'SNGC']:
#             ra, dec = 192, 13.0
#         elif region in ['DS', 'SSGC']:
#             ra, dec = 6.4, 5.3
#         else:
#             ValueError(f'photometric_region_center is not defined for region={region}')
#         return ra, dec

#     def apply_rsd_and_cutsky(catalog, dmin, dmax, rsd_factor, center_ra=0, center_dec=0):
#         """
#         Rotate the box to the final position, apply RSD and masks.

#         Note
#         ----
#         RSD needs to be applied before applying the distance cuts.

#         Parameters
#         ----------
#         catalog: BoxCatalog
#             Box containing the simulation. Must be large enough for the desired ``dmax`` and ``dmin``.

#         dmin : float
#             Minimal distance desired for the cutsky. Can be computed with `cosmo.comoving_radial_distance(zmin)`.

#         dmax : float
#             Maximal distance desired for the cutsky. Can be computed with `cosmo.comoving_radial_distance(zmax)`.

#         rsd_factor: float
#             Factor to apply to ``catalog.velocity`` to obtain RSD displacement in positions units, to be added to ``catalog.position``.
#             It depends on the choice of velocity units in ``catalog``.

#         center_ra, center_dec : float, default=0.
#             Add angles to rotate the box. The box is centered around (RA, Dec) = (center_ra, center_dec).

#         Returns
#         -------
#         cutsky : CutskyCatalog
#             Catalog with desired cutsky and RSD positions.
#         """
#         from mockfactory import box_to_cutsky, utils

#         # Collect limit for the cone
#         print(catalog.boxsize)
#         drange, rarange, decrange = box_to_cutsky(catalog.boxsize, dmax, dmin=dmin)

#         # Slice rarange et decrange:
#         # rarange = np.array(rarange) + center_ra
#         # decrange = np.array(decrange) + center_dec

#         # Collect isometry (transform) and masks to be applied
#         isometry, mask_radial, mask_angular = catalog.isometry_for_cutsky(drange, rarange, decrange)
#         # First move data to its final position
#         data_cutsky = catalog.cutsky_from_isometry(isometry, rdd=None)
#         # For data, we apply RSD *before* distance cuts
#         data_cutsky['RSDPosition'] = data_cutsky.rsd_position(f=rsd_factor)
#         # Collect distance, ra, dec
#         data_cutsky['DISTANCE'], data_cutsky['RA'], data_cutsky['DEC'] = utils.cartesian_to_sky(data_cutsky['RSDPosition'])
#         # Apply selection function (purely geometric)
#         mask = mask_radial(data_cutsky['DISTANCE']) & mask_angular(data_cutsky['RA'], data_cutsky['DEC'])
#         return data_cutsky
#         # return data_cutsky[mask]

#     def apply_radial_mask(cutsky, zmin=0., zmax=6., nz_filename='nz_qso_final.dat',
#                         apply_redshift_smearing=False, tracer_smearing='QSO',
#                         cosmo=None, seed=145):
#         """
#         Match the input n(z) distribution between ``zmin`` and ``zmax``.
#         Here, we extract the largest number of galaxy as possible (as default).

#         Parameters
#         ----------
#         cutsky: CutskyCatalog
#             Catalog containing at least a column 'Z'.

#         zmin: float, default=0.
#             Minimal redshift to consider in the n(z).

#         zmax: float, default=6.
#             Maximum redshift to consider in the n(z).

#         nz_filename: string, default='nz_qso_final.dat'
#             Where the n(z) is saved, in ``cutsky.position`` units, e.g. (Mpc/h)^(-3). For now, only the final TS format is accepted.

#         apply_redshift_smearing: bool, default=False
#             If true, apply redshift smearing as in https://github.com/echaussidon/mockfactory/blob/341d915bd37c725e10c0b2f490960efc916a56dd/mockfactory/desi/redshift_smearing.py

#         tracer_smearing: str, default='QSO'
#             What king of smearing you want to apply. Use the default filename used in mockfactory/desi/redshift_smearing.py

#         cosmo : Cosmology
#             Cosmology of the input mock, to convert n(z) in ``nz_filename`` to mock units.

#         seed : int, default=145
#             Random seed, for reproductibility during the masking.

#         Returns
#         -------
#         cutsky : CutskyCatalog
#             Catalog with matched n(z) distribution.
#         """
#         from mockfactory import TabulatedRadialMask

#         # Load nz
#         zbin_min, zbin_max, n_z = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
#         zbin_mid = (zbin_min + zbin_max) / 2
#         # Compute comobile volume
#         zedges = np.insert(zbin_max, 0, zbin_min[0])
#         dedges = cosmo.comoving_radial_distance(zedges)
#         volume = dedges[1:]**3 - dedges[:-1]**3
#         mask_radial = TabulatedRadialMask(z=zbin_mid, nbar=n_z / volume, interp_order=2, zrange=(zmin, zmax))

#         if apply_redshift_smearing:
#             from mockfactory.desi import TracerRedshiftSmearing
#             # Note: apply redshift smearing before the n(z) match since n(z) is what we observe (ie) containing the smearing
#             cutsky['Z'] = cutsky['Z'] + TracerRedshiftSmearing(tracer=tracer_smearing).sample(cutsky['Z'], seed=seed + 13)

#         return cutsky[mask_radial(cutsky['Z'], seed=seed)]

#     def is_in_photometric_region(ra, dec, region, rank=0):
#         """DN=NNGC and DS = SNGC"""
#         region = region.upper()
#         assert region in ['N', 'DN', 'DS', 'N+SNGC', 'SNGC', 'SSGC', 'DES']

#         DR9Footprint = None
#         try:
#             from regressis import DR9Footprint
#         except ImportError:
#             if rank == 0: logger.info('Regressis not found, falling back to RA/Dec cuts')

#         if DR9Footprint is None:
#             mask = np.ones_like(ra, dtype='?')
#             if region == 'DES':
#                 raise ValueError('Do not know DES cuts, install regressis')
#             dec_cut = 32.375
#             if region == 'N':
#                 mask &= dec > dec_cut
#             else:  # S
#                 mask &= dec < dec_cut
#             if region in ['DN', 'DS', 'SNGC', 'SSGC']:
#                 mask_ra = (ra > 100 - dec)
#                 mask_ra &= (ra < 280 + dec)
#                 if region in ['DN', 'SNGC']:
#                     mask &= mask_ra
#                 else:  # DS
#                     mask &= dec > -25
#                     mask &= ~mask_ra
#             return np.nan * np.ones(ra.size), mask
#         else:
#             from regressis.utils import build_healpix_map
#             # Precompute the healpix number
#             nside = 256
#             _, pixels = build_healpix_map(nside, ra, dec, return_pix=True)

#             # Load DR9 footprint and create corresponding mask
#             dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=False, mask_around_des=False, cut_desi=False, verbose=(rank == 0))
#             convert_dict = {'N': 'north', 'DN': 'south_mid_ngc', 'N+SNGC': 'ngc', 'SNGC': 'south_mid_ngc', 'DS': 'south_mid_sgc', 'SSGC': 'south_mid_sgc', 'DES': 'des'}
#             return pixels, dr9_footprint(convert_dict[region])[pixels]


#     def apply_photo_desi_footprint(cutsky, region, release, program='dark', npasses=None, rank=0):
#         """
#         Remove part of the cutsky to match as best as possible (precision is healpix map at nside)
#         the DESI release (e.g. y1) footprint and DR9 photometric footprint.
#         """
#         from mockfactory.desi import is_in_desi_footprint

#         # Mask objects outside DESI footprint:
#         is_in_desi = is_in_desi_footprint(cutsky['RA'], cutsky['DEC'], release=release, program=program, npasses=npasses)
#         cutsky['HPX'], is_in_photo = is_in_photometric_region(cutsky['RA'], cutsky['DEC'], region, rank=rank)
#         return cutsky[is_in_desi & is_in_photo]

#         def run(self, hod_params, nthreads)