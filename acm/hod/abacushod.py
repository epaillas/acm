import os
from pathlib import Path
import yaml
import numpy as np
from abacusnbody.hod import abacus_hod
from cosmoprimo.fiducial import AbacusSummit
import mockfactory
from astropy.io import fits
from astropy.table import Table
import logging
import warnings
import sys
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from acm.data.paths import LRG_Abacus_DM as DM_DICT

class BoxHOD:
    """
    BoxHOD is a wrapper around AbacusHOD, a class for handling Halo Occupation Distribution (HOD) modeling 
    using the AbacusSummit simulations.
    
    Note
    ----
    Only the 'LRG' tracers are currently supported.
    Note that BGS is also supported, by using `tracer='LRG'` and BGS characteristics (mean density, redhsift, etc.)
    """
    def __init__(
        self,
        varied_params, 
        config_file: str = None, 
        cosmo_idx: int = 0, 
        phase_idx: int = 0,
        sim_type: str = 'base', 
        redshift: float = 0.5,
        DM_DICT: dict = DM_DICT):
        """
        Initialize the BoxHOD class.
        
        Parameters
        ----------
        varied_params : dict
            Dictionary of parameters that vary.
        config_file : str, optional
            Path to the configuration file. If None, defaults to 'box.yaml' in `acm.hod`.
            See `setup()` for more details. Default is None.
        cosmo_idx : int, optional
            Index of the cosmology to use. Default is 0.
        phase_idx : int, optional
            Index of the phase to use. Default is 0.
        sim_type : str, optional
            Type of simulation. Must be either 'base' or 'small'. Default is 'base'.
        redshift : float, optional
            Redshift value. Default is 0.5.
        DM_DICT : dict, optional
            Dictionary containing dark matter information. Default is the LRG_Abacus_DM dictionary from `acm.data.paths`.
            
        Raises
        ------
        ValueError
            If `sim_type` is not 'base' or 'small'.
        """
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

    def setup(self, config: dict, DM_DICT: dict): # Will override most of the config file !
        """
        Set up the simulation parameters and initialize the AbacusHOD object.
        This method overrides most of the configuration file settings with the provided
        `config` and `DM_DICT` parameters. 
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing simulation parameters and HOD parameters.
        DM_DICT : dict
            Dictionary containing dark matter simulation directories.
        """
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

    def abacus_simdirs(self, DM_DICT: dict):
        """
        Get the simulation and subsample directories from the dark matter dictionary.

        Parameters
        ----------
        DM_DICT : dict
            Dictionary containing dark matter simulation directories.

        Returns
        -------
        tuple
            Tuple containing the simulation and subsample directories.
        """
        sim_dir = DM_DICT[self.sim_type]['sim_dir']
        subsample_dir = DM_DICT[self.sim_type]['subsample_dir']
        return sim_dir, subsample_dir

    def abacus_simname(self):
        """
        Get the simulation name.

        Returns
        -------
        str
            Simulation name, following the Abacus format.
        """
        return f'AbacusSummit_{self.sim_type}_c{self.cosmo_idx:03}_ph{self.phase_idx:03}'

    def check_params(self, params):
        """
        Check if the parameters are valid, i.e. if they are in the list of valid parameters.

        Parameters
        ----------
        params : list
            List of parameters to check.

        Raises
        ------
        ValueError
            If the parameters are invalid.
        """
        params = list(params)
        params = self.param_mapping(params) # re-map custom keys to Abacus keys
        for param in params:
            if param not in self.ball.tracers['LRG'].keys():
                raise ValueError(f'Invalid parameter: {param}. Valid list '
                                 f'of parameters include: {list(self.ball.tracers["LRG"].keys())}')
        self.logger.info(f'Varied parameters: {params}.')
        self.varied_params = params
        default = {key: value for key, value in self.ball.tracers['LRG'].items() if key not in params}
        self.logger.info(f'Default parameters: {default}.')

    def run(
        self, 
        hod_params: dict, 
        nthreads: int = 1, 
        tracer: str = 'LRG', 
        tracer_density_mean: float = None,
        seed = None, 
        save_fn: str = None, 
        add_rsd: bool = False,
        )-> dict:
        """
        Run the HOD model with the given parameters.

        Parameters
        ----------
        hod_params : dict
            Dictionary of HOD parameters.
        nthreads : int, optional
            Number of threads to use. Default is 1.
        tracer : str, optional
            Tracer type. Default is 'LRG'.
        tracer_density_mean : float, optional
            To force the mean density of the tracers. Will downsample the catalog to the desired density if needed.
            Default is None.
        seed : int, optional
            Random seed. Default is None.
        save_fn : str, optional
            Filename to save the catalog. Default is None.
        add_rsd : bool, optional
            Wether to add redshift-space distortions to the catalog or not. Default is False.

        Returns
        -------
        dict
            Dictionary containing the HOD catalog.

        Raises
        ------
        ValueError
            If the tracer is not 'LRG'.
        ValueError
            If the HOD parameters do not match the varied parameters.
        """
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

    def format_catalog(
        self, 
        hod_dict: dict, 
        save_fn: str = False, 
        tracer: str = 'LRG', 
        add_rsd: bool = False):
        """
        Format the HOD catalog and save it to a FITS file if requested.

        Parameters
        ----------
        hod_dict : dict
            Dictionary containing the HOD catalog.
        save_fn : str, optional
            Filename to save the catalog. Default is False.
        tracer : str, optional
            Tracer type. Default is 'LRG'.
        add_rsd : bool, optional
            Wether to add redshift-space distortions to the catalog or not. Default is False.
        """
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

    def _add_rsd(
        self, 
        hod_dict: dict, 
        tracer: str = 'LRG',
        )-> dict:
        """
        Add redshift-space distortions to the catalog.
        
        Parameters
        ----------
        hod_dict : dict
            Dictionary containing the HOD catalog.
        tracer : str, optional
            Tracer type. Default is 'LRG'.
        
        Returns
        -------
        dict
            Dictionary containing the HOD catalog with redshift-space distortions.
        """
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
        seed=None, save_fn=None, full_sky=False, alpha_rand=1, apply_radial_mask=False):
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
        self.format_catalog(hod_dict, save_fn, tracer, full_sky, apply_radial_mask)
        if make_randoms:
            zmin = hod_dict[tracer]['Z'].min()
            zmax = hod_dict[tracer]['Z'].max()
            nbar = self.get_data_nbar(hod_dict, tracer, full_sky)
            randoms_dict = self._make_randoms(nbar=nbar, zmin=zmin, zmax=zmax, apply_radial_mask=apply_radial_mask,
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

    def format_catalog(self, hod_dict, save_fn=False, tracer='LRG', full_sky=False, apply_radial_mask=False):
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
        if apply_radial_mask:
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

    def _make_randoms(self, nbar, zmin, zmax, alpha=1, full_sky=False, apply_radial_mask=False, tracer='LRG'):
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
        if apply_radial_mask:
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


class CutskyHOD:
    """
    Patch together cubic boxes to form a pseudo-lightcone.
    """
    def __init__(self, varied_params, config_file=None, cosmo_idx=0, phase_idx=0,
        zranges=[[0.41, 0.6]], snapshots=[0.5]):
        self.logger = logging.getLogger('CutskyHOD')
        self.varied_params = varied_params
        self.cosmo_idx = cosmo_idx
        self.phase_idx = phase_idx
        self.sim_type = 'base'
        self.zranges = zranges
        self.snapshots = snapshots
        self.boxsize = 2000
        self.boxcenter = 0
        self.setup()

    def setup(self):
        self.balls = []
        for zsnap in self.snapshots:
            # self.logger.info(f'Processing {self.abacus_simname()} at z = {self.redshift}')
            ball = BoxHOD(varied_params=self.varied_params, sim_type=self.sim_type,
                          redshift=zsnap, cosmo_idx=self.cosmo_idx, phase_idx=self.phase_idx)
            self.balls += [ball]
        self.cosmo = AbacusSummit(self.cosmo_idx)

    def run(self, hod_params, nthreads=1, seed=0, generate_randoms=False, alpha_randoms=5,
            randoms_seed=42):
        data_cutsky = {}
        randoms_cutsky = {}
        for ball, zsnap, zranges in zip(self.balls, self.snapshots, self.zranges):
            hod_dict_i = ball.run(hod_params, seed=seed, nthreads=nthreads)['LRG']
            pos = np.c_[hod_dict_i['X'], hod_dict_i['Y'], hod_dict_i['Z']]
            vel = np.c_[hod_dict_i['VX'], hod_dict_i['VY'], hod_dict_i['VZ']]
            print('Generating data')
            data = mockfactory.BoxCatalog(
                data={'Position': pos, 'Velocity': vel},
                position='Position',
                velocity='Velocity',
                boxsize=self.boxsize,
                boxcenter=self.boxcenter,
            )
            data.recenter()
            data_nbar = len(data) / (self.boxsize**3)
            tmp_data_cutsky = self._to_cutsky(data, *zranges, zsnap, 
                                          apply_rsd=True,
                                          apply_radial_mask=True,
                                          radial_mask_norm=1/data_nbar,
                                          apply_footprint_mask=True)
            if generate_randoms:
                print('Generating randoms.')
                nbar_randoms = data_nbar * alpha_randoms
                randoms =  mockfactory.RandomBoxCatalog(
                    nbar=nbar_randoms, boxsize=self.boxsize,
                    boxcenter=self.boxcenter, seed=randoms_seed,
                )
                tmp_randoms_cutsky = self._to_cutsky(randoms, *zranges, zsnap,
                                                 apply_rsd=False,
                                                 apply_radial_mask=True,
                                                 radial_mask_norm=1/data_nbar,
                                                 apply_footprint_mask=True)
            # concatenate to previous shell, if any
            data_keys = ['RA', 'DEC', 'Z', 'RSDPosition', 'Distance', 'Position']
            randoms_keys = ['RA', 'DEC', 'Z', 'Position', 'Distance']
            if data_cutsky:
                for key in data_keys:
                    data_cutsky[key] = np.concatenate([data_cutsky[key], tmp_data_cutsky[key]])
                if generate_randoms:
                    for key in randoms_keys:
                        randoms_cutsky[key] = np.concatenate([randoms_cutsky[key], tmp_randoms_cutsky[key]])
            else:
                for key in data_keys:
                    data_cutsky[key] = tmp_data_cutsky[key]
                if generate_randoms:
                    for key in randoms_keys:
                        randoms_cutsky[key] = tmp_randoms_cutsky[key]
        if generate_randoms:
            return data_cutsky, randoms_cutsky
        return data_cutsky

    def _to_cutsky(self, catalog, zmin, zmax, zsnap, apply_rsd=False, apply_radial_mask=False,
        apply_footprint_mask=False, radial_mask_norm=None):
        nbar = len(catalog) / (self.boxsize**3)
        dist = self.cosmo.comoving_radial_distance((zmin + zmax) / 2)
        cutsky = self._apply_geometric_cuts(catalog, self.boxsize, dist)
        if apply_rsd: 
            cutsky = self._apply_rsd(cutsky, zsnap)
        cutsky = self._get_sky_positions(cutsky, apply_rsd)
        if apply_radial_mask:
            cutsky = self._apply_radial_mask(cutsky, zmin=zmin, zmax=zmax,
                                             norm=radial_mask_norm)
        if apply_footprint_mask:
            cutsky = self._apply_footprint_mask(cutsky)
        return cutsky

    def _apply_geometric_cuts(self, catalog, boxsize, dist):
        print('Applying geometric cuts.')
        # largest (RA, Dec) range we can achieve for a maximum distance of dist + boxsize / 2.
        drange, rarange, decrange = mockfactory.box_to_cutsky(boxsize=boxsize, dmax=dist + boxsize / 2.)
        rarange = np.array(rarange) + 192
        decrange = np.array(decrange) + 35
        # returned isometry corresponds to a displacement of the box along the x-axis to match drange, then a rotation to match rarange and decrange
        isometry, mask_radial, mask_angular = catalog.isometry_for_cutsky(drange=drange, rarange=rarange, decrange=decrange)
        return catalog.cutsky_from_isometry(isometry, rdd=None)

    def _apply_rsd(self, catalog, zsnap):
        print('Applying RSD.')
        a = 1 / (1 + zsnap) # scale factor
        H = 100.0 * self.cosmo.efunc(zsnap)  # Hubble parameter in km/s/Mpc
        rsd_factor = 1 / (a * H)  # multiply velocities by this factor to convert to Mpc/h
        catalog['RSDPosition'] = catalog.rsd_position(f=rsd_factor)
        return catalog

    def _get_sky_positions(self, catalog, apply_rsd=False):
        print('Converting to sky positions.')
        distance_to_redshift = mockfactory.DistanceToRedshift(distance=self.cosmo.comoving_radial_distance)
        pos = 'RSDPosition' if apply_rsd else 'Position'
        catalog['Distance'], catalog['RA'], catalog['DEC'] = mockfactory.cartesian_to_sky(catalog[pos])
        catalog['Z'] = distance_to_redshift(catalog['Distance'])
        return catalog

    def _apply_radial_mask(self, catalog, zmin=0., zmax=6., seed=42, norm=None):
        print('Applying radial mask.')
        from mockfactory import TabulatedRadialMask
        nz_filename = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/LRG_NGC_nz.txt'
        zbin_min, zbin_max, n_z = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
        zbin_mid = (zbin_min + zbin_max) / 2
        zedges = np.insert(zbin_max, 0, zbin_min[0])
        dedges = self.cosmo.comoving_radial_distance(zedges)
        volume = dedges[1:]**3 - dedges[:-1]**3
        mask_radial = mockfactory.TabulatedRadialMask(z=zbin_mid, nbar=n_z, interp_order=2,
                                                      zrange=(zmin, zmax), norm=norm)
        return catalog[mask_radial(catalog['Z'], seed=seed)]

    def _apply_footprint_mask(self, catalog):
        print('Applying footprint mask.')
        from mockfactory.desi import is_in_desi_footprint
        is_in_desi = is_in_desi_footprint(catalog['RA'], catalog['DEC'], release='y1', program='dark', npasses=None)
        return catalog[is_in_desi]