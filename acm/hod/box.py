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

from acm.data.paths import LRG_Abacus_DM

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
        DM_DICT: dict = LRG_Abacus_DM['box']):
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
            Dictionary containing dark matter information. Default is the LRG_Abacus_DM dictionary for boxes from `acm.data.paths`.
            
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