import os
from pathlib import Path
import yaml
import numpy as np
from abacusnbody.hod import abacus_hod
from cosmoprimo.fiducial import DESI, AbacusSummit
import mockfactory
from astropy.io import fits
from astropy.table import Table
import logging
import warnings
import sys
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


from acm.utils.paths import get_Abacus_dirs
LRG_Abacus_DM = get_Abacus_dirs(tracer='LRG', simtype='box')

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
        DM_DICT: dict = LRG_Abacus_DM):
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
        if sim_type not in ['base', 'small', 'png']:
            raise ValueError('Invalid sim_type. Must be either "base", "small", or "png".')
        self.sim_type = sim_type
        self.boxsize = 2000 if sim_type in ['base', 'png'] else 500
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
        self.cosmo_fid = DESI()
        if self.cosmo_idx in [300, 301, 302, 303]:
            self.cosmo = AbacusSummit(0)
        else:
            self.cosmo = AbacusSummit(self.cosmo_idx)
        self.az = 1 / (1 + self.redshift)
        self.hubble = 100 * self.cosmo.efunc(self.redshift)
        self.a_par = 100 * self.cosmo_fid.efunc(self.redshift) / self.hubble
        self.a_perp = self.cosmo.angular_diameter_distance(self.redshift) / self.cosmo_fid.angular_diameter_distance(self.redshift)
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
        if self.sim_type == 'png':
            return f'Abacus_{self.sim_type}base_c{self.cosmo_idx:03}_ph{self.phase_idx:03}'
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
        tracer_density_mean: list = None,
        seed = None, 
        save_fn: str = None, 
        add_rsd: bool = False,
        add_ap: bool = False,
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
        tracer_density_mean : list, optional
            List containing (min_nbar, max_nbar) for downsampling catalogue to desired density (nbar > max_nbar) or cutting from sample (nbar < min_nbar). Default is None (no thresholds applied).
        seed : int, optional
            Random seed. Default is None.
        save_fn : str, optional
            Filename to save the catalog. Default is None.
        add_rsd : bool, optional
            Whether to add redshift-space distortions to the catalog or not. Default is False.
        add_ap: bool, optional
            Whether to add Alcock-Paczynski distortions to the catalog or not. Default is False.

        Returns
        -------
        dict
            Dictionary containing the HOD catalog. None is returned 
        int
            Value 0 or 1 returned based on whether catalogue is within target density theshold.

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
        # NOTE: compute_ngal not working for HODs with high sigma values
        # ngal_dict = self.ball.compute_ngal(Nthread=nthreads)[0]
        # n_tracers = ngal_dict[tracer]
        # if tracer_density_mean is not None:
            # n_target = tracer_density_mean * self.boxsize ** 3
            # if add_ap: n_target *= self.a_par * self.a_perp**2
            # if (n_target[0] / n_tracers) > 1: 
                # return None, 0
            # else:
                # self.ball.tracers[tracer]['ic'] = min(
                    # 1, n_target[1] / n_tracers
                # )
        hod_dict = self.ball.run_hod(self.ball.tracers, want_rsd=False, Nthread=nthreads, reseed=seed)
        # workaround for compute_ngal issue
        n_tracers = len(hod_dict[tracer]['x'])
        if tracer_density_mean is not None:
            n_target = tracer_density_mean * self.boxsize ** 3
            if add_ap: n_target *= self.a_par * self.a_perp**2
            if (n_target[0] / n_tracers) > 1: 
                return hod_dict, 0
            elif (n_target[1] / n_tracers) < 1:
                subsample = np.random.choice(range(n_tracers), size=int(n_target[1]), replace=False)
            else:
                subsample = None

        hod_dict = self.postprocess_catalog(hod_dict, tracer, subsample, add_rsd, add_ap)
        if save_fn is not None:
            self.save_catalog(hod_dict, save_fn)
        return hod_dict, 1

    def postprocess_catalog(
        self, 
        hod_dict: dict, 
        tracer: str = 'LRG', 
        subsample: list = None,
        add_rsd: bool = False,
        add_ap: bool = False,
        ):
        """
        Add distortion effects and format the HOD catalog.

        Parameters
        ----------
        hod_dict : dict
            Dictionary containing the HOD catalog.
        tracer : str, optional
            Tracer type. Default is 'LRG'.
        subsample: list, optional
            List of indices used to subsample the catalogue.
        add_rsd : bool, optional
            Whether to add redshift-space distortions to the catalog or not. Default is False.
        add_ap : bool, optional
            Whether to add Alcock-Paczynski distortions to the catalog or not. Default is False.
        """
        Ncent = hod_dict[tracer]['Ncent']
        hod_dict[tracer].pop('Ncent', None)
        is_central = np.zeros(len(hod_dict[tracer]['x']))
        is_central[:Ncent] += 1
        hod_dict[tracer]['is_cent'] = is_central

        # workaround for compute_ngal issue
        if subsample is None:
            hod_dict[tracer] = {k.upper():v  for k, v in hod_dict[tracer].items()}
        else:
            hod_dict[tracer] = {k.upper():v[subsample]  for k, v in hod_dict[tracer].items()}

        # add distortions to catalogue
        if add_rsd:
            hod_dict = self._add_rsd(hod_dict, tracer)
        if add_ap:
            hod_dict = self._add_ap(hod_dict, tracer)

        return hod_dict

    def save_catalog(self, hod_dict: dict, save_fn: str):
        """
        Save the HOD catalog to a FITS file.

        Parameters
        ----------
        hod_dict : dict
            Dictionary containing the HOD catalog.
        save_fn : str
            Filename to save the catalog.
        """
        table = Table(hod_dict['LRG'])
        header = fits.Header({'gal_type': 'LRG', **self.ball.tracers['LRG']})
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
        offset = self.boxsize / 2

        x = data['X'] + offset
        y = data['Y'] + offset
        z = data['Z'] + offset

        # remove velocities from catalogue
        vx = hod_dict[tracer].pop('VX')
        vy = hod_dict[tracer].pop('VY')
        vz = hod_dict[tracer].pop('VZ')

        x_rsd = (x + vx / (self.hubble * self.az)) % self.boxsize
        y_rsd = (y + vy / (self.hubble * self.az)) % self.boxsize
        z_rsd = (z + vz / (self.hubble * self.az)) % self.boxsize

        hod_dict[tracer]['X_RSD'] = x_rsd - offset
        hod_dict[tracer]['Y_RSD'] = y_rsd - offset
        hod_dict[tracer]['Z_RSD'] = z_rsd - offset

        return hod_dict

    def _add_ap(
        self,
        hod_dict: dict,
        tracer: str = 'LRG',
        )-> dict:
        """
        Add Alcock-Paczynski distortions to the catalog.

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

        # take AP parameters in as argument to prevent recalculation
        hod_dict[tracer]['X'] *= self.a_par
        hod_dict[tracer]['Y'] *= self.a_perp
        hod_dict[tracer]['Y'] *= self.a_perp

        hod_dict[tracer]['X_RSD'] *= self.a_par
        hod_dict[tracer]['Y_RSD'] *= self.a_perp
        hod_dict[tracer]['Y_RSD'] *= self.a_perp

        return hod_dict
