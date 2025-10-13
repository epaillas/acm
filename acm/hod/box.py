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
        self.q_par = 100 * self.cosmo_fid.efunc(self.redshift) / self.hubble
        self.q_perp = self.cosmo.angular_diameter_distance(self.redshift) / self.cosmo_fid.angular_diameter_distance(self.redshift)
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
        process_underdense: bool = True,
        seed = None, 
        save_fn: str = None, 
        add_rsd: bool = False,
        add_ap: bool = False,
        los: str = None,
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
            List containing (min_nbar, max_nbar) for downsampling catalogue to desired density (nbar > max_nbar) or cutting from sample (nbar < min_nbar). If only one value provided, this is taken as the maximum threshold (no minimum threshold applied). Default is None (no thresholds applied).
        process_underdense: bool, optional
            If set to False, does not process (and save) catalogs that are not in tracer_density_mean limits (only used if tracer_density_mean is provided). Defaults to True.
        seed : int, optional
            Random seed. Default is None.
        save_fn : str, optional
            Filename to save the catalog. Default is None.
        add_rsd : bool, optional
            Whether to add redshift-space distortions to the catalog or not. Default is False.
        add_ap: bool, optional
            Whether to add Alcock-Paczynski distortions to the number density or not. Default is False.
        los: str, optional
            Line-of-sight for RSD and AP distortions. If None, distortions along every axis are saved to catalogue.

        Returns
        -------
        dict
            Dictionary containing the HOD catalog. Galaxy positions ('X','Y','Z') are provided along with optional RSD distorted positions ('X_RSD', 'Y_RSD' and 'Z_RSD'). If `add_ap` is True, AP distotions are applied to the real-space and redshift-space positions along chosen `los` and stored as ('X_PAR', 'X_PERP', 'Y_PAR', 'Y_PERP', 'Z_PAR', 'Z_PERP'). If `los` is not None, only the relevant components will be stored.

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
        self.in_density = True  # Flag if mock is within density threshold
        # # NOTE: compute_ngal not working for HODs with high sigma values
        # ngal_dict = self.ball.compute_ngal(Nthread=nthreads)[0]
        # n_gal = ngal_dict[tracer]
        # if tracer_density_mean is not None:
            # n_target = np.array(tracer_density_mean) * self.boxsize ** 3
            # if add_ap: n_target /= self.q_par * self.q_perp**2
            # if (n_target.size > 1) & (n_target.min() / n_gal > 1): 
                # self.logger.info('Catalogue below minimum density threshold')
                # self.in_density = False  Flag that mock is below density threshold
                # return hod_dict
            # else:
                # self.ball.tracers[tracer]['ic'] = min(
                    # 1, n_target[1] / n_gal
                # )
        hod_dict = self.ball.run_hod(self.ball.tracers, want_rsd=False, Nthread=nthreads, reseed=seed)
        # workaround for compute_ngal issue
        n_gal = len(hod_dict[tracer]['x'])
        subsample = None
        if tracer_density_mean is not None:
            n_target = np.array(tracer_density_mean) * self.boxsize ** 3
            if add_ap: n_target /= self.q_par * self.q_perp**2
            if (n_target.size > 1) & (n_target.min() / n_gal > 1): 
                self.logger.info('Catalogue below minimum density threshold')
                self.in_density = False  # Flag that mock is below density threshold
                if not process_underdense:
                    return hod_dict  # Unprocessed catalog, should not be used
            elif (n_target.max() / n_gal) < 1:
                self.logger.info('Downsampling mock')
                subsample = np.random.choice(range(n_gal), size=int(n_target.max()), replace=False)
            else:
                self.logger.info('Mock within density thresholds')

        # Catalogue positions not distorted by AP to allow freedom of applying to any axis at a later stage 
        hod_dict = self.postprocess_catalog(hod_dict, tracer, subsample, add_rsd, add_ap, los=los)
        if save_fn is not None:
            self.save_catalog(save_fn, hod_dict, tracer)
        return hod_dict

    def postprocess_catalog(
        self, 
        hod_dict: dict, 
        tracer: str = 'LRG', 
        subsample: list = None,
        add_rsd: bool = False,
        add_ap: bool = False,
        los: str = None,
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
        los: str, optional
            Line-of-sight for RSD and AP distortions. If None, distortions along every axis are saved to catalogue.

        Returns
        -------
        dict
            Dictionary containing the HOD catalog.
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
            hod_dict = self._add_rsd(hod_dict, tracer, los=los)
        if add_ap:
            hod_dict = self._add_ap(hod_dict, tracer, los=los)

        # remove velocities from catalogue
        hod_dict[tracer].pop('VX', None)
        hod_dict[tracer].pop('VY', None)
        hod_dict[tracer].pop('VZ', None)

        return hod_dict

    def save_catalog(
        self, 
        save_fn: str,
        hod_dict: dict, 
        tracer: str = 'LRG', 
        ):
        """
        Save the HOD catalog to a FITS file.

        Parameters
        ----------
        save_fn : str
            Filename to save the catalog.
        hod_dict : dict
            Dictionary containing the HOD catalog.
        tracer : str, optional
            Tracer type. Default is 'LRG'.
        """
        table = Table(hod_dict[tracer])
        header = fits.Header({'gal_type': tracer, 'q_par': self.q_par, 
                              'q_perp': self.q_perp, **self.ball.tracers[tracer]})
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
        los: str = None,
        )-> dict:
        """
        Add redshift-space distortions to the catalog.
        
        Parameters
        ----------
        hod_dict : dict
            Dictionary containing the HOD catalog.
        tracer : str, optional
            Tracer type. Default is 'LRG'.
        los: str, optional
            Line-of-sight for RSD distortion. If None, distortions along every axis are saved to catalogue.
        
        Returns
        -------
        dict
            Dictionary containing the HOD catalog with redshift-space distortions.
        """
        self.logger.debug('Distorting galaxy positions with RSD effect')

        data = hod_dict[tracer]
        offset = self.boxsize / 2

        axes = ('X', 'Y', 'Z') if los is None else los.upper()
        for ax in axes:
            pos = data[ax] + offset
            vel = hod_dict[tracer].pop(f'V{ax}')
            pos_rsd = (pos + vel / (self.hubble * self.az)) % self.boxsize
            hod_dict[tracer][f'{ax}_RSD'] = pos_rsd - offset

        return hod_dict

    def _add_ap(
        self,
        hod_dict: dict,
        tracer: str = 'LRG',
        los: str = None,
        )-> dict:
        """
        Add Alcock-Paczynski distortions to the catalog.

        Parameters
        ----------
        hod_dict : dict
            Dictionary containing the HOD catalog.
        tracer : str, optional
            Tracer type. Default is 'LRG'.
        los: str, optional
            Line-of-sight for AP distortion. If None, distortions along every axis are saved to catalogue.

        Returns
        -------
        dict
            Dictionary containing the HOD catalog with AP distortions. Position axis will be replaced by `X_PAR` or `X_PERP` (similar for Y and Z axes) depending on `los` chosen. If `los` is None, `X_PAR` and `X_PERP` components will be stored for all axes. Also applies `q_par` distortions on the RSD axis.
        """
        self.logger.debug('Distorting galaxy positions with AP effect')

        for ax in ('X', 'Y', 'Z'):
            pos = hod_dict[tracer].pop(ax)
            if los is None or (ax == los.upper()):
                hod_dict[tracer][f'{ax}_PAR'] = pos / self.q_par
            if los is None or (ax != los.upper()):
                hod_dict[tracer][f'{ax}_PERP'] = pos / self.q_perp
            if f'{ax}_RSD' in hod_dict[tracer]: hod_dict[tracer][f'{ax}_RSD'] /= self.q_par

        return hod_dict
