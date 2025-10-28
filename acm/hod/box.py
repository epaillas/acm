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
    Note that BGS is also supported, by using `tracer='LRG'` and BGS characteristics (mean density, redshift, etc.)
    """
    
    logger = logging.getLogger('AbacusHOD') # Set up logger for the class as a class attribute
    
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
        tracer_density: list = None,
        process_underdense: bool = True,
        seed = None, 
        save_fn: str|Path = None, 
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
        tracer_density : list, optional
            List containing (min_nbar, max_nbar) for downsampling catalogue to desired density (nbar > max_nbar) or cutting from sample (nbar < min_nbar). If only one value provided, this is taken as the maximum threshold (no minimum threshold applied). Default is None (no thresholds applied).
        process_underdense: bool, optional
            If set to False, does not process (and save) catalogs that are not in tracer_density limits (only used if tracer_density is provided). Defaults to True.
        seed : int, optional
            Random seed. Default is None.
        save_fn : str|Path, optional
            Filename to save the catalog. Creates parent tree if it does not exist. Default is None.
        add_ap: bool, optional
            Whether to take Alcock-Paczynski distortions into account when computing the number density. 
            To use if you plan to apply AP distortions to the catalog later on. 
            Default is False.

        Returns
        -------
        dict
            Dictionary containing the HOD catalog. Galaxy positions ('X','Y','Z') and velocities ('VX','VY','VZ') are in real-space.
            To get positions with RSD and/or AP distortions, use the `get_positions` class method.

        Raises
        ------
        ValueError
            If the tracer is not 'LRG'.
        ValueError
            If the HOD parameters do not match the varied parameters.
        """
        self.add_ap = add_ap # flag to indicate if AP distortions were applied to number density
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
        hod_dict = self.ball.run_hod(self.ball.tracers, want_rsd=False, Nthread=nthreads, reseed=seed)
        # workaround for compute_ngal issue with high sigma values
        n_gal = len(hod_dict[tracer]['x'])
        subsample = None
        if tracer_density is not None:
            n_target = np.array(tracer_density) * self.boxsize ** 3
            if self.add_ap: n_target /= self.q_par * self.q_perp**2
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
        hod_dict = self.postprocess_catalog(hod_dict, tracer, subsample)
        if save_fn is not None:
            self.save_catalog(save_fn, hod_dict, tracer)
        return hod_dict

    def postprocess_catalog(
        self, 
        hod_dict: dict, 
        tracer: str = 'LRG', 
        subsample: list = None,
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

        return hod_dict

    def save_catalog(
        self, 
        save_fn: str|Path,
        hod_dict: dict, 
        tracer: str = 'LRG', 
        ):
        """
        Save the HOD catalog to a FITS file.

        Parameters
        ----------
        save_fn : str|Path
            Filename to save the catalog. If parent tree directories do not exist, they will be created.
        hod_dict : dict
            Dictionary containing the HOD catalog.
        tracer : str, optional
            Tracer type. Default is 'LRG'.
        """
        # Ensure parent directories exist
        save_fn = Path(save_fn)
        save_fn.parent.mkdir(parents=True, exist_ok=True)
        
        table = Table(hod_dict[tracer])
        header = fits.Header({
            'gal_type': tracer, 
            'hubble': self.hubble, 
            'az': self.az,
            'boxsize': self.boxsize,
            'q_par': self.q_par, 
            'q_perp': self.q_perp, 
            **self.ball.tracers[tracer],
        })
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
    
    @classmethod
    def get_boxsize(cls, boxsize: float|list, add_ap: bool = False, los: str = None, q_par: float = None, q_perp: float = None) -> float|list:
        """
        Get the box size, taking into account Alcock-Paczynski distortions if specified.

        Parameters
        ----------
        boxsize : float|list
            Original box size (as a float or a list of three floats for each axis).
        add_ap : bool, optional
            Whether to add Alcock-Paczynski distortions to the box size or not. Default is False.
        los : str, optional
            Line-of-sight for AP distortions. If None, no distortions are applied.
        q_par : float, optional
            Parallel AP distortion factor. Required if `los` is not None.
        q_perp : float, optional
            Perpendicular AP distortion factor. Required if `los` is not None.

        Returns
        -------
        float or np.ndarray
            Box size after applying AP distortions, or original box size if no distortions are applied.
        """
        if not add_ap:
            return boxsize
        elif any(v is None for v in [los, q_par, q_perp]):
            raise ValueError('los, q_par and q_perp must be provided when add_ap is True.')
        
        if isinstance(boxsize, (float, int)): 
            boxsizes = [boxsize] * 3
        else:
            if len(boxsize) != 3: # Sanity check
                raise ValueError('boxsize must be a float or a list of three floats.')
            boxsizes = boxsize

        for i, ax in enumerate(('X', 'Y', 'Z')):
            if ax == los.upper():
                boxsizes[i] = boxsizes[i] / q_par
            else:
                boxsizes[i] = boxsizes[i] / q_perp
        return np.array(boxsizes)

    @classmethod
    def get_positions(
        cls,
        hod_dict: dict, 
        tracer: str = None,
        los: str = None,
        add_rsd: bool = False,
        hubble: float = None,
        az: float = None,
        boxsize: float = None, 
        add_ap: bool = False, 
        q_par: float = None, 
        q_perp: float = None,
    ) -> np.ndarray:
        """
        Get the galaxy positions from the HOD catalog.

        Parameters
        ----------
        hod_dict : dict
            Dictionary containing the tracer positions.
        tracer : str, optional
            Tracer type to read from `hod_dict`. If None, uses the top-level keys of `hod_dict`. Default is None.
        los: str, optional
            Line-of-sight for RSD and AP distortions. If None, no distortions are applied.
        add_rsd : bool, optional
            Whether to add redshift-space distortions to the catalog or not. Default is False.
        hubble : float, optional
            Hubble parameter at the redshift of the catalog. Required if `add_rsd` is True.
        az : float, optional
            Scale factor at the redshift of the catalog. Required if `add_rsd` is True.
        boxsize : float, optional
            Box size of the simulation. Required if `add_rsd` is True.
        add_ap: bool, optional
            Whether to add Alcock-Paczynski distortions to the number density or not. Default is False.
        q_par : float, optional
            Parallel AP distortion factor. Required if `add_ap` is True.
        q_perp : float, optional
            Perpendicular AP distortion factor. Required if `add_ap` is True.

        Returns
        -------
        np.ndarray
            Array of galaxy positions with shape (N_gal, 3).
        """
        tracer_dict = hod_dict[tracer] if tracer is not None else hod_dict
        
        # Apply RSD before AP distortions
        if add_rsd:
            if any(v is None for v in [hubble, az, boxsize, los]):  # Check we have everything we need to add RSD
                raise ValueError('hubble, az, boxsize and los must be provided to add RSD distortions.')
            cls.logger.debug('Applying RSD distortions to positions.')
            tracer_dict = cls._add_rsd(tracer_dict, hubble=hubble, az=az, boxsize=boxsize, los=los)

        if add_ap:
            if any(v is None for v in [q_par, q_perp, los]):  # Check we have everything we need to add AP
                raise ValueError('q_par, q_perp and los must be provided to add AP distortions.')
            cls.logger.debug('Applying AP distortions to positions.')
            tracer_dict = cls._add_ap(tracer_dict, q_par=q_par, q_perp=q_perp, los=los)
            
        positions = np.column_stack([tracer_dict[key] for key in ['X', 'Y', 'Z']])
        cls.logger.debug(f'Obtained positions array of shape {positions.shape}.')
        return positions

    @staticmethod
    def _add_rsd( 
        tracer_dict: dict, 
        hubble: float,
        az: float,
        boxsize: float,
        los: str,
    )-> dict:
        """
        Add redshift-space distortions to the catalog.
        
        Parameters
        ----------
        tracer_dict : dict
            Dictionary containing the tracer positions in `X`, `Y`, `Z`.
        hubble : float
            Hubble parameter at the redshift of the catalog.
        az : float
            Scale factor at the redshift of the catalog.
        boxsize : float
            Box size of the simulation.
        los: str
            Line-of-sight for RSD distortion.
        
        Returns
        -------
        dict
            Dictionary containing the HOD catalog with redshift-space distortions to the specified axis.
        """
        ax = los.upper()
        offset = boxsize / 2 
        pos = tracer_dict[ax] + offset
        vel = tracer_dict[f'V{ax}']
        pos_rsd = (pos + vel / (hubble * az)) % boxsize
        tracer_dict[ax] = pos_rsd - offset # Overwrite real-space positions with RSD positions

        return tracer_dict

    @staticmethod
    def _add_ap(
        tracer_dict: dict,
        q_par: float,
        q_perp: float,
        los: str,
        )-> dict:
        """
        Add Alcock-Paczynski distortions to the catalog.

        Parameters
        ----------
        tracer_dict : dict
            Dictionary containing the tracer positions in `X`, `Y`, `Z`.
        q_par : float
            Parallel AP distortion factor.
        q_perp : float
            Perpendicular AP distortion factor.
        los: str
            Line-of-sight for AP distortion. 

        Returns
        -------
        dict
            Dictionary containing the HOD catalog with Alcock-Paczynski distortions applied to the specified axis.
        """
        for ax in ('X', 'Y', 'Z'):
            pos = tracer_dict[ax]
            if ax == los.upper():
                tracer_dict[ax] = pos / q_par
            else: 
                tracer_dict[ax] = pos / q_perp

        return tracer_dict
