import os
import yaml
import logging
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.table import Table
import logging
import warnings
import sys
from abacusnbody.hod import abacus_hod
from cosmoprimo.fiducial import DESI, AbacusSummit
from acm.utils.paths import lookup_registry_path



class BoxHOD:
    """
    BoxHOD is a wrapper around AbacusHOD, a class for handling Halo Occupation Distribution (HOD) modeling 
    using the AbacusSummit simulations.
    """
    
    logger = logging.getLogger('BoxHOD') # Set up logger for the class as a class attribute
    
    def __init__(
        self,
        varied_params: list[str],
        tracer: str = 'LRG',
        config_file: str | None = None,
        cosmo_idx: int = 0,
        phase_idx: int = 0,
        sim_type: str = 'base',
        redshift: float = 0.5,
        DM_DICT: dict = None,
        DM_DICT_simtype: str = None,
        sim_geometry: str = None,
    ):
        """
        Initialize the BoxHOD class.
        
        Parameters
        ----------
        tracer : str, optional
            Tracer type. Default is 'LRG'. Each BoxHOD object uses a single tracer, 
            since the varied_params list is itself fixed for each BoxHOD object,
            while varied_params may differ for different tracers.
        varied_params : list[str]
            List of parameters that vary.
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
            Dictionary containing dark matter information. Defaults to None, which 
            together with the user-specified tracer maps to a value in utils.paths.
        DM_DICT_simtype : str, optional
            The simtype paramter used by get_Abacus_dirs, either 'box' or 
            'lightcone'. Defaults to 'box' if None
        sim_geometry : str, optional
            The simtype paramter used for choices related to the mock geometry. 
            Either 'box', 'cutsky', or 'lightcone'. Defaults to 'box' if None
            
        Raises
        ------
        ValueError
            If `sim_type` is not 'base' or 'small'.
        """
        self.cosmo_idx = cosmo_idx
        self.phase_idx = phase_idx
        if sim_geometry is None:
            sim_geometry = 'box'
        self.sim_geometry = sim_geometry
        if sim_type not in ['base', 'small', 'png']:
            raise ValueError('Invalid sim_type. Must be either "base", "small", or "png".')
        self.sim_type = sim_type
        self.boxsize = 2000 if sim_type in ['base', 'png'] else 500
        self.redshift = redshift
        if config_file is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
            # TODO: remove tracer condition and make all tracer config files use the naming convention 'box_{tracer}.yaml' for consistency ?
            if tracer == 'LRG':
                config_file = Path(config_dir) /  'box.yaml'
            else:
                box_yaml_file = 'box_' + tracer + '.yaml'
                config_file = Path(config_dir) / box_yaml_file
        config = yaml.safe_load(open(config_file))
        if DM_DICT is None:
            if DM_DICT_simtype is None:
                DM_DICT_simtype = 'box'
            DM_DICT = lookup_registry_path('Abacus.yaml', tracer, DM_DICT_simtype)
        self.setup(config, DM_DICT)
        # AbacusHOD doesn't work with BGS, so after loading the BGS subsample files,
        # we use tracer = LRG for subsequent steps
        if tracer == 'BGS':
            tracer = 'LRG'
        self.tracer = tracer
        self.check_params(varied_params)

    def setup(self, config: dict, DM_DICT: dict) -> None:
        """
        Set up the simulation parameters and initialize the AbacusHOD object.
        This method overrides most of the configuration file settings with the provided
        `config` and `DM_DICT` parameters. 
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing simulation parameters and HOD parameters.
        DM_DICT : dict
            Dictionary containing dark matter simulation directories. Expected structure:
            {
                'base': {'sim_dir': str, 'subsample_dir': str},
                'small': {'sim_dir': str, 'subsample_dir': str},
                'png': {'sim_dir': str, 'subsample_dir': str}
            }
        """
        sim_params = config['sim_params']
        sim_dir, subsample_dir = self.abacus_simdirs(DM_DICT) 
        sim_params['sim_dir'] = sim_dir
        sim_params['subsample_dir'] = subsample_dir
        sim_params['sim_name'] = self.abacus_simname()
        sim_params['z_mock'] = self.redshift
        HOD_params = config['HOD_params']
        self.logger.info(f'Initializing AbacusHOD with parameters {sim_params}.')
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

    def abacus_simdirs(self, DM_DICT: dict) -> tuple:
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

    def abacus_simname(self) -> str:
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

    def check_params(self, params: list[str]) -> None:
        """
        Check if the parameters are valid, i.e. if they are in the list of valid parameters.

        Parameters
        ----------
        params : list[str]
            List of parameters to check.

        Raises
        ------
        ValueError
            If the parameters are invalid.
        """
        params = list(params)
        params = self.param_mapping(params) # re-map custom keys to Abacus keys
        for param in params:
            if param not in self.ball.tracers[self.tracer].keys():
                raise ValueError(f'Invalid parameter: {param}. Valid list '
                                 f'of parameters include: {list(self.ball.tracers[self.tracer].keys())}')
        self.logger.info(f'Varied parameters: {params}.')
        self.varied_params = params
        default = {key: value for key, value in self.ball.tracers[self.tracer].items() if key not in params}
        self.logger.info(f'Default parameters: {default}.')

    def check_catalogue(self, hod_dict: dict, n_target: float, rtol: float = 0.01) -> None:
        """
        Check if catalogue number density and satellite fractions match expectations, i.e. halo/particle catalogue subsampling is sufficient for given parameter values.

        Parameters
        ----------
        hod_dict : dict
            Dictionary containing the HOD catalog.

        n_target: float
            Target number density for HOD catalogue downsampling.

        rtol: float, default=0.01
            Relative tolerance of true and expected values for number density and satellite fraction.

        Raises
        ------
        logger.warning
            If the values are outside of the tolerance.
        """

        # ensure number density matches expectation of HMF
        N_gal_mock = len(hod_dict[self.tracer]['x'])
        n_gal_mock = N_gal_mock / self.boxsize**3
        if self.add_ap: n_gal_mock *= self.q_par * self.q_perp**2
        n_gal_diff = n_gal_mock / min(n_target.max(), self.n_gal) - 1
        if abs(n_gal_diff) > rtol:
            self.logger.warning(f'Number density of mock does not match expectation ({n_gal_diff*100:.0f}% offset). Adjust the halo catalogue subsampling!')

        # ensure satellite fraction matches expectation of HMF
        f_sat_mock = 1 - hod_dict[self.tracer]['Ncent'] / N_gal_mock
        f_sat_diff = f_sat_mock / self.f_sat - 1
        if abs(f_sat_diff) > rtol:
            self.logger.warning(f'Satellite fraction of mock does not match expectation ({f_sat_diff*100:.0f}% offset). Adjust the particle catalogue subsampling!')

    def run(
        self,
        hod_params: dict,
        nthreads: int = 1,
        tracer_density: list[float] | None = None,
        process_underdense: bool = True,
        seed: int | None = None,
        save_fn: str | Path | None = None,
        add_ap: bool = False,
        use_logsigma: bool = True,
        save_distortions: bool = False,
    ) -> dict:
        """
        Run the HOD model with the given parameters.

        Parameters
        ----------
        hod_params : dict
            Dictionary of HOD parameters.
        nthreads : int, optional
            Number of threads to use. Default is 1.
        tracer_density : list[float], optional
            List containing (min_nbar, max_nbar) for downsampling catalogue to desired density (nbar > max_nbar) or cutting from sample (nbar < min_nbar). If only one value provided, this is taken as the maximum threshold (no minimum threshold applied). Default is None (no thresholds applied).
        process_underdense: bool, optional
            If set to False, does not process (and save) catalogs that are not in tracer_density limits (only used if tracer_density is provided). Defaults to True.
        seed : int, optional
            Random seed. Default is None.
        save_fn : str | Path, optional
            Filename to save the catalog. Creates parent tree if it does not exist. Default is None.
        add_ap: bool, optional
            Whether to take Alcock-Paczynski distortions into account when computing the number density. 
            To use if you plan to apply AP distortions to the catalog later on. 
            Default is False.
        use_logsigma: bool, optional
            Whether to use the logarithm of sigma as the parameter for the HOD instead of sigma itself. 
            This is useful for sampling purposes, since sigma can vary over several orders of magnitude. 
            Default is True.
        save_distortions: bool, default=False
            Save positions distorted by RSD and AP effects to catalog (only used if save_fn is provided). 
            AP distortion will not be saved if add_ap is False.

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
        tracer = self.tracer
        self.add_ap = add_ap # flag to indicate if AP distortions were applied to number density
        if seed == 0: seed = None
        #if tracer not in ['LRG']:
        #    raise ValueError('Only LRGs are currently supported.')
        hod_params = self.param_mapping(hod_params)
        if set(hod_params.keys()) != set(self.varied_params):
            raise ValueError('Invalid HOD parameters. Must match the varied parameters.')
        for key in hod_params.keys():
            if key == 'sigma' and use_logsigma:
                self.ball.tracers[tracer][key] = 10**hod_params[key]
            else:
                self.ball.tracers[tracer][key] = hod_params[key]
        self.ball.tracers[tracer]['ic'] = 1
        self.in_density = True  # Flag if mock is within density threshold
        # set want_nfw (unique for ELG cutsky)
        if tracer == 'ELG' and self.sim_geometry == 'cutsky':
            want_nfw = True
        else:
            want_nfw = False

        N_gal_dict, f_sat_dict = self.ball.compute_ngal(Nthread=nthreads)
        self.f_sat = f_sat_dict[tracer]
        self.n_gal = N_gal_dict[tracer] / self.boxsize**3
        if self.add_ap: self.n_gal *= self.q_par * self.q_perp**2

        if tracer_density is not None:
            n_target = np.array(tracer_density)
            if (n_target.size > 1) & (n_target.min() / self.n_gal > 1): 
                self.logger.info(f'Catalogue below minimum density threshold (n={self.n_gal:.5f})')
                self.in_density = False  # Flag that mock is below density threshold
                if not process_underdense:
                    return None
            else:
                self.logger.info(f'Catalogue above minimum density threshold (n={self.n_gal:.5f})')
                self.ball.tracers[tracer]['ic'] = min(1, n_target.max() / self.n_gal)

        hod_dict = self.ball.run_hod(self.ball.tracers, want_rsd=False, Nthread=nthreads, reseed=seed, want_nfw=want_nfw)

        self.check_catalogue(hod_dict, n_target.max())

        # Catalogue positions not distorted by AP to allow freedom of applying to any axis at a later stage 
        hod_dict = self.postprocess_catalog(hod_dict, subsample)
        if save_fn is not None:
            self.save_catalog(save_fn, hod_dict, save_distortions=save_distortions)
        return hod_dict

    def postprocess_catalog(
        self,
        hod_dict: dict,
    ) -> dict:
        """
        Add distortion effects and format the HOD catalog.

        Parameters
        ----------
        hod_dict : dict
            Dictionary containing the HOD catalog.

        Returns
        -------
        dict
            Dictionary containing the HOD catalog.
        """
        tracer = self.tracer
        Ncent = hod_dict[tracer]['Ncent']
        hod_dict[tracer].pop('Ncent', None)
        is_central = np.zeros(len(hod_dict[tracer]['x']))
        is_central[:Ncent] += 1
        hod_dict[tracer]['is_cent'] = is_central

        hod_dict[tracer] = {k.upper():v  for k, v in hod_dict[tracer].items()}

        return hod_dict

    def save_catalog(
        self,
        save_fn: str | Path,
        hod_dict: dict,
        save_distortions: bool = False,
    ) -> None:
        """
        Save the HOD catalog to a FITS file.

        Parameters
        ----------
        save_fn : str | Path
            Filename to save the catalog. If parent tree directories do not exist, they will be created.
        hod_dict : dict
            Dictionary containing the HOD catalog.
        save_distortions: bool, default=False
            Save positions distorted by RSD and AP effects to catalog (only used if save_fn is provided). 
            AP distortion will not be saved if add_ap is False.
        """
        tracer = self.tracer
        # Ensure parent directories exist
        save_fn = Path(save_fn)
        save_fn.parent.mkdir(parents=True, exist_ok=True)
        
        catalog = hod_dict[tracer].copy()
        if save_distortions:
            axes = ['X', 'Y', 'Z']
            append = '_PERP' if self.add_ap else ''
            for (i, los) in enumerate(axes):
                label = axes.copy()
                label[i] += '_RSD'
                label[i-1] += append
                label[i-2] += append

                positions = self.get_positions(
                                    catalog,
                                    los=los,
                                    add_rsd=True,
                                    hubble=self.hubble,
                                    az=self.az,
                                    boxsize=self.boxsize,
                                    add_ap=self.add_ap,
                                    q_par=self.q_par,
                                    q_perp=self.q_perp,
                                )
                catalog[label[0]], catalog[label[1]], catalog[label[2]] = positions.T

        table = Table(catalog)
        header = fits.Header({
            'gal_type': tracer, 
            'n_gal': self.n_gal,
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

    def param_mapping(self, hod_params: dict | list[str]) -> dict | list[str]:
        """
        Map custom HOD parameters to Abacus HOD parameters. 

        Parameters
        ----------
        hod_params : dict | list[str]
            Dictionary or list of HOD parameters.

        Returns
        -------
        dict | list[str]
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
    def get_boxsize(
        cls,
        boxsize: float | list[float],
        add_ap: bool = False,
        los: str | None = None,
        q_par: float | None = None,
        q_perp: float | None = None
    ) -> float | np.ndarray:
        """
        Get the box size, taking into account Alcock-Paczynski distortions if specified.

        Parameters
        ----------
        boxsize : float | list[float]
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
        float | np.ndarray
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
        tracer: str | None = None,
        los: str | None = None,
        add_rsd: bool = False,
        hubble: float | None = None,
        az: float | None = None,
        boxsize: float | None = None,
        add_ap: bool = False,
        q_par: float | None = None,
        q_perp: float | None = None,
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
        # Avoid modifying the original dictionary
        tracer_dict = hod_dict[tracer].copy() if tracer is not None else hod_dict.copy()
        
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
            Will overwrite the input `tracer_dict` in place, use a copy if needed !
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
            Will overwrite the input `tracer_dict` in place, use a copy if needed !

        """
        for ax in ('X', 'Y', 'Z'):
            pos = tracer_dict[ax]
            if ax == los.upper():
                tracer_dict[ax] = pos / q_par
            else: 
                tracer_dict[ax] = pos / q_perp
        return tracer_dict
