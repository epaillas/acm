import logging
import os
from pathlib import Path

import yaml
from abacusnbody.hod.abacus_hod import AbacusHOD
from cosmoprimo import Cosmology
from cosmoprimo.fiducial import DESI, AbacusSummit

ABACUS_MAP = {
    "logM1": ["log_1"],
    "Acent": ["A_cen"],
    "Asat": ["A_sat"],
    "Bcent": ["B_cen"],
    "Bsat": ["B_sat"],
}

SIM_PARAMS_MAP = {"z_mock": ["redshift"]}


def map_params(
    params: dict | list[str], mapping: dict[str, list[str]] = ABACUS_MAP
) -> dict | list[str]:
    """
    Map custom parameters names to fixed parameters.

    Parameters
    ----------
    params : dict | list[str]
        Dictionary or list of custom parameters.
    mapping : dict[str, list[str]]
        Mapping from custom parameter names to fixed parameter names.
        Keys are fixed parameter names, values are lists of custom parameter names that map to the fixed parameter name.

    Returns
    -------
    dict | list[str]
        Dictionary or list of fixed parameters. Use the same type as the input params.

    Raises
    ------
    ValueError
        If the type of params is not dict or list.
    """
    is_dict = type(params) is dict
    if type(params) not in [dict, list]:
        raise ValueError("Invalid type for params. Must be either dict or list.")

    for abacus_key, custom_keys in mapping.items():
        for custom_key in custom_keys:
            if custom_key in params:  # Check if the custom key is used
                # Replace custom key with Abacus key
                if is_dict:
                    params[abacus_key] = params.pop(custom_key)
                else:  # is list
                    params[params.index(custom_key)] = abacus_key
    return params


def get_abacus_simname(sim_type: str, cosmo_idx: int, phase_idx: int) -> str:
    """
    Get Abacus simulation name based on the simulation type, cosmology index, and phase index.

    Parameters
    ----------
    sim_type : str
        Simulation type (e.g., 'base', 'huge', 'hugebase', 'png').
    cosmo_idx : int
        Cosmology index.
    phase_idx : int
        Phase index.

    Returns
    -------
    str
        Abacus simulation name.
    """
    if sim_type == "png":
        return f"Abacus_{sim_type}base_c{cosmo_idx:03}_ph{phase_idx:03}"
    return f"AbacusSummit_{sim_type}_c{cosmo_idx:03}_ph{phase_idx:03}"


class BaseGalaxycatalog:
    """
    BaseGalaxycatalog is a base class for handling galaxy catalogs in the context of Halo Occupation Distribution (HOD) modeling.
    It provides common functionalities for loading and processing galaxy catalogs, which can be extended by child classes for specific use cases.

    The BaseGalaxycatalog class is designed to handle a multi-tracer galaxy gatalog at a fixed redshift.
    """

    logger = logging.getLogger(
        "acm.hod.BaseGalaxycatalog"
    )  # Set up logger for the class as a class attribute

    def __init__(
        self,
        redshift: float,
        boxsize: float,
        cosmology: Cosmology,
        fiducial_cosmology: Cosmology | None = None,
    ) -> None:
        """
        Parameters
        ----------
        redshift : float
            Redshift of the galaxy catalog.
        boxsize : float
            Box size of the galaxy catalog in Mpc/h.
        cosmology : Cosmology
            Cosmology to use for the galaxy catalog.
            Must have an `efunc` method to compute the expansion function
            and an `angular_diameter_distance` method to compute the angular diameter distance.
        fiducial_cosmology : Cosmology | None
            Fiducial cosmology to use for computing the Alcock-Paczynski (AP) parameters.
            Must have an `efunc` method to compute the expansion function
            and an `angular_diameter_distance` method to compute the angular diameter distance.
            If set to None, the fiducial cosmology will default to DESI.
        """
        self.boxsize = boxsize
        self.redshift = redshift
        self.cosmo_fid = (
            fiducial_cosmology if fiducial_cosmology is not None else DESI()
        )  # Fiducial cosmology (for AP parameters)

        self._setup_cosmology(cosmology)  # Set up cosmology and AP parameters

        self.catalogs = {}  # Dictionary to store the different tracers in the galaxy catalog, with tracer names as keys and tracer catalogs as values

    def _setup_cosmology(self, cosmology: Cosmology) -> None:
        """
        Set up the cosmology for the galaxy catalog and compute the Alcock-Paczynski (AP) parameters based on the redshift and cosmology.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology to use for the galaxy catalog.
            Must have an `efunc` method to compute the expansion function
            and an `angular_diameter_distance` method to compute the angular diameter distance.
        """
        self.cosmo = cosmology

        self.az = 1 / (1 + self.redshift)
        self.hubble = 100 * self.cosmo.efunc(self.redshift)
        self.q_par = 100 * self.cosmo_fid.efunc(self.redshift) / self.hubble
        self.q_perp = self.cosmo.angular_diameter_distance(
            self.redshift
        ) / self.cosmo_fid.angular_diameter_distance(self.redshift)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_tracer_catalog(key)
        else:
            # NOTE: Kept for future extension to allow more complex indexing if needed.
            raise KeyError(
                "Invalid key type. Key must be a string representing the tracer name."
            )

    def get_tracer_catalog(self, tracer: str) -> dict:
        if tracer not in self.catalogs:
            raise ValueError(
                f"Tracer {tracer} not found in the galaxy catalog. Available tracers: {list(self.catalogs.keys())}."
            )
        return self.catalogs.get(tracer)

    # TODO: Handle density measurements, filtering, etc.


class BaseHOD:
    """
    BaseHOD is a wrapper around AbacusHOD, a class for handling Halo Occupation Distribution (HOD) modeling
    using the AbacusSummit simulations.

    The BaseHOD class loads AbacusSummit snapshots for a given simulation, and samples the HOD parameters to greate galaxy catalogs.
    """

    logger = logging.getLogger(
        "acm.hod.BaseHOD"
    )  # Set up logger for the class as a class attribute

    def __init__(
        self,
        snapshots: list[float] = [0.5],
        cosmology: Cosmology | None = None,
        config_file: str | None = None,
        sim_params: dict | None = None,
        hod_setup_params: dict
        | None = None,  # TODO: Find a better name, as we will use hod_params with another method element
        load_balls: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        snapshots : list[float]
            List of redshifts for which to load the AbacusHOD balls.
        cosmology : Cosmology | None
            Cosmology to use for the HOD modeling.
            Must have an `efunc` method to compute the expansion function
            and an `angular_diameter_distance` method to compute the angular diameter distance.
            If set to None, the cosmology will be determined based on the simulation name or default to the fiducial cosmology.
            Default is None.
        config_file : str | None
            Path to the configuration file containing default values for the simulation and HOD parameters.
            If set to None, a default configuration file will be used.
            Default is None.
        sim_params : dict | None
            Dictionary of simulation parameters to use for the AbacusHOD balls.
            If set to None, default values from the configuration file will be used.
            Default is None.
        hod_setup_params : dict | None
            Dictionary of HOD setup parameters to use for the AbacusHOD balls.
            If set to None, default values from the configuration file will be used.
            Default is None.
        load_balls : bool
            Whether to load the AbacusHOD balls for the specified snapshots.
            If set to False, some functionalities of the class may not be available until the balls are loaded.
            Default is True.
        **kwargs
            Additional keyword arguments to override specific simulation parameters.
            See the `_sim_params_override` method for more details.
        """
        self.snapshots = snapshots

        # Load default values for simulation and HOD parameters from the configuration file, and update with provided values in sim_params and hod_setup_params
        config_dir = os.path.dirname(os.path.abspath(__file__))
        if config_file is None:
            config_file = (
                Path(config_dir) / "box.yaml"
            )  # TODO: change this to be handled in child classes with different default config files
            self.logger.info(
                f"No config file provided. Using default config: {config_file}."
            )

        defaults = yaml.safe_load(open(config_file))

        for d, n in zip(
            [sim_params, hod_setup_params], ["sim_params", "hod_setup_params"]
        ):
            if d is None:
                d = {}
            d.update(
                defaults.pop(n, {})
            )  # Remove elements from defaults after updating

        # NOTE: Extra defaults values are not stored or used !

        self.sim_params = map_params(sim_params, SIM_PARAMS_MAP)
        self.hod_setup_params = hod_setup_params

        self._sim_params_override(**kwargs)  # Override sim_params values from kwargs

        self._setup_cosmology(cosmology)  # Set up cosmology

        if load_balls:
            self._load_balls(
                snapshots
            )  # Load AbacusHOD balls for the specified snapshots

    def _sim_params_override(
        self,
        cosmo_idx: int = None,
        phase_idx: int = None,
        sim_type: str = None,
    ) -> None:
        """
        Overwrites class elements with specific values for the simulation and HOD parameters.
        """
        if cosmo_idx is not None and phase_idx is not None and sim_type is not None:
            self.sim_params["sim_name"] = get_abacus_simname(
                sim_type, cosmo_idx, phase_idx
            )

    def _setup_cosmology(self, cosmology: Cosmology | None = None) -> None:
        """
        Set up the cosmology for the HOD modeling.

        Parameters
        ----------
        cosmology : Cosmology | None
            Cosmology to use for the HOD modeling.
            Must have an `efunc` method to compute the expansion function
            and an `angular_diameter_distance` method to compute the angular diameter distance.
            If set to None, the cosmology will be determined based on the simulation name or default to the fiducial cosmology.
        """
        self.cosmo_fid = DESI()  # Fiducial cosmology (for AP parameters)

        if cosmology is not None:
            self.cosmo = cosmology
        elif "sim_name" in self.sim_params:
            # Extract cosmo_idx from sim_name
            sim_name = self.sim_params["sim_name"]
            cosmo_idx = int(sim_name.split("_c")[1].split("_")[0])

            if cosmo_idx in [300, 301, 302, 303]:  # Edge case for pdf cosmologies
                self.cosmo = AbacusSummit(0)
            else:
                self.cosmo = AbacusSummit(cosmo_idx)
        else:
            self.cosmo = self.cosmo_fid
            self.logger.info("No cosmology provided. Defaulting to fiducial cosmology.")

    def _load_balls(self, snapshots: list[float]) -> None:
        """
        Load AbacusHOD balls for the specified snapshots.

        Parameters
        ----------
        snapshots : list[float]
            List of redshifts for which to load the AbacusHOD balls.
            Each snapshot will be loaded with the same simulation parameters, except for the redshift which will be set to the corresponding snapshot value in sim_params.
        """
        self.balls = []
        for zsnap in snapshots:
            sim_params = self.sim_params.copy()
            sim_params["z_mock"] = zsnap
            self.logger.info(
                f"Initializing AbacusHOD ball with parameters {sim_params}."
            )
            ball = AbacusHOD(sim_params, self.hod_setup_params)
            self.balls.append(ball)
        return self.balls

    def get_ball(self, zsnap: float) -> AbacusHOD:
        """
        Get the AbacusHOD ball for a specific snapshot.

        Parameters
        ----------
        snapshot : float
            Redshift of the snapshot for which to get the AbacusHOD ball.

        Returns
        -------
        AbacusHOD
            The AbacusHOD ball corresponding to the specified snapshot.

        Raises
        ------
        ValueError
            If the specified snapshot is not in the list of loaded snapshots.
        """
        if zsnap not in self.snapshots:
            raise ValueError(
                f"Snapshot {zsnap} not found in loaded snapshots: {self.snapshots}."
            )
        return self.balls[self.snapshots.index(zsnap)]

    def _sample_hod(self):
        pass

    def sample_hods(self):
        pass


# TODO: on the first run for each tracer, read from ball the default value for HOD parameters to avoid overwriting a default value between runs
# TODO: at run level, check the hod parameters are allowed by the HOD model, and if not, raise an error.

# TODO: Specific handling for BGS tracer (not implemented in AbacusHOD)


# %% Temporary notes

# cosmo_idx and phase_idx are no longer class attributes
# Removed sim_type class attribute as it's only needed to get the filenames
# DM_DICT is now replaced by passing sim_params = lookup_registry_path('Abacus.yaml', tracer, geometry, sim_type)
# sim_geometry can be set in the child class as a class attribute
# boxsize is now read from the AbacusHOD ball
