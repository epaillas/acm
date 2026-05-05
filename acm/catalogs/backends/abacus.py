import logging
import time
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import yaml
from abacusnbody.hod.abacus_hod import (
    AbacusHOD,  # pyright: ignore[reportMissingImports]
)

from acm.catalogs.backends.base import SnapshotBackend, register_backend
from acm.catalogs.dataclasses import Tracer
from acm.utils.abacus import BOXSIZES, get_abacus_simname, map_params

logger = logging.getLogger(__name__)

_TRACER_NAME_ALIASES = {"BGS": "LRG"}


@register_backend("AbacusHOD")
class AbacusHODBackend(SnapshotBackend):
    """Dark matter backend for AbacusSummit simulations using a simple HOD model to populate the galaxy catalog."""

    def __init__(
        self,
        cosmo_idx: int = 0,
        phase_idx: int = 0,
        sim_type: str = "base",
        config_file: str | Path | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the AbacusHODBackend with any necessary parameters.

        Parameters
        ----------
        cosmo_idx : int
            Index of the cosmology to use from the AbacusSummit suite.
        phase_idx : int
            Index of the phase to use from the AbacusSummit suite.
        sim_type : str
            Type of simulation to use (e.g. "base", "huge", "small").
            This determines the box size and resolution of the simulation.
        config_file : str | Path, optional
            Path to the configuration file containing simulation and HOD parameters.
            If not provided, default parameters will be used.
        **kwargs
            Extra parameters to override simulation parameters.
            These will take precedence over the config file parameters.

        Raises
        ------
        ValueError
            If the provided sim_type is not recognized.
        ValueError
            If the config file is not found.
        """
        if sim_type not in BOXSIZES:
            raise ValueError(
                f"Unknown simulation type '{sim_type}'. Available types: {list(BOXSIZES)}"
            )

        # Default config file if not provided
        config_file = config_file or Path(__file__).parent / "abacus_default.yaml"
        config_file = Path(config_file)  # Ensure Path behavior
        if not config_file.exists():
            raise ValueError(
                f"Config file '{config_file}' not found. Please provide a valid config file with simulation and HOD parameters."
            )

        # Get sim_params and HOD_params from config file
        config = yaml.safe_load(config_file.open())
        sim_params = config.get("sim_params", {})
        hod_params = config.get("HOD_params", {})
        logger.debug(f"Loaded sim_params from config: {sim_params}")
        logger.debug(f"Loaded HOD_params from config: {hod_params}")

        # Update sim_params with kwargs
        sim_params.update(kwargs)

        # Build simname based on the provided parameters
        sim_name = get_abacus_simname(sim_type, cosmo_idx, phase_idx)
        sim_params["sim_name"] = sim_name

        logger.debug(
            f"Loading AbacusHODBackend {sim_type} with parameters: {sim_params}"
        )

        # Store relevant parameters as attributes for later use
        self.sim_type = sim_type
        self.sim_params = sim_params
        self.hod_params = hod_params

    @override
    def get_dark_matter_catalog(
        self,
        redshift: float,
        **kwargs,
    ) -> AbacusHOD:
        sim_params = self.sim_params.copy()
        sim_params["z_mock"] = redshift

        hod_params = self.hod_params
        self.update_default_tracers(hod_params, **kwargs)

        t0 = time.time()
        dark_matter_catalog = AbacusHOD(sim_params, hod_params)
        logger.debug(
            f"Loaded dark matter catalog for redshift z={redshift:.3f} in {time.time() - t0:.2f} seconds."
        )

        return dark_matter_catalog

    @override
    def make_galaxy_catalog(
        self,
        dm_catalog: AbacusHOD,
        tracers: list[Tracer],
        use_logsigma: bool = False,
        mapping: dict | None = None,
        **kwargs,
    ) -> dict[Tracer, pd.DataFrame]:
        """
        Generate galaxy catalogs for each tracer using the HOD model.

        Parameters
        ----------
        dm_catalog : AbacusHOD
            Dark matter catalog instance returned by get_dark_matter_catalog.
        tracers : list[Tracer]
            List of tracers to populate. HOD parameters in each tracer instance
            override the defaults loaded at initialization.
        use_logsigma : bool, optional
            If True, interprets the 'sigma' HOD parameter as log10(sigma) and
            converts it before passing to AbacusHOD. Default is False.
        mapping : dict, optional
            Optional parameter name remapping, e.g. {"my_sigma": "sigma"}.
            Useful when parameter names differ from AbacusHOD's convention.
        **kwargs
            Extra arguments forwarded to AbacusHOD.run_hod. Supports aliases:
            - reseed / seed : random seed for HOD resampling
            - Nthread / nthreads : number of threads

        Returns
        -------
        dict[Tracer, DataFrame]
            Dictionary mapping each Tracer to a DataFrame of galaxy positions
            and properties (columns are uppercased AbacusHOD output keys,
            plus an 'IS_CENT' boolean column).

        Raises
        ------
        ValueError
            If BGS is requested alongside other tracers.
        ValueError
            If any HOD parameter key is not a valid AbacusHOD parameter for that tracer.
        """
        if any(t.name == "BGS" for t in tracers) and len(tracers) > 1:
            raise ValueError(
                "BGS tracer cannot be generated together with other tracers using the current implementation. Please generate BGS separately or remove it from the tracers list."
            )

        # Extract and update HOD parameters to modify and pass to run_hod
        catalog_tracers = dm_catalog.tracers.copy()
        final_tracers = {}
        for tracer in tracers:
            tracer_name = self._resolve_tracer_name(tracer.name)

            default_params = catalog_tracers[tracer_name]
            hod_params = tracer.params.copy()
            hod_params = map_params(
                hod_params, mapping=mapping
            )  # Map custom parameter names to AbacusHOD parameter names if needed

            # Convert log(sigma) to sigma and/or remove logsigma
            if use_logsigma and "sigma" in hod_params:
                hod_params["sigma"] = 10 ** hod_params["sigma"]
            if "logsigma" in hod_params:
                hod_params["sigma"] = 10 ** hod_params.pop("logsigma")

            # Check HOD parameters can override the default HOD parameters.
            if not set(hod_params).issubset(set(default_params)):
                invalid = set(hod_params) - set(default_params)
                valid = set(default_params)
                raise ValueError(
                    f"HOD parameters for tracer '{tracer.name}' contain invalid keys: {invalid}. Valid keys are: {valid}."
                )

            # NOTE: Do we want to provide this option to the user ?
            hod_params["ic"] = 1  # set incompleteness to 1 (i.e. no incompleteness)

            default_params.update(hod_params)
            final_tracers[tracer_name] = default_params
            logger.debug(
                f"Updating tracer '{tracer.name}' with HOD parameters: {hod_params}"
            )

        # Handle kwarg names for backwards compatibility (pop all)
        seed = kwargs.pop("seed", None)
        reseed = (
            kwargs.pop("reseed", None) or seed or None
        )  # default to None if not specified or 0
        nthreads = kwargs.pop("nthreads", None)
        Nthread = (
            kwargs.pop("Nthread", None) or nthreads or 1
        )  # default to 1 thread if not specified

        # TODO: handle density & incompleteness here ? NOTE: requires cosmology information !
        # TODO: handle NFW profile for ELG here ?

        galaxy_dict = dm_catalog.run_hod(
            final_tracers,
            want_rsd=False,
            reseed=reseed,
            Nthread=Nthread,
            **kwargs,
        )

        # Process AbacusHOD output
        galaxy_catalogs = {}
        for tracer in tracers:
            tracer_name = self._resolve_tracer_name(tracer.name)

            self._add_centrals(galaxy_dict, tracer_name)

            columns = [k.upper() for k in galaxy_dict[tracer_name]]
            galaxy_catalogs[tracer] = pd.DataFrame.from_dict(
                galaxy_dict[tracer_name], columns=columns
            )

        return galaxy_catalogs

    def update_default_tracers(
        self, hod_params: dict, tracers: list[Tracer] | None = None
    ) -> None:
        """
        Update the default HOD parameters dictionary for each tracer in hod_params based on the provided tracer instances.

        Required for the correct loading of the AbacusHOD class, which expects default HOD parameters for each tracer at initialization.
        If

        Parameters
        ----------
        hod_params : dict
            The initial HOD parameters to be passed to AbacusHOD, which may contain default parameters for each tracer.
        tracers : list[Tracer], optional
            List of tracer instances to override the default HOD parameters.
            HOD parameters in each tracer instance will override the defaults in hod_params.

        Raises
        ------
        ValueError
            If default HOD parameters for any tracer are missing after the update.
        """
        tracers = tracers or []
        tracer_flags = hod_params.get("tracer_flags", {})

        for tracer in tracers:
            tracer_key = f"{tracer.name}_params"

            # Override tracer-specific HOD parameters with any provided in the tracer instance
            tracer_params = hod_params.get(tracer_key, {})
            tracer_params.update(tracer.params)
            hod_params[tracer_key] = tracer_params

            if len(tracer_params) == 0:
                raise ValueError(
                    f"Default HOD parameters for tracer '{tracer.name}' must be provided either through the config file, as kwargs, or in the tracer instance."
                )

            logger.debug(
                f"Setting default HOD parameters for tracer '{tracer.name}': {tracer_params}"
            )

            # Ensure flag is True, even if it wasn't set in the config file
            tracer_flags[tracer.name] = True

        # Update tracer_flags in hod_params
        hod_params["tracer_flags"] = tracer_flags
        active_tracers = [k for k in tracer_flags if tracer_flags[k]]
        if len(active_tracers) == 0:
            raise ValueError(
                "At least one tracer must be active (i.e. have tracer_flags[tracer_name] = True). Please check your config file and tracer parameters."
            )
        logger.debug(f"Loading AbacusHOD for tracers: {active_tracers}")

    @staticmethod
    def _add_centrals(galaxy_dict: dict, tracer_name: str) -> None:
        """
        Add an 'is_cent' boolean column to the galaxy dict in-place.

        AbacusHOD stores the number of centrals as 'Ncent' and orders galaxies
        so that the first Ncent entries are centrals. This method pops 'Ncent'
        and replaces it with a boolean array.

        Parameters
        ----------
        galaxy_dict : dict
            Raw output dictionary from AbacusHOD.run_hod, modified in-place.
        tracer_name : str
            Key into galaxy_dict identifying the tracer to process.

        Raises
        ------
        KeyError
            If 'Ncent' key is not found in the galaxy_dict for the specified tracer.
        """
        n_cent = galaxy_dict[tracer_name].pop("Ncent", None)
        if n_cent is None:
            raise KeyError(
                f"'Ncent' key not found in galaxy_dict for tracer '{tracer_name}'."
            )

        n_gal = len(galaxy_dict[tracer_name]["x"])
        logger.debug(
            f"Adding central/satellite flag for tracer '{tracer_name}' with {n_gal} galaxies (Ncent={n_cent})."
        )
        is_central = np.zeros(n_gal, dtype=bool)
        is_central[:n_cent] = 1
        galaxy_dict[tracer_name]["is_cent"] = is_central

    def _resolve_tracer_name(self, name: str) -> str:
        """
        Resolve a tracer name to the name expected by AbacusHOD, using the _TRACER_NAME_ALIASES mapping.

        Parameters
        ----------
        name : str
            Tracer name as used in the pipeline (e.g. "BGS").

        Returns
        -------
        str
            Tracer name as expected by AbacusHOD (e.g. "LRG").
        """
        resolved = _TRACER_NAME_ALIASES.get(name, name)
        if resolved != name:
            logger.warning(
                f"Tracer '{name}' is not directly supported. Using '{resolved}' as a proxy."
            )
        return resolved

    # NOTE: do we really need this here ?
    @property
    def boxsize(self) -> float:
        """Box size of the AbacusSummit simulations in Mpc/h."""
        return BOXSIZES.get(self.sim_type, 500)
