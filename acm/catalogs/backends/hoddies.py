"""Dark matter backend based on the HODDIES HOD implementation.

The backend wraps :class:`HODDIES.HOD` and accepts halo catalogs and particle
data directly in memory, so that it can be used with simulations whose output
format is not known to ``acm``. Input data must use the column names and units
expected by HODDIES; no column mapping or unit conversion is performed here.
"""

import logging
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Protocol, cast, override

import HODDIES
import numpy as np
import pandas as pd
import yaml
from HODDIES import HOD
from mpytools import Catalog

from acm.catalogs.backends.base import SnapshotBackend, register_backend
from acm.catalogs.dataclasses import Tracer
from acm.utils.default import recursive_update

logger = logging.getLogger(__name__)

# NOTE: Intentionally duplicated from acm/catalogs/backends/abacus.py to avoid
# importing abacusnbody from this module. If a third backend needs the same
# aliases, extract this mapping and the resolver below into a shared module.
_TRACER_NAME_ALIASES = {"BGS": "LRG"}

# Columns required by HODDIES.make_mock_cat for a custom halo catalog. Rs and
# Vrms are optional (Rh and c are used by the NFW satellite model), while
# log10_Mh is derived by HODDIES when missing.
_HALO_REQUIRED_COLUMNS = ("x", "y", "z", "vx", "vy", "vz", "Mh", "Rh", "c", "halo_id")
# Columns required by HODDIES to position satellites from particle data.
_PARTICLE_REQUIRED_COLUMNS = ("x", "y", "z", "vx", "vy", "vz", "halo_id")
# Keep the same galaxy-level columns as AbacusHODBackend.
_GALAXY_COLUMN_MAP = {
    "x": "x",
    "y": "y",
    "z": "z",
    "vx": "vx",
    "vy": "vy",
    "vz": "vz",
    "mass": "Mh",
    "id": "halo_id",
    "is_cent": "Central",
}
_OUTPUT_COLUMNS = tuple(_GALAXY_COLUMN_MAP)
_CATALOG_COLUMNS = (*_GALAXY_COLUMN_MAP.values(), "TRACER")


class CatalogLike(Protocol):
    """Column-addressable catalog, e.g. a mapping or an mpytools.Catalog."""

    def __getitem__(self, key: str, /) -> np.ndarray:
        """Return the column named ``key``."""
        ...


class _Catalog:
    """Uniform column interface around a catalog container."""

    def __init__(self, data: object) -> None:
        if isinstance(data, (str, Path)):
            data = Catalog.read(str(data))
        if isinstance(data, pd.DataFrame):
            names = tuple(str(name) for name in data.columns)
            data = data.to_dict(orient="list")
        elif isinstance(data, np.ndarray):
            if data.dtype.names is None:
                raise TypeError(
                    "Structured array with named fields required for halo or particle data."
                )
            names = data.dtype.names
            data = {name: data[name] for name in names}
        elif isinstance(data, Mapping):
            names = tuple(str(name) for name in data)
        else:
            columns = getattr(data, "columns", None)
            if not callable(columns):
                raise TypeError(
                    "Unsupported catalog container "
                    f"'{type(data).__name__}'. Use a mapping, a structured numpy array, "
                    "a pandas DataFrame, an mpytools.Catalog or a file path."
                )
            names = tuple(str(name) for name in columns())
        self.data = cast("CatalogLike", data)
        self.names = names

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def extract(self, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
        """Copy the requested columns from the catalog."""
        available = set(self.names)
        missing = [name for name in columns if name not in available]
        if missing:
            raise KeyError(
                f"The HODDIES catalog is missing expected columns {missing}. "
                f"Available columns are {sorted(available)}."
            )
        return {name: np.array(self[name], copy=True) for name in columns}


def _prepare_data(
    data: object,
    required_columns: tuple[str, ...],
    kind: str,
) -> _Catalog:
    """Normalize and validate halo or particle data."""
    prepared = _Catalog(data)
    available = set(prepared.names)
    missing = [name for name in required_columns if name not in available]
    if missing:
        raise ValueError(
            f"The {kind} is missing required columns {missing}. "
            f"Required columns are {list(required_columns)}; "
            f"available columns are {sorted(available)}."
        )
    return prepared


def _as_redshift(value: float | str) -> float:
    """Convert a batch-dictionary key to a float redshift.

    Parameters
    ----------
    value : float or str
        Key of a ``{redshift: data}`` mapping.

    Returns
    -------
    float
        The redshift as a float.

    Raises
    ------
    TypeError
        If the key cannot be interpreted as a redshift.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Batch keys must be redshifts, got {value!r}. For a single snapshot, "
            "pass hcat directly together with redshift."
        ) from exc


@register_backend("HODDIES")
class HODDIESBackend(SnapshotBackend):
    """Dark matter backend that populates galaxies with the HODDIES HOD model.

    Unlike :class:`~acm.catalogs.backends.abacus.AbacusHODBackend`, this backend
    knows nothing about a particular simulation suite. Halo catalogs and
    particle data are provided in memory when loading a snapshot, which makes it
    usable with any simulation output that follows the HODDIES column convention.

    Notes
    -----
    Positions are expected in Mpc/h, velocities in km/s, ``Mh`` in Msun/h,
    ``Rh``/``Rs`` in kpc/h, ``Vrms`` in km/s and ``c`` is dimensionless.

    HODDIES adds a ``log10_Mh`` column to the halo catalog it receives. When an
    ``mpytools.Catalog`` is passed directly, that object is modified in place.

    When particles are provided and ``use_particles`` is enabled, satellites are
    positioned with HODDIES' native generic particle path. That path requires a
    HODDIES version containing the halo-wise assignment fix; if upstream does
    not yet include it, install a patched HODDIES version manually.
    """

    def __init__(self, config_file: str | Path | None = None, **kwargs) -> None:
        """Initialize the backend from a HODDIES YAML configuration file.

        Parameters
        ----------
        config_file : str or Path, optional
            HODDIES parameter file. It is passed directly as ``param_file`` to
            :class:`HODDIES.HOD`. Defaults to the ``default_HOD_parameters.yaml``
            shipped with HODDIES.
        **kwargs
            Nested overrides for the configuration, using the same semantics as
            ``HODDIES.HOD``. At least ``hcat.boxsize`` must be set, either in
            the file or through ``kwargs`` (e.g. ``hcat={"boxsize": 1000}``).

        Raises
        ------
        ValueError
            If the configuration file does not exist or no positive box size
            is provided.
        """
        super().__init__()

        if config_file is None:
            config_file = (
                Path(HODDIES.__file__).resolve().parent / "default_HOD_parameters.yaml"
            )
        self._config_file = Path(config_file)
        if not self._config_file.exists():
            raise ValueError(
                f"Config file '{self._config_file}' not found. Provide a valid HODDIES parameter file."
            )

        with self._config_file.open() as file:
            config = yaml.safe_load(file) or {}
        if not isinstance(config, Mapping):
            raise TypeError(
                f"Config file '{self._config_file}' must contain a mapping at the top level."
            )

        self._config = recursive_update(dict(config), kwargs)
        self._hod_kwargs = deepcopy(dict(kwargs))

        hcat_config = self._config.get("hcat", {})
        boxsize = (
            hcat_config.get("boxsize") if isinstance(hcat_config, Mapping) else None
        )
        if boxsize is None:
            raise ValueError(
                "A box size is required for HODDIESBackend. Set 'hcat.boxsize' in "
                "the config file or pass e.g. hcat={'boxsize': 1000}."
            )
        try:
            boxsize = float(boxsize)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid box size '{boxsize}'. It must be a positive number."
            ) from exc
        if boxsize <= 0:
            raise ValueError(
                f"Invalid box size '{boxsize}'. It must be a positive number."
            )
        self._boxsize = boxsize

        self._cache: dict[float, HOD] = {}

    @override
    def load_dark_matter_catalog(
        self,
        redshift: float | None = None,
        hcat: object | None = None,
        particles: object | None = None,
        **kwargs,
    ) -> None:
        """Build and cache a HODDIES catalog for one or more snapshots.

        Parameters
        ----------
        redshift : float, optional
            Redshift of the snapshot. If ``None``, ``hcat`` must be a mapping
            of ``{redshift: halo_data}`` and all snapshots are loaded at once.
        hcat : object, optional
            Halo catalog for a single snapshot. Supported containers are
            dictionaries, structured numpy arrays, pandas DataFrames,
            ``mpytools.Catalog`` objects and file paths. It must contain at
            least ``x, y, z, vx, vy, vz, Mh, Rh, c, halo_id``. If ``None``,
            HODDIES reads the path set in ``hcat.path_to_sim``.
        particles : object, optional
            Particle data used to position satellites. It must contain at
            least ``x, y, z, vx, vy, vz, halo_id`` and requires
            ``use_particles: true`` in the configuration. When omitted,
            satellites are drawn from an NFW profile.
        **kwargs
            Configuration overrides for this load only. They are merged into
            the constructor overrides and do not modify the backend.

        Raises
        ------
        TypeError
            If ``redshift`` is ``None`` and ``hcat`` is not a mapping.
        ValueError
            If particle data are provided while ``use_particles`` is disabled,
            if batch inputs have inconsistent redshift keys, or if required
            columns are missing.
        """
        if redshift is not None:
            self.load_snapshot(float(redshift), hcat, particles, kwargs)
            return

        if not isinstance(hcat, Mapping):
            raise TypeError(
                "When redshift is None, hcat must be a mapping of "
                "{redshift: halo_data}. Pass redshift for a single snapshot."
            )
        hcat_batch = {_as_redshift(z): data for z, data in hcat.items()}
        if particles is None:
            particle_batch: dict[float, object | None] = dict.fromkeys(hcat_batch)
        else:
            if not isinstance(particles, Mapping):
                raise TypeError(
                    "When redshift is None, particles must be a mapping of "
                    "{redshift: particle_data}."
                )
            particle_batch = {_as_redshift(z): data for z, data in particles.items()}
            if set(particle_batch) != set(hcat_batch):
                raise ValueError(
                    "The redshifts in particles must match the redshifts in "
                    f"hcat. Got {sorted(particle_batch)} and {sorted(hcat_batch)}."
                )
        for snapshot_redshift, snapshot_hcat in hcat_batch.items():
            self.load_snapshot(
                snapshot_redshift,
                snapshot_hcat,
                particle_batch[snapshot_redshift],
                kwargs,
            )

    @override
    def get_dark_matter_catalog(self, redshift: float) -> HOD:
        """Return the cached HODDIES catalog for ``redshift``.

        Parameters
        ----------
        redshift : float
            Redshift of the snapshot.

        Returns
        -------
        HODDIES.HOD
            The cached HODDIES object.

        Raises
        ------
        KeyError
            If no catalog has been loaded for ``redshift``.
        """
        redshift = float(redshift)
        if redshift not in self._cache:
            raise KeyError(
                f"Dark matter catalog for redshift z={redshift:.3f} has not been "
                f"loaded yet. Available redshifts: {sorted(self._cache)}. "
                "Call load_dark_matter_catalog first."
            )
        return self._cache[redshift]

    @override
    def make_galaxy_catalog(
        self,
        dm_catalog: HOD,
        tracers: list[Tracer],
        fix_seed: int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> dict[Tracer, pd.DataFrame]:
        """Generate galaxy catalogs for each requested tracer.

        Parameters
        ----------
        dm_catalog : HODDIES.HOD
            Object returned by :meth:`get_dark_matter_catalog`.
        tracers : list of Tracer
            Tracers to populate. ``Tracer.params`` may override any parameter
            already defined for that tracer in the configuration file.
        fix_seed : int, optional
            Random seed forwarded to :meth:`HODDIES.HOD.make_mock_cat`.
        verbose : bool, optional
            Forwarded to :meth:`HODDIES.HOD.make_mock_cat`. Defaults to False.
        **kwargs
            Forwarded to :meth:`HODDIES.HOD.make_mock_cat`.

        Returns
        -------
        dict[Tracer, DataFrame]
            One DataFrame per tracer with exactly the columns
            ``x, y, z, vx, vy, vz, mass, id, is_cent``.

        Raises
        ------
        KeyError
            If a requested tracer is not defined in the configuration.
        ValueError
            If tracer parameters are unknown or two tracers resolve to the
            same HODDIES tracer.

        Notes
        -----
        When the cached HOD was loaded with particle data and ``use_particles``
        is enabled, HODDIES positions satellites with host-halo particles,
        falling back to NFW when a halo has too few particles.
        """
        available = dm_catalog.args.get("tracers", [])
        if isinstance(available, str):
            available = [available]

        resolved = []
        for tracer in tracers:
            tracer_name = self._resolve_tracer_name(tracer.name)
            if tracer_name not in available:
                raise KeyError(
                    f"Tracer '{tracer.name}' (resolved to '{tracer_name}') is not "
                    f"defined in the configuration. Available tracers: {available}."
                )
            if tracer_name in resolved:
                raise ValueError(
                    f"Tracer '{tracer.name}' resolves to '{tracer_name}', which is "
                    "already requested by another tracer."
                )
            resolved.append(tracer_name)

        merged_args = deepcopy(dm_catalog.args)
        for tracer, tracer_name in zip(tracers, resolved, strict=True):
            params = merged_args[tracer_name]
            invalid = set(tracer.params) - set(params)
            if invalid:
                raise ValueError(
                    f"HOD parameters for tracer '{tracer.name}' contain invalid "
                    f"keys: {sorted(invalid)}. Valid keys are: {sorted(params)}."
                )
            merged_args[tracer_name] = recursive_update(params, tracer.params)

        original_args = dm_catalog.args
        try:
            dm_catalog.args = merged_args
            catalog = dm_catalog.make_mock_cat(
                tracers=resolved, fix_seed=fix_seed, verbose=verbose, **kwargs
            )
        finally:
            dm_catalog.args = original_args

        columns = _Catalog(catalog).extract(_CATALOG_COLUMNS)
        return {
            tracer: self._to_dataframe(columns, tracer_name)
            for tracer, tracer_name in zip(tracers, resolved, strict=True)
        }

    @property
    @override
    def boxsize(self) -> float:
        """Box size in Mpc/h used by all snapshots handled by this backend."""
        return self._boxsize

    def load_snapshot(
        self,
        redshift: float,
        hcat: object | None,
        particles: object | None,
        overrides: Mapping,
    ) -> None:
        """Build and cache the HODDIES object for a single snapshot."""
        load_config = recursive_update(deepcopy(self._config), overrides)
        hod_kwargs = recursive_update(deepcopy(self._hod_kwargs), overrides)

        use_particles = bool(load_config.get("use_particles", False))
        if particles is not None and not use_particles:
            raise ValueError(
                "Particle data were provided but 'use_particles' is not enabled. "
                "Set 'use_particles: true' in the config file, pass "
                "use_particles=True, or omit particles to draw NFW satellites."
            )

        prepared_hcat = (
            None
            if hcat is None
            else _prepare_data(hcat, _HALO_REQUIRED_COLUMNS, "halo catalog")
        )
        prepared_particles = (
            None
            if particles is None
            else _prepare_data(
                particles, _PARTICLE_REQUIRED_COLUMNS, "particle catalog"
            )
        )
        hcat_overrides = hod_kwargs.get("hcat", {})
        if not isinstance(hcat_overrides, Mapping):
            raise TypeError("The 'hcat' configuration entry must be a mapping.")
        hcat_overrides = dict(hcat_overrides)
        hcat_overrides["z_simu"] = redshift
        hod_kwargs["hcat"] = hcat_overrides

        logger.debug(
            "Loading HODDIES catalog at z=%.3f with use_particles=%s",
            redshift,
            use_particles and prepared_particles is not None,
        )
        hod = HOD(
            param_file=self._config_file,
            hcat_file=None if prepared_hcat is None else prepared_hcat.data,
            subsample=None if prepared_particles is None else prepared_particles.data,
            **hod_kwargs,
        )
        self._cache[redshift] = hod

    @staticmethod
    def _resolve_tracer_name(name: str) -> str:
        """Resolve a tracer name to the name expected by the configuration."""
        resolved = _TRACER_NAME_ALIASES.get(name, name)
        if resolved != name:
            logger.warning(
                f"Tracer '{name}' is not directly supported. Using '{resolved}' as a proxy."
            )
        return resolved

    @staticmethod
    def _to_dataframe(
        catalog: Mapping[str, np.ndarray],
        tracer_name: str,
    ) -> pd.DataFrame:
        """Extract one tracer from a HODDIES catalog and align its columns."""
        mask = np.asarray(catalog["TRACER"]) == tracer_name
        data = {
            target: np.asarray(catalog[source])[mask]
            for target, source in _GALAXY_COLUMN_MAP.items()
        }
        data["is_cent"] = data["is_cent"].astype(bool)
        return pd.DataFrame(data)[list(_OUTPUT_COLUMNS)]
