import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import h5py
import pandas as pd
from cosmoprimo import Cosmology

from acm.catalogs.dataclasses import Tracer, Transform

logger = logging.getLogger(__name__)


class BaseGalaxyCatalog(ABC):
    """
    Stores galaxy data for multiple tracers.

    GalaxyCatalog is geometry-agnostic: it does not know how the data was
    produced or what columns it contains. Subclasses (CubicGalaxyCatalog,
    CutskyGalaxyCatalog, etc.) may add geometry-specific behaviour, but the
    base storage and retrieval interface is defined here.

    Cosmology is passed in from the factory and stored as references, so all
    snapshots in a factory share the same cosmo and cosmo_fid objects.
    """

    def __init__(
        self,
        cosmo: Cosmology,
        cosmo_fid: Cosmology,
    ) -> None:
        """
        Initialize the galaxy catalog with the given cosmologies.

        Parameters
        ----------
        cosmo : cosmoprimo.Cosmology, optional
            The cosmology for the simulation.
        cosmo_fid : cosmoprimo.Cosmology, optional
            The fiducial cosmology.
        """
        self.cosmo = cosmo
        self.cosmo_fid = cosmo_fid
        self.tracers: dict[str, Tracer] = {}
        self._data: dict[str, pd.DataFrame] = {}
        self._transforms: dict[str, Transform] = {}
        self._transform_state: int = 0  # Incremented on each transform change

    def __repr__(self) -> str:
        """Provide a string representation of the galaxy catalog, including tracer information."""
        return f"{self.__class__.__name__}(tracers={list(self.tracers.keys())})"

    def register_tracer(self, tracer: Tracer) -> None:
        """Register a tracer in the catalog."""
        if tracer.name in self.tracers:
            logger.warning(
                f"Tracer '{tracer.name}' already exists and will be replaced."
            )
        self.tracers[tracer.name] = tracer

    def set_tracer_data(self, tracer: Tracer, data: pd.DataFrame) -> None:
        """Set the galaxy data for a given tracer."""
        if not self._check_data_columns(data):
            raise ValueError(
                f"Data for tracer '{tracer.name}' is missing required columns."
            )
        self.register_tracer(tracer)  # Ensure tracer is registered before setting data
        self._data[tracer.name] = data

    def get_tracer_data(self, *tracers: str, raw: bool = False) -> pd.DataFrame:
        """
        Return tracer data with all pipeline transforms applied for the specified tracers.

        No transformations are applied if `raw=True`.

        Parameters
        ----------
        tracers : str
            Name of the tracers to retrieve.
        raw : bool, optional
            If True, return the raw data without applying transforms. Default is False.

        Returns
        -------
        pd.DataFrame
            The tracer data with transforms applied (or raw if `raw=True`).

        Raises
        ------
        RuntimeError
            If no tracers are loaded in the catalog.
        ValueError
            If no tracer names are specified.
        KeyError
            If any specified tracer names are not found in the catalog.
        """
        if not self.tracers:
            raise RuntimeError(
                "No tracers loaded in the catalog, cannot retrieve data."
            )

        if not tracers:
            raise ValueError("At least one tracer name must be specified.")

        if any(tracer not in self.tracers for tracer in tracers):
            missing = [tracer for tracer in tracers if tracer not in self.tracers]
            raise KeyError(f"Tracers not found in catalog: {missing}")

        tracers_data = []
        for tracer in tracers:
            data = self._data[tracer].copy()
            if not raw:
                for transform in self._transforms.values():
                    if transform.tracer is None or transform.tracer == tracer:
                        data = transform.apply(data)
            tracers_data.append(data)
        return pd.concat(tracers_data, ignore_index=True)

    def _ngal(self, *tracers: str) -> int:
        """Return the total number of galaxies for specified tracers, or the full catalog otherwise."""
        tracers = tracers or tuple(self.tracers)
        return len(self.get_tracer_data(*tracers))

    @property
    def ngal(self) -> int:
        """Total number of galaxies in the catalog across all tracers."""
        return self._ngal()

    @abstractmethod
    def _check_data_columns(self, data: pd.DataFrame) -> bool:
        """Check that the required columns for a tracer are present in the data before assignment."""
        ...

    @property
    def transform_pipeline(self) -> list[str]:
        """Return the list of transform names in the current pipeline."""
        return list(self._transforms)

    def _add_transform(self, transform: Transform) -> None:
        """Register or replace a transform in the pipeline."""
        if transform.name in self._transforms:
            logger.warning(
                f"Transform '{transform.name}' already exists and will be replaced."
            )
        self._transforms[transform.name] = transform
        self._transform_state += 1

    def _remove_transform(self, name: str) -> None:
        """Remove a transform from the pipeline."""
        if name not in self._transforms:
            raise KeyError(f"Transform '{name}' is not in the pipeline.")
        del self._transforms[name]
        self._transform_state += 1

    def clear_transforms(self) -> None:
        """Clear all transforms from the pipeline."""
        self._transforms.clear()
        self._transform_state += 1

    def __getitem__(self, tracers: str | tuple[str]) -> pd.DataFrame:
        """Allow direct indexing to get tracer data, e.g. catalog['ELG']."""
        _tracers = (tracers,) if isinstance(tracers, str) else tuple(tracers)
        return self.get_tracer_data(*_tracers)

    def __len__(self) -> int:
        """Return the total number of galaxies across all raw catalogs."""
        return sum(len(data) for data in self._data.values())

    def save(self, path: str | Path, *columns: str) -> None:
        """
        Save the catalog to an HDF5 file.

        Saves raw (pre-transform) tracer data and tracer metadata.
        The transform pipeline is not serialized; transforms must be
        re-registered after loading.

        Parameters
        ----------
        path : str | Path
            Path to the output HDF5 file.
        *columns : str
            Columns to save. If no columns are specified, all columns are saved.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._transforms:
            logger.warning(
                f"Transform pipeline is not serialized. "
                f"Re-register transforms after loading: {self.transform_pipeline}"
            )

        with h5py.File(path, "w") as f:
            f.attrs["catalog_class"] = self.__class__.__name__
            f.attrs["tracers"] = json.dumps(
                {name: tracer.params for name, tracer in self.tracers.items()}
            )
            self._save_attrs(f)  # subclass-specific attributes

            for tracer_name, data in self._data.items():
                grp = f.create_group(tracer_name)
                cols = list(columns or data.columns)
                grp.attrs["columns"] = cols  # preserve column order
                for col in cols:
                    grp.create_dataset(col, data=data[col].values)

        logger.info(f"Saved {self.__class__.__name__} to {path}")

    @classmethod
    def load(cls, path: str | Path, cosmo: Cosmology, cosmo_fid: Cosmology) -> Self:
        """
        Load a catalog from an HDF5 file.

        Parameters
        ----------
        path : str | Path
            Path to the HDF5 file.
        cosmo : Cosmology
            Simulation cosmology — not serialized, must be provided explicitly.
        cosmo_fid : Cosmology
            Fiducial cosmology — not serialized, must be provided explicitly.

        Returns
        -------
        BaseGalaxyCatalog
            An instance of the calling class with tracer data loaded.
        """
        path = Path(path)

        with h5py.File(path, "r") as f:
            tracer_meta = json.loads(f.attrs["tracers"])
            extra_attrs = dict(f.attrs)  # Extra attributes saved by the subclass

            catalog = cls._from_attrs(
                extra_attrs, cosmo, cosmo_fid
            )  # subclass reconstruction

            for tracer_name, params in tracer_meta.items():
                tracer = Tracer(name=tracer_name, params=params)
                grp = f[tracer_name]
                columns = list(grp.attrs["columns"])  # preserve column order
                data = pd.DataFrame({col: grp[col][:] for col in columns})
                catalog.set_tracer_data(tracer, data)

        logger.info(f"Loaded {cls.__name__} from {path}")
        return catalog

    @abstractmethod
    def _save_attrs(self, f: h5py.File) -> None:
        """Save subclass-specific attributes to the HDF5 file root."""
        ...

    @classmethod
    @abstractmethod
    def _from_attrs(cls, attrs: dict, cosmo: Cosmology, cosmo_fid: Cosmology) -> Self:
        """Reconstruct a catalog instance from HDF5 root attributes."""
        ...
