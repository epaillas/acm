"""Concrete catalog factories for snapshot-based pipelines."""

import logging
from typing import override
from abc import abstractmethod

from cosmoprimo import Cosmology

from acm.catalogs.base import BaseCatalogFactory
from acm.catalogs.backends import SnapshotBackend
from acm.catalogs.dataclasses import Tracer
from acm.catalogs.galaxy_catalogs import SnapshotCatalog

logger = logging.getLogger(__name__)


class SnapshotCatalogFactory(BaseCatalogFactory):
    """
    Abstract base class for snapshot-based catalog factories.

    Subclasses must implement make_catalogs and get_catalog.
    """

    def __init__(
        self,
        backend: str | SnapshotBackend,
        catalog_class: type[SnapshotCatalog],
        cosmo: Cosmology,
        cosmo_fid: Cosmology | None = None,
        **kwargs,
    ) -> None:
        super().__init__(backend, catalog_class, cosmo, cosmo_fid, **kwargs)
        self.backend: SnapshotBackend  # type hint for better autocompletion
        self.catalog_class: type[SnapshotCatalog]  # type hint for better autocompletion

    @abstractmethod
    def make_catalogs(
        self,
        redshifts: list[float],
        tracers: list[Tracer] | dict[float, list[Tracer]],
        **kwargs,
    ) -> None:
        """
        Load dark matter snapshots and populate galaxy catalogs for each redshift.

        Parameters
        ----------
        redshifts : list[float]
            List of redshifts at which to load dark matter snapshots.
        tracers : list[Tracer] | dict[float, list[Tracer]]
            Tracers to populate for each redshift. Can be a single list applied to all redshifts
            or a dictionary mapping each redshift to its own list of tracers.
        dark_matter_kwargs : dict, optional
            Keyword arguments forwarded to the backend when loading the dark matter catalog (e.g. default tracer parameters).
        **kwargs
            Extra arguments forwarded to the backend.
        """
        ...

    @abstractmethod
    def get_catalog(self, redshift: float) -> SnapshotCatalog:
        """
        Retrieve the galaxy catalog at a given redshift.

        Parameters
        ----------
        redshift : float
            The redshift of the desired snapshot.
        """
        ...


class GalaxyCatalogFactory(SnapshotCatalogFactory):
    """Snapshot-based factory: Load a dark matter backend and create galaxy catalogs across multiple redshift snapshots."""

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"backend={self.backend.__class__.__name__}, "
            f"catalog_class={self.catalog_class.__name__}, "
            f"redshifts={self.redshifts})"
        )

    @property
    def redshifts(self) -> list[float]:
        """List of redshifts for which catalogs have been loaded."""
        return list(self._catalogs.keys())

    @override
    def make_catalogs(
        self,
        redshifts: list[float],
        tracers: list[Tracer] | dict[float, list[Tracer]],
        dark_matter_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        for z in redshifts:
            snapshot_tracers = tracers if isinstance(tracers, list) else tracers[z]

            logger.info(f"Loading dark matter catalog at redshift z={z:.3f}")
            dm_kwargs = dark_matter_kwargs or {}
            dm_catalog = self.backend.get_dark_matter_catalog(redshift=z, **dm_kwargs)

            logger.info(
                f"Populating galaxy catalog at redshift z={z:.3f} for tracers {[t.name for t in snapshot_tracers]}"
            )
            tracer_data = self.backend.make_galaxy_catalog(
                dm_catalog=dm_catalog,
                tracers=snapshot_tracers,
                **kwargs,
            )

            galaxy_catalog = self.catalog_class(
                redshift=z,
                cosmo=self.cosmo,
                cosmo_fid=self.cosmo_fid,
            )
            for tracer, data in tracer_data.items():
                galaxy_catalog.set_tracer_data(tracer, data)

            self._catalogs[z] = galaxy_catalog

    @override
    def get_catalog(self, redshift: float) -> SnapshotCatalog:
        """
        Retrieve the galaxy catalog at a given redshift.

        Parameters
        ----------
        redshift : float
            The redshift of the desired snapshot.
        """
        if redshift not in self._catalogs:
            raise KeyError(
                f"No catalog loaded at z={redshift}. "
                f"Available redshifts: {list(self._catalogs.keys())}"
            )
        return self._catalogs[redshift]
