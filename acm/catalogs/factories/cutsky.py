"""Concrete catalog factories for cutsky-based pipelines."""
# ruff: noqa
# TODO: remove noqa once implementation starts.

import logging
from typing import override
from abc import abstractmethod

from cosmoprimo import Cosmology

from acm.catalogs.backends import SnapshotBackend
from acm.catalogs.dataclasses import Tracer
from acm.catalogs.factories import BaseCatalogFactory
from acm.catalogs.products import CutskyCatalog

logger = logging.getLogger(__name__)

logger.warning("This module is in development and might return unexpected results!")

class BaseCutskyFactory(BaseCatalogFactory):
    """Factory for creating cutsky-based catalogs."""

    def __init__(
        self,
        backend: str | SnapshotBackend,
        catalog_class: type[CutskyCatalog],
        cosmo: Cosmology,
        cosmo_fid: Cosmology | None = None,
        **kwargs,
    ) -> None:
        super().__init__(backend, catalog_class, cosmo, cosmo_fid, **kwargs)
        # Type hints
        self.backend: SnapshotBackend
        self.catalog_class: type[CutskyCatalog]
        self._catalogs: dict[tuple[float, float], CutskyCatalog]

    @abstractmethod
    def make_catalogs(
        self,
        redshifts: list[float],
        redshift_ranges: list[tuple[float, float]],
        tracers: list[Tracer] | dict[float, list[Tracer]],
        **kwargs,
    ) -> None:
        """
        Load dark matter snapshots and populate galaxy catalogs for each redshift.

        Parameters
        ----------
        redshifts : list[float]
            List of redshifts at which to load dark matter snapshots.
        redshift_ranges : list[tuple[float, float]]
            List of redshift ranges to index each catalog. Must correspond to the redshifts list.
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
    def get_catalog(self, redshift_range: tuple[float, float]) -> CutskyCatalog:
        """
        Retrieve the galaxy catalog at a given redshift range.

        Parameters
        ----------
        redshift_range : tuple[float, float]
            The redshift range of the desired catalog.
        """
        ...


class CutskyCatalogFactory(BaseCutskyFactory):
    """Factory for creating a single cutsky-based galaxy catalog spanning a redshift range."""

    @property
    def redshift_range(self) -> tuple[float, float]:
        """Redshift range covered by the catalog."""
        if len(self._catalogs) != 1:
            raise ValueError(
                "Multiple catalogs found, cannot determine redshift range. Use get_catalog with specific redshift range instead."
            )
        return list(self._catalogs)[0]

    @override
    def make_catalogs(
        self,
        redshifts: list[float],
        redshift_ranges: list[tuple[float, float]],
        tracers: list[Tracer] | dict[float, list[Tracer]],
        **kwargs,
    ) -> None:
        for i, (zsnap, zranges) in enumerate(
            zip(redshifts, redshift_ranges, strict=True)
        ):
            # TODO: Get boxes trough backend
            # TODO: Populate snapshot catalogs with tracers through backend - return SnapshotCatalogs
            # TODO: Convert snapshot catalogs to cutsky catalogs through cutsky_to_box utility function
            # NOTE: this utility function should handle box replication and periodic wrapping,
            # and will depend on the cosmology for distance-redshift conversion and on the redshift range for angle values.
            pass

        # TODO: assemble cutsky catalogs into a single catalog spanning the full redshift range, store in self._catalogs with key zranges

        # NOTE: do not match nbar, or apply masks at this step, those should be available in the galaxy catalog class instead :)

    def get_catalog(self, redshift_range: tuple[float, float]) -> CutskyCatalog:
        """
        Retrieve the galaxy catalog at a given redshift range.

        Parameters
        ----------
        redshift_range : tuple[float, float]
            The redshift range of the desired catalog.
        """
        return self._catalogs[redshift_range]


# NOTE: maybe move box_to_cutsky and cutsky_to_box utilities here, or to a separate geometry_utils module?
