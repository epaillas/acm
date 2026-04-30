import logging

from .base import BaseCatalogFactory, GalaxyCatalog
from .dataclasses import Tracer

logger = logging.getLogger(__name__)


class GalaxyCatalogFactory(BaseCatalogFactory):
    """
    Snapshot-based factory: Load a dark matter backend
    and create galaxy catalogs across multiple redshift snapshots.
    """

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
        **kwargs
            Extra arguments forwarded to the backend.
        """
        for z in redshifts:
            snapshot_tracers = tracers if isinstance(tracers, list) else tracers[z]

            logger.info(f"Loading dark matter catalog at redshift z={z:.3f}")
            dm_catalog = self.backend.get_dark_matter_catalog(redshift=z, **kwargs)

            logger.info(
                f"Populating galaxy catalog at redshift z={z:.3f} for tracers {[t.name for t in snapshot_tracers]}"
            )
            tracer_data = self.backend.make_galaxy_catalog(
                dm_catalog=dm_catalog,
                tracers=snapshot_tracers,
                **kwargs,
            )

            galaxy_catalog = self.catalog_class(
                redshift=z, cosmo=self.cosmo, cosmo_fid=self.cosmo_fid
            )
            for tracer, data in tracer_data.items():
                galaxy_catalog.set_tracer_data(tracer, data)

            self._catalogs[z] = galaxy_catalog

    def get_catalog(self, redshift: float) -> GalaxyCatalog:
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
