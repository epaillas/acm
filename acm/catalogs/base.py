import logging

from .backends import DarkMatterBackend, load_backend

logger = logging.getLogger(__name__)


class GalaxyCatalog:
    """
    Base class for handling galaxy catalogs.

    It provides common functionalities for loading and processing galaxy catalogs, 
    which can be extended by child classes for specific use cases.
    The class is designed to handle a multi-tracer galaxy catalog at a fixed redshift.
    """


class GalaxyCatalogFactory:
    """
    Loads a dark matter backend and creates galaxy catalogs across multiple
    redshift snapshots.
    """

    def __init__(
        self,
        backend: str | DarkMatterBackend,
        catalog_class: type[GalaxyCatalog] = GalaxyCatalog,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        backend : str | DarkMatterBackend
            The dark matter backend to load catalogs from.
        catalog_class : type[GalaxyCatalog]
            The galaxy catalog class to instantiate. Defaults to GalaxyCatalog.
        *args
            Positional arguments to pass to the backend and catalog class constructors.
        **kwargs
            Keyword arguments to pass to the backend and catalog class constructors.
        """
        self.backend = load_backend(backend, *args, **kwargs)
        self.catalog_class = catalog_class
        self._catalogs: dict[float, GalaxyCatalog] = {}

    def load(self, redshifts: list[float], **kwargs) -> None:
        """
        Load dark matter snapshots and populate galaxy catalogs for each redshift.

        Parameters
        ----------
        redshifts : list[float]
            List of redshifts at which to load dark matter snapshots.
        **kwargs
            Extra arguments forwarded to both the backend and the catalog class.
        """
        for z in redshifts:
            logger.info(f"Loading dark matter catalog at redshift z={z:.3f}")
            dm_catalog = self.backend.get_dark_matter_catalog(redshift=z, **kwargs)

            logger.info(f"Populating galaxy catalog at redshift z={z:.3f}")
            galaxy_catalog = self.catalog_class(redshift=z, **kwargs)
            self.backend.make_galaxy_catalog(
                dm_catalog=dm_catalog,
                galaxy_catalog=galaxy_catalog,
                **kwargs,
            )
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

    @property
    def redshifts(self) -> list[float]:
        """List of redshifts for which catalogs have been loaded."""
        return list(self._catalogs.keys())

    @property
    def catalogs(self) -> dict[float, GalaxyCatalog]:
        """Dictionary of all loaded galaxy catalogs, keyed by redshift."""
        return dict(self._catalogs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"backend={self.backend.__class__.__name__}, "
            f"catalog_class={self.catalog_class.__name__}, "
            f"redshifts={self.redshifts})"
        )
