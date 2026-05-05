import logging

from cosmoprimo import Cosmology
from pandas import DataFrame

from acm.catalogs.dataclasses import Tracer

logger = logging.getLogger(__name__)


class BaseGalaxyCatalog:
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
        self._data: dict[str, DataFrame] = {}

    def __repr__(self) -> str:
        """Provide a string representation of the galaxy catalog, including tracer information."""
        return f"{self.__class__.__name__}(tracers={list(self.tracers.keys())})"

    def register_tracer(self, tracer: Tracer) -> None:
        """Register a tracer in the catalog."""
        if tracer.name in self.tracers:
            logger.warning(f"Tracer '{tracer.name}' already exists.")
        self.tracers[tracer.name] = tracer

    def set_tracer_data(self, tracer: Tracer, data: DataFrame) -> None:
        """Set the galaxy data for a given tracer."""
        self.register_tracer(tracer)  # Ensure tracer is registered before setting data
        self._data[tracer.name] = data

    def get_tracer_data(self, tracer_name: str) -> DataFrame:
        """Get the galaxy data for a given tracer."""
        if tracer_name not in self._data:
            raise KeyError(f"No data loaded for tracer '{tracer_name}'.")
        return self._data[tracer_name]

    def __getitem__(self, tracer_name: str) -> DataFrame:
        """Allow direct indexing to get tracer data, e.g. catalog['ELG']."""
        return self.get_tracer_data(tracer_name)