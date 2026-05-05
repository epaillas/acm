from cosmoprimo import Cosmology

from acm.catalogs.base import BaseGalaxyCatalog


class SnapshotCatalog(BaseGalaxyCatalog):
    """Snapshot-based galaxy catalog at a fixed redshift."""

    def __init__(
        self,
        redshift: float,
        cosmo: Cosmology,
        cosmo_fid: Cosmology,
    ) -> None:
        """
        Initialize the galaxy catalog with the given redshift and cosmologies.

        Extends BaseGalaxyCatalog by adding a redshift attribute and related cosmological properties.

        Parameters
        ----------
        redshift : float
            Redshift of the snapshot.
        cosmo : cosmoprimo.Cosmology, optional
            The cosmology for the simulation.
        cosmo_fid : cosmoprimo.Cosmology, optional
            The fiducial cosmology.
        """
        super().__init__(cosmo, cosmo_fid)
        self.redshift = redshift

    def __repr__(self) -> str:
        """Provide a string representation of the galaxy catalog, including redshift and tracer information."""
        return (
            f"{self.__class__.__name__}("
            f"redshift={self.redshift}, "
            f"tracers={list(self.tracers.keys())})"
        )
    
    @property
    def az(self) -> float:
        """Scale factor at this snapshot's redshift."""
        return 1.0 / (1.0 + self.redshift)

    @property
    def hubble(self) -> float:
        """H(z) in km/s/(Mpc/h) for the simulation cosmology."""
        return 100.0 * self.cosmo.efunc(self.redshift)

    @property
    def hubble_fid(self) -> float:
        """H(z) in km/s/(Mpc/h) for the fiducial cosmology."""
        return 100.0 * self.cosmo_fid.efunc(self.redshift)

    @property
    def q_par(self) -> float:
        """AP parallel scaling factor."""
        return self.hubble_fid / self.hubble

    @property
    def q_perp(self) -> float:
        """AP perpendicular scaling factor."""
        return self.cosmo.angular_diameter_distance(
            self.redshift
        ) / self.cosmo_fid.angular_diameter_distance(self.redshift)