import logging
from typing import Self

import numpy as np
import pandas as pd

from acm.catalogs.products.base import BaseGalaxyCatalog

logger = logging.getLogger(__name__)


def box_to_cutsky(*args, **kwargs) -> pd.DataFrame:
    """Convert a box geometry to a cutsky geometry."""
    # Input: SnapshotCatalog (positions, cosmology & boxsize), observer position, redshift range
    # Depends on cosmology for distance-redshift conversion.
    # Depends on boxsize & observer position for angle values and eventual periodic wrapping.


class CutSkyCatalog(BaseGalaxyCatalog):
    """
    Galaxy catalog with cutsky geometry and redshift evolution.

    Expects galaxy positions in spherical coordinates (ra, dec, z) with angular coordinates
    in degrees and redshift as a dimensionless quantity.
    """

    pos_columns = ("ra", "dec", "z")

    # TODO: are extra properties needed to define that catalog at init ? Must repr reflect that ?

    # Properties: what is needed ?
    # TODO: also make this tracer-aware
    def _range_from_data(self, coord: str, periodic_wrap: float | None = None) -> tuple[float, float]:
        """Compute the range of a coordinate from the data."""
        all_values = np.concatenate(
            [self._data[tracer_name][coord].values for tracer_name in self.tracers]
        )
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        cout = [min_val, max_val]
        if periodic_wrap is not None:
            # Accounting for periodicity, with wrap-around if the range crosses the periodic boundary
            cout = np.mod(cout, periodic_wrap).tolist()
        return tuple(cout)

    @property
    def zrange(self) -> tuple[float, float]:
        """Return the redshift range of the catalog."""
        return self._range_from_data("z")

    @property
    def rarange(self) -> tuple[float, float]:
        """Return the right ascension range of the catalog."""
        return self._range_from_data("ra", periodic_wrap=360.0)

    @property
    def decrange(self) -> tuple[float, float]:
        """Return the declination range of the catalog."""
        return self._range_from_data("dec")

    # Methods: n(z), nbar ?

    # Transforms:
    # angle wrapping: what type of range do we want to support ? e.g. do we want rarange=(300, 40) or rarange=(-60, 40) ?
    # Z to distance ?
    # Angular mask / footprint ?
    # downsampling ?


class RandomCutSkyCatalog(CutSkyCatalog):
    """A random catalog with cutsky geometry and redshift evolution."""

    @classmethod
    def from_snapshot(
        cls, 
        catalog: CutSkyCatalog, 
        full_sky: bool = True, 
        seed: int | None = None
    ) -> Self:
        """
        Create a random catalog from an existing CutSkyCatalog.

        Inherits cosmology and tracers from the source catalog,
        replacing all position data with uniform random draws.

        Parameters
        ----------
        catalog : CutSkyCatalog
            Source catalog to copy metadata and tracer counts from.
        full_sky : bool, optional
            Whether to generate galaxies across the full sky.
            If set to False, the random positions will be generated within the same angular ranges as the source catalog.
            Default is True.
        seed : int | None
            Random seed for reproducibility.
        """
        rarange = (0, 360) if full_sky else catalog.rarange
        decrange = (-90, 90) if full_sky else catalog.decrange
        
        # Ensure independent random states for each tracer and between calls of this method (with spawn)
        ntracers = len(catalog.tracers)
        seeds = np.random.SeedSequence(seed).spawn(ntracers)
        
        random_catalog = cls(
            cosmo=catalog.cosmo,
            cosmo_fid=catalog.cosmo_fid,
        )
        for i, (tracer_name, tracer) in enumerate(catalog.tracers.items()):
            n_gal = len(catalog._data[tracer_name])
            random_catalog.set_tracer_data(
                tracer,
                cls._random_positions(
                    n_gal,
                    rarange=rarange,
                    decrange=decrange,
                    zrange=catalog.zrange,
                    seed=seeds[i],
                ),
            )
        return random_catalog

    @staticmethod
    def _random_positions(
        n_gal: int,
        rarange: tuple[float, float],
        decrange: tuple[float, float],
        zrange: tuple[float, float],
        seed: int | np.random.SeedSequence | None = None,
    ) -> pd.DataFrame:
        """
        Generate a pandas DataFrame of uniform random positions across the specified ranges for right ascension, declination, and redshift.

        Parameters
        ----------
        n_gal : int
            Number of random galaxies to generate.
        zrange : tuple[float, float]
            Redshift range (z_min, z_max) for the random galaxies.
        rarange : tuple[float, float]
            Right ascension range (ra_min, ra_max) for the random galaxies.
        decrange : tuple[float, float]
            Declination range (dec_min, dec_max) for the random galaxies.
        seed : int | np.random.SeedSequence | None
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed=seed)
        _rarange = list(rarange) # Make a mutable copy of rarange to handle potential wrapping
        if rarange[0] > rarange[1]:  # Handle cases where the range wraps around 360 degrees
            _rarange[0] -= 360
        
        ra = rng.uniform(*_rarange, size=n_gal) % 360  # Wrap RA to [0, 360)
        u = np.sin(np.radians(decrange))
        dec = np.degrees(np.arcsin(rng.uniform(u[0], u[1], size=n_gal)))  # Uniform in sin(dec) for proper area weighting
        z = rng.uniform(*zrange, size=n_gal)  # Uniform redshift distribution for simplicity; can be modified to match n(z) if needed
        
        return pd.DataFrame({"ra": ra, "dec": dec, "z": z})
