import logging
from typing import Callable, Self

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
    def _range(
        self, 
        coord: str, 
        tracer: str | None = None,
        periodic_wrap: float | None = None,
    ) -> tuple[float, float]:
        """Compute the range of a coordinate from the data."""
        tracer_names = [tracer] or list(self.tracers) 
        all_values = np.concatenate(
            [self._data[tracer_name][coord].values for tracer_name in tracer_names]
        )
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        cout = [min_val, max_val]
        if periodic_wrap is not None:
            # Accounting for periodicity, with wrap-around if the range crosses the periodic boundary
            cout = np.mod(cout, periodic_wrap).tolist()
        return tuple(cout)
    
    def _zrange(self, tracer: str) -> tuple[float, float]:
        """Return the redshift range of a specific tracer."""
        return self._range("z", tracer=tracer)

    @property
    def zrange(self) -> tuple[float, float]:
        """Return the redshift range of the catalog."""
        return self._range("z")
    
    def _interpolated_nz(self, tracer: str | None = None) -> Callable[[float], float]:
        """Interpolate the number density on the full redshift range."""
        # Placeholder implementation - replace with actual interpolation logic
        return lambda z: 0.0
    
    def n(self, z: float, tracer: str | None = None) -> float:
        """Return the number density at a given redshift."""
        # Placeholder implementation - replace with actual number density calculation
        return 0.0

    # Methods: n(z), nbar ?, footprint area

    # Transforms:
    # Z to distance ?
    # Angular mask / footprint ?
    # downsampling ?


class RandomCutSkyCatalog(CutSkyCatalog):
    """A random catalog with cutsky geometry and redshift evolution."""

    @classmethod
    def from_snapshot(
        cls, catalog: CutSkyCatalog, seed: int | None = None
    ) -> Self:
        """
        Create a random catalog from an existing CutSkyCatalog.

        Inherits cosmology and tracers from the source catalog,
        replacing all position data with uniform random draws, assuming
        fullsky coverage and a uniform redshift distribution within the source catalog's redshift range.

        Parameters
        ----------
        catalog : CutSkyCatalog
            Source catalog to copy metadata and tracer counts from.
        seed : int | None
            Random seed for reproducibility.
        """
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
                    rarange=(0, 360),
                    decrange=(-90, 90),
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
            Right ascension range (ra_min, ra_max) in degrees for the random galaxies.
        decrange : tuple[float, float]
            Declination range (dec_min, dec_max) in degrees for the random galaxies.
        seed : int | np.random.SeedSequence | None
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed=seed)
        
        # Handle cases where the rarange wraps around 360 degrees
        _rarange = list(rarange)  # Mutable copy to handle potential wrapping
        if (rarange[0] > rarange[1]):  
            _rarange[0] -= 360

        ra = rng.uniform(*_rarange, size=n_gal) % 360  # Wrap RA to [0, 360)
        # Uniform distribution in sin(dec) for proper area weighting
        u = np.sin(np.radians(decrange))
        dec = np.degrees(np.arcsin(rng.uniform(u[0], u[1], size=n_gal)))  
        z = rng.uniform(*zrange, size=n_gal) # TODO: match n(z) ?

        return pd.DataFrame({"ra": ra, "dec": dec, "z": z})
