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
    def _range_from_data(self, coord: str, is_angle: bool) -> tuple[float, float]:
        """Compute the range of a coordinate from the data. Ensure that the range is correctly computed for angular coordinates."""
        all_values = np.concatenate(
            [self._data[tracer_name][coord].values for tracer_name in self.tracers]
        )
        if is_angle:
            # TODO: Handle periodicity to prevent returning the opposite of the true range, e.g. (40, 300) instead of (300, 40)
            # FIXME: Requires to tranform angles to negative values
            all_values = np.mod(all_values, 360)
            # Check if the range crosses the 0/360 boundary
            # if np.ptp(all_values) > 180:
            #     all_values = np.mod(all_values + 180, 360) - 180
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        return (min_val, max_val)

    @property
    def zrange(self) -> tuple[float, float]:
        """Return the redshift range of the catalog."""
        return self._range_from_data("z", is_angle=False)

    @property
    def rarange(self) -> tuple[float, float]:
        """Return the right ascension range of the catalog."""
        return self._range_from_data("ra", is_angle=True)

    @property
    def decrange(self) -> tuple[float, float]:
        """Return the declination range of the catalog."""
        return self._range_from_data("dec", is_angle=True)

    # Methods: n(z), nbar ?

    # Transforms:
    # angle wrapping: what type of range do we want to support ? e.g. do we want rarange=(300, 40) or rarange=(-60, 40) ?
    # Z to distance ?
    # Angular mask / footprint ?
    # downsampling ?


class RandomCutSkyCatalog(CutSkyCatalog):
    """A random catalog with cutsky geometry and redshift evolution."""

    @classmethod
    def from_snapshot(cls, catalog: CutSkyCatalog, seed: int | None = None) -> Self:
        """
        Create a random catalog from an existing CutSkyCatalog.

        Inherits cosmology and tracers from the source catalog,
        replacing all position data with uniform random draws.

        Parameters
        ----------
        catalog : CutSkyCatalog
            Source catalog to copy metadata and tracer counts from.
        seed : int | None
            Random seed for reproducibility.
        """
        random_catalog = cls(
            cosmo=catalog.cosmo,
            cosmo_fid=catalog.cosmo_fid,
        )
        for tracer_name, tracer in catalog.tracers.items():
            n_gal = len(catalog._data[tracer_name])
            random_catalog.set_tracer_data(
                tracer,
                cls._random_positions(
                    n_gal,
                    rarange=catalog.rarange,
                    decrange=catalog.decrange,
                    zrange=catalog.zrange,
                    seed=seed,
                ),
            )
        return random_catalog

    @staticmethod
    def _random_positions(
        n_gal: int,
        rarange: tuple[float, float],
        decrange: tuple[float, float],
        zrange: tuple[float, float],
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate a pandas DataFrame of uniform random positions within the box.

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
        seed : int | None
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed=seed)
        return pd.DataFrame(
            {
                "ra": rng.uniform(*rarange, size=n_gal),
                "dec": rng.uniform(*decrange, size=n_gal),
                "z": rng.uniform(*zrange, size=n_gal),  # Example redshift range
            }
        )
