"""Concrete catalog factories for cutsky-based pipelines."""
# ruff: noqa
# TODO: remove noqa once implementation starts.

import logging
from typing import override, Callable
from abc import abstractmethod

import numpy as np
import pandas as pd
from pathlib import Path

from cosmoprimo import Cosmology

from acm.catalogs.backends import SnapshotBackend
from acm.catalogs.dataclasses import Tracer, Transform
from acm.catalogs.factories import BaseCatalogFactory, SnapshotCatalogFactory
from acm.catalogs.products import CutskyCatalog, SnapshotCatalog
from acm.catalogs.geometry import minmax_xyz_desi

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
        boxpad : float = 1000,
        **kwargs,
    ) -> None:
        super().__init__(backend, catalog_class, cosmo, cosmo_fid, **kwargs)
        # Type hints
        self.backend: SnapshotBackend
        self.catalog_class: type[CutskyCatalog]
        self._catalogs: dict[tuple[float, float], SnapshotCatalog]
        self.box_catalog_class : type[SnapshotCatalog],
        # Type checks
        if not isinstance(self.backend, SnapshotBackend):
            backend_type = type(self.backend)
            error_message = f'The provided backend must be an instance of SnapshotBackend. The received type was {backend_type}'
            raise TypeError(error_message)

        self.boxpad = boxpad  # Mpc/h

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

    def save(self, path: str | Path) -> None:
        """
        Save all loaded catalogs to a directory, one HDF5 file per redshift.

        Catalogs are saved with filenames like "catalog_z0.500.h5"
        using the save method of the catalog class.

        Parameters
        ----------
        path : str | Path
            Output directory. Created if it does not exist.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for zranges, catalog in self._catalogs.items():
            zlow, zhigh = zranges
            catalog.save(path / f"catalog_z{zlow:.3f}-{zhigh:.3f}.h5")
        logger.info(f"Saved {len(self._catalogs)} catalog(s) to {path}")

    def load_catalogs(self, path: str | Path) -> None:
        """
        Load all HDF5 catalogs from a directory into the factory.

        Searches for files with names like "catalog_z0.500.h5"
        and loads them using the load method of the catalog class.

        Parameters
        ----------
        path : str | Path
            Directory containing catalog HDF5 files.
        """
        path = Path(path)
        files = sorted(path.glob("catalog_z*_z*.h5"))
        if not files:
            raise FileNotFoundError(f"No catalog files found in {path}")
        for file in files:
            catalog = self.catalog_class.load(file, self.cosmo, self.cosmo_fid)
            self._catalogs[catalog.redshift_range] = catalog
        logger.info(f"Loaded {len(files)} catalog(s) from {path}")


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
        region: str = 'NGC',
        release: str = 'Y1',
        program : str = 'dark',
        mask_tracer : str = 'LRG',
        custom_healpix_mask : np.typing.NDArray | None = None,
        #transform: Callable[SnapshotCatalog] | None = None, # TypeError: Callable must be used as Callable[[arg, ...], result].
        **kwargs,
    ) -> None:

        # TODO: replace program, mask_tracer with something that iterates over tracers
        
        # Populate snapshot catalogs with tracers through backend - return SnapshotCatalogs
        # self.backend is already initialized so load_backend will just return self.backend
        snapshot_catalog_factory = SnapshotCatalogFactory(self.backend, self.box_catalog_class, self.cosmo, self.cosmo_fid) 
        snapshot_catalog_factory.make_catalogs(redshifts, tracers, **kwargs) 
        boxsize = self.backend.boxsize 
        
        for i, (zsnap, zranges) in enumerate(
            zip(redshifts, redshift_ranges, strict=True)
        ):

            snapshot_catalog = snapshot_catalog_factory.get_catalog(zsnap) #these are all tracers in a class
            distance_limits = self.cosmo.comoving_radial_distance(zranges)

            # Apply user defined transformations to the galaxies
            #if transform is not None:
            #    transform(snapshot_catalog)

            # replicate the box along each axis to cover more volume
            pos_min, pos_max = self.get_reference_borders(
                zranges,
                boxsize,
                region=region,
                release=release,
                program = program,
                tracer = mask_tracer,
                custom_healpix_mask=custom_healpix_mask,
            )
            shifts = self.get_box_shifts(pos_min, pos_max, boxsize)
            # doesn't make sense to have snapshotcatalog as input (multitracer) while having dataframe as output (all tracers forced into single dataframe)
            replications = self.get_box_replications(
                snapshot_catalog,
                pos_min,
                pos_max,
                shifts=shifts,
                distance_limits = (distance_limits[0], distance_limits[1]),
            )

            galaxy_catalog = self.catalog_class(
                #redshift=zsnap,
                cosmo=self.cosmo,
                cosmo_fid=self.cosmo_fid,
                #boxsize=boxsize,
                hp_res = 256,
            )
            
            for tracer, data in replications.items():
                galaxy_catalog.set_tracer_data(tracer, data)

            # assemble cutsky catalogs into a single catalog spanning the full redshift range, store in self._catalogs with key zranges
            self._catalogs[zranges] = galaxy_catalog


            # NOTE: do not match nbar, or apply masks at this step, those should be available in the galaxy catalog class instead :)



    def get_catalog(self, redshift_range: tuple[float, float] | None = None) -> CutskyCatalog:
        """
        Retrieve the galaxy catalog at a given redshift range.

        Parameters
        ----------
        redshift_range : tuple[float, float]
            The redshift range of the desired catalog.
        """
        if redshift_range  is None:
            # TODO: fix this
            cutsky_catalog = self.catalog_class(
                cosmo=self.cosmo,
                cosmo_fid=self.cosmo_fid,
                hp_res = 256,
            )
            cutsky_catalog.set_snapshot_catalogs(self._catalogs)
            return cutsky_catalog
        return self._catalogs[redshift_range]

    def get_box_shifts(
        self,
        pos_min: np.ndarray,
        pos_max: np.ndarray,
        boxsize : float,
    ) -> list:
        """
        Get the shifts that need to be applied to replicate the box along
        one or more axes of the simulation.
        Parameters
        ----------
        pos_min : np.ndarray
            1-d array, the minimum position from the reference mock.
        pos_max : np.ndarray
            1-d array, the maximum position from the reference mock.

        Returns
        -------
        list
            List of shifts to be applied to the box positions.
        """
        mappings_max = np.int32(np.ceil((pos_max - self.boxpad)/boxsize))
        mappings_min = np.int32(np.floor((pos_min + self.boxpad)/boxsize))
        shifts = []
        mappings = [np.arange(mappings_min[i],mappings_max[i]+1) for i in range(3)]
        for i in mappings[0]:
            for j in mappings[1]:
                for k in mappings[2]:
                    shifts.append([boxsize * np.array([i, j, k])])
        return shifts

    def get_pos_within_borders(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        pos_min: np.ndarray,
        pos_max: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Force the positions and velocities to be within the specified borders.

        Parameters
        ----------
        pos : np.ndarray
            Positions of the particles.
        vel : np.ndarray
            Velocities of the particles.
        pos_min : np.ndarray
            Minimum position in each dimension.
        pos_max : np.ndarray
            Maximum position in each dimension.

        Returns
        -------
        pos : np.ndarray
            Filtered positions of the particles within the specified borders.
        """
        for i in range(3):
            chosen = np.logical_and(pos[:,i] > pos_min[i], pos[:,i] < pos_max[i])
            pos = pos[chosen]
            vel = vel[chosen]
        return pos,vel
        
    def get_box_replications(
        self,
        snapshot_catalog : SnapshotCatalog,
        pos_min: np.ndarray,
        pos_max: np.ndarray,
        shifts: list | None = None,
        distance_limits: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the positions, velocities, and box centers of the replications of the simulations,
        obtained by applying the input shift values.

        Parameters
        ----------
        snapshot_catalog : SnapshotCatalog
            The box catalog object
        boxcenter : np.ndarray
            Center of the original box.
        shifts : list, optional
            List of shifts to apply to the box positions, by default None.
            If None, the default shifts are used, which replicate the box along all axes.

        Returns
        -------
        tuple
            Tuple containing:
            - new_pos: np.ndarray of positions in the replicated boxes.
            - new_vel: np.ndarray of velocities in the replicated boxes.
        """
        
        replications = {}
        
        for (tracer_name, tracer) in snapshot_catalog.tracers.items(): 
            data = snapshot_catalog.get_tracer_data(tracer_name, raw=False)
            position = data[list(snapshot_catalog.pos_columns)].to_numpy()
            velocity = data[list(snapshot_catalog.vel_columns)].to_numpy()
            
            if shifts is None:
                shifts = self.get_box_shifts()
            new_pos = []
            new_vel = []
            for shift in shifts:
                temp_pos, temp_vel = self.get_pos_within_borders(
                    position + shift,
                    velocity,
                    pos_min,
                    pos_max,
                )
                if distance_limits is not None:
                    distance = np.linalg.norm(temp_pos, axis=1)
                    dist_in_limits = (distance > distance_limits[0] - self.boxpad) * (distance < distance_limits[1] + self.boxpad)
                    temp_pos = temp_pos[dist_in_limits]
                    temp_vel = temp_vel[dist_in_limits]
                    
                new_pos.append(temp_pos)
                new_vel.append(temp_vel)
            new_pos = np.concatenate(new_pos)
            new_vel = np.concatenate(new_vel)
            
            tracer_replication = np.hstack((new_pos, new_vel))
            tracer_replication = pd.DataFrame(tracer_replication, columns=list(snapshot_catalog.pos_columns) + list(snapshot_catalog.vel_columns))
            replications[tracer] = tracer_replication
        return replications


    def get_reference_borders(
        self,
        zranges: list,
        boxsize: float,
        region: str = 'NGC',
        release: str = 'Y1',
        program : str = 'dark',
        tracer = 'LRG',
        custom_healpix_mask : np.typing.NDArray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the minimum and maximum cartesian coordinates from a reference galaxy catalog
        to restrict the volume spanned by the replicated HOD boxes. This avoids wasting
        memory by keeping particles that fall outside of the survey volume.

        Parameters
        ----------
        zranges : list
            List of redshift ranges for which to get the borders.
        region : str, optional
            The DESI photometric region, e.g., 'NGC', by default 'NGC'.
        release : str, optional
            The DESI data release, e.g., 'Y1', by default 'Y1'.
        custom_xyz_file : str
            If not None, a custom file is read for the positions of the tracers that define
            the survey volume bounds

        Returns
        -------
        tuple
            A tuple containing the minimum and maximum positions in each dimension (x, y, z).
            If boxpad > 1, add a padding value in Mpc/h. If boxpad <= 1, add it as a fracton
            of the base box size.
        """
        boxpad = self.boxpad
        if boxpad <= 0:
            raise ValueError(f"boxpad must be positive, got {boxpad}")
        pos_min, pos_max = minmax_xyz_desi(
            zranges,
            self.cosmo,
            region=region,
            release=release,
            program=program,
            tracer=tracer,
            custom_healpix_mask=custom_healpix_mask,
        ) 
        if boxpad > 1:
            return pos_min - boxpad, pos_max + boxpad
        else:
            return pos_min - boxpad * boxsize, pos_max + boxpad * boxsize