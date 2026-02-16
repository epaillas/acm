from  abc import ABC
import os
import yaml
import numpy as np
from pathlib import Path

# cosmodesi/acm
import mockfactory
from cosmoprimo.fiducial import AbacusSummit
from .cutsky import CutskyHOD, CutskyRandoms

from mockfactory import RandomCutskyCatalog
from mockfactory.utils import radecbox_area

from scipy.interpolate import InterpolatedUnivariateSpline
import logging
import warnings
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class BaseLightconeCatalog(ABC):
    """
    Base class for mock lightcone catalogs.
    """

    def __init__(self):
        pass

    """
    def apply_angular_mask(self):
        # TODO: make angular mask code
        # This code will be similar to the cutsky class method apply_angular_mask, except the 
        # lightcone octant will first be rotated to providee the maximal possible overlap with
        # angular mask prior to downsampling. Prior to this implmentation, users can use the 
        # fullsky version of the lightcone mock and the inherited apply_angular_mask from 
        # the cutsky class
        pass
    """
    
    def apply_radial_mask(self, nz_filename: str, shape_only: bool = False, full_sky: bool = False, ):
        """
        Applies the radial selection function to a lightcone catalog based on 
        an input n(z) file (number desity as a function of redshift).

        Parameters
        ----------
        nz_filename : str
            Path to the n(z) file containing the target number density. Columns
            (1, 2, 3) are zbin_min, zbin_max, and target_nz respectively.
        shape_only : bool, optional
            If True, match only the shape of the n(z), disregarding the amplitude.
        full_sky: bool
            If True, the survey volunme is scaled to the full sky rather than an octant
        Returns
        -------
        None
            The lightcone catalog is modified in place.
        """
        data_nbar = self.get_data_nbar(self.catalog, full_sky)
        self.logger.info(f'Raw data nbar: {data_nbar}' )
        
        self.logger.info('Applying radial mask.')
        #nz_filename = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/{tracer}_NGC_nz.txt'

        zmin_data = self.catalog['Z'].min()
        zmax_data = self.catalog['Z'].max()

        # read n(z) file
        zbin_min, zbin_max, target_nz = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
        zbin_mid = (zbin_min + zbin_max) / 2

        nz_spline = InterpolatedUnivariateSpline(zbin_mid, target_nz, k=1, ext=3)
        
        #impose lightcone redshift limits on zbins
        select_zbins = (zbin_max > zmin_data) * (zbin_min < zmax_data)
        zbin_min = zbin_min[select_zbins]
        zbin_max = zbin_max[select_zbins]
        zbin_min[0] = zmin_data
        zbin_max[-1] = zmax_data
        zbin_mid = (zbin_min + zbin_max) / 2
        target_nz = nz_spline(zbin_mid)

        #calculate volumes of shells
        zedges = np.insert(zbin_max, 0, zbin_min[0])
        dbin_max = self.cosmo.comoving_radial_distance(zbin_max)
        dedges =  np.insert(dbin_max, 0, self.cosmo.comoving_radial_distance(zbin_min[0]))
        volume = 4/3 * np.pi * (dedges[1:]**3 - dedges[:-1]**3) / 8 
        
        # Handle shells exterior to the Abacus boxes
        if np.any(dbin_max > self.boxsize):
    
            exterior_indices = np.where(dbin_max > self.boxsize)[0]
            exterior_shells = (dbin_max[exterior_indices] + dedges[exterior_indices]) / 2

            # fraction of each shell inside the simulation volume
            filling_fractions = self.shell_filling_fraction(exterior_shells)
            
            # Adjust volumes using volume filling fractions
            volume[exterior_indices] = volume[exterior_indices] * filling_fractions

        # calculate downsampling ratio
        data_nz = np.histogram(self.catalog['Z'], bins=zedges)[0] / volume
        ratio = target_nz / data_nz
        if shape_only:
            ratio /= np.max(ratio[~np.isinf(ratio)])
        ratio_spline = InterpolatedUnivariateSpline(zbin_mid, ratio, k=1, ext=3)

        # use the spline to get the number density at the redshift of every galaxy
        # then assign a random number to each and compare it to the ratio to determine
        # if the galaxy should be kept or not
        data_nz = nz_spline(self.catalog['Z'])
        select_mask = np.random.uniform(size=len(self.catalog['Z'])) < ratio_spline(self.catalog['Z'])
        for key in self.catalog.keys():
            self.catalog[key] = self.catalog[key][select_mask]
        self.catalog['NZ'] = data_nz[select_mask]
        self.logger.info(f'Downsampled data nbar: {self.get_data_nbar(self.catalog, full_sky)}' )
    
    def get_data_nbar(self, data, full_sky: bool = False):
        """
        Compute the number density of the data catalog, which is defined
        by an octant of the spherical shell delimited by the redshift cuts.
        """
        
        dmin, dmax = self.cosmo.comoving_radial_distance(self.zrange)
        
        if self.sim_type=='base':
            # Abacus base
            # Monte Carlo sampling of Abacus lightcone volume
            nsamples_per_box=self.monte_carlo_sampling_count
            samples_x = np.random.uniform(low=0, high=2000, size = (3*nsamples_per_box))
            samples_y = np.concatenate([np.random.uniform(low=0, high=2000, size = (nsamples_per_box)), 
                                        np.random.uniform(low=2000, high=4000, size = (nsamples_per_box)), 
                                        np.random.uniform(low=0, high=2000, size = (nsamples_per_box))])
            samples_z = np.concatenate([np.random.uniform(low=0, high=2000, size = (2*nsamples_per_box)),  
                                        np.random.uniform(low=2000, high=4000, size = (nsamples_per_box))])
            samples = np.vstack([samples_x, 
                                 samples_y, 
                                 samples_z] )
            norm = np.linalg.norm(samples, axis=0)
            num_in_lightcone = np.sum((norm>dmin) * (norm<dmax))
            volume = 2000**3 * num_in_lightcone/nsamples_per_box
            correction = 8 if full_sky else 1  # multiply by 8 if only using the full sky
            nbar = len(data['Z']) / (volume * correction)
            
        else:
            # Abacus huge (assumes that shells are entirely contained within the survey box)
            # Monte carlo sampling would be more accurate if this is not the case
            # TODO: not yet supported with BoxHOD
            volume = 4/3 * np.pi * (dmax**3 - dmin**3)
            correction = 1 if full_sky else 8  # divide by 8 if only using a sky octant
            nbar = len(data['Z']) / (volume / correction)
        return nbar

    def shell_filling_fraction(self, shells):
    
        # Muller-Marsaglia octant sampling
        # evenly distribute points along an octant of a unit sphere
        num_samples = self.monte_carlo_sampling_count
        sample_points = np.abs(np.random.normal(0,1, (3,num_samples)))
        sample_points /= np.linalg.norm(sample_points, axis=0)
    
        # fraction of each shell inside the simulation volume
        vol_fractions = []
    
        if self.sim_type=='base':
            # Abacus base
            for shell_radius in shells:
                # count how many points in the shell are in the L-shaped stack of periodic boxes
                vol_fractions.append(
                    np.sum((sample_points[0]<self.boxsize/shell_radius)*\
                           (sample_points[1]<2*self.boxsize/shell_radius)*\
                           (sample_points[2]<2*self.boxsize/shell_radius)*\
                           ((sample_points[1]<=self.boxsize/shell_radius)+(sample_points[2]<=self.boxsize/shell_radius))\
                          )/len(sample_points[0])
                )
        else:
            # Abacus huge (TODO: not yet supported with BoxHOD)
            for shell_radius in shells:
                # count how many points in the shell are in the box
                vol_fractions.append(
                    np.sum((sample_points[0]<self.boxsize/shell_radius)*\
                           (sample_points[1]<self.boxsize/shell_radius)*\
                           (sample_points[2]<self.boxsize/shell_radius)\
                          )/len(sample_points[0])
                ) 

        return np.array(vol_fractions)   
        
class LightconeHOD(CutskyHOD, BaseLightconeCatalog):
    def __init__(
        self, varied_params, config_file: str = None, cosmo_idx: int = 0, 
        phase_idx: int = 0, zrange: list = [0.4, 0.8],
        fixed_redshift_snapshot = None,
        DM_DICT: dict = None, load_existing_hod: bool = False,
        sim_type: str = 'base', tracer: str = 'LRG',
        ):
        """
        Initialize the CutskyHOD class. This checks the HOD parameters and 
        loads the relevant simulation data that will be used to sample the
        HOD galaxies from the snapshots later.

        Parameters
        ----------
        varied_params : list
            List of HOD parameters that will be varied.
        config_file : str, optional
            Path to the configuration file for HOD parameters. If None,
            it will read the default file stored in the package.
        cosmo_idx : int, optional
            Index of the cosmology to use for the AbacusSummit simulation.
        phase_idx : int, optional
            Index of the phase to use for the AbacusSummit simulation.
        zrange : list, optional
            List containing the redshift range for which to build the lightcone catalog.
            Should contain two elements: [zmin, zmax].
        fixed_redshift_snapshot : float, optional
            Value that overrides zrange with a single redshift snapshot. Used for 
            debugging purposes. When set to None (defualt value), zrange is used instead.
        DM_DICT : dict, optional
            Dictionary containing the DM fields for the HOD sampling.
            Defaults to None, which together with the user-specified tracer maps to 
            a value in utils.paths.
        load_existing_hod : bool, optional 
            If True, load an existing HOD catalog instead of generating a new one
            (useful for quick debugging). Defaults to False.
        sim_type : str, optional
            Type of simulation to use for the HOD sampling. Defaults to 'base' (2 Gpc/h).
            Other valid options include 'huge' (7.5 Gpc/h)
            TODO: huge not yet supported with BoxHOD
        tracer : str, optional
            The type of tracer to use for the HOD sampling. Defaults to 'LRG'.
        """
        BaseLightconeCatalog.__init__(self)
        self.DM_DICT_simtype = 'lightcone'
        self.sim_geometry = 'lightcone'
        self.logger = logging.getLogger('LightconeHOD')
        self.load_existing_hod = load_existing_hod
        self.varied_params = varied_params
        self.cosmo_idx = cosmo_idx
        self.phase_idx = phase_idx
        self.sim_type = sim_type
        self.tracer = tracer
        self.zrange = zrange
        self.fixed_redshift_snapshot = fixed_redshift_snapshot
        self.boxsize = 7500 if sim_type == 'huge' else 2000
        if config_file is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
            if tracer == "LRG":
                self.config_file = Path(config_dir) /  'lightcone.yaml'
            else:
                lightcone_yaml_file = 'lightcone_' + tracer + '.yaml'
                self.config_file = Path(config_dir) / lightcone_yaml_file
        self.setup_hod(DM_DICT=DM_DICT, tracer = tracer)
        self.monte_carlo_sampling_count = 10000
        self.keys_lightcone = ['RA', 'DEC', 'Z', 'RSDPosition', 'Distance', 'Position']

    def init_lightcone(self):
        """Initialize the catalog dictionary."""
        lightcone = {}
        for key in self.keys_lightcone:
            lightcone[key] = []
        return lightcone

    @property
    def snap_redshifts(self):
        """
        Provide the full list of redshift snapshots in Abacus lightcone simulations
        
        Returns
        -------
        list
            List of redshift snapshots 
        """
        if self.tracer == 'LRG':
            return [0.400, 0.450, 0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950, 1.025, 1.100]
        elif self.tracer == 'QSO' or self.tracer == 'ELG':
            # fills shell between 0.763 and 2.627
            return [0.800,  0.875,  0.950, 1.025, 1.100, 1.175, 1.250, 1.325, 1.400, 1.475, 1.550, 1.625, 1.700, 1.850, 2.000, 2.250, 2.500]
        elif self.tracer == 'BGS':
            raise ValueError('BGS lightcone snap_redshifts not yet implemented')
        else:
            error_string = f'invalid tracer for lightcone snap_redshifts: {self.tracer}'
            raise ValueError(error_string)

    @property
    def snapshots(self):
        """
        Provide list of redshift snapshots from the Abacus lightcone simulations that span the
        user input redshift range 'zrange'

        Returns
        -------
        snaps: list
            List of redshift snapshots 
        """
        if self.fixed_redshift_snapshot is not None:
            return [self.fixed_redshift_snapshot]
        snap_min = np.abs(np.array(self.snap_redshifts) - self.zrange[0]).argmin()
        snap_max = np.abs(np.array(self.snap_redshifts) - self.zrange[1]).argmin()
        snaps = self.snap_redshifts[snap_min:snap_max+2]  # Include an extra snapshot at high-z to avoid edge effects
        self.logger.info(f'Lightcone composed of snapshots at z: {snaps}.')
        return snaps
        # return [z for z in self.snap_redshifts if z >= self.zrange[0] and z <= self.zrange[1]]

    def sample_hod(
            self, hod_params: dict, nthreads: int = 1, seed: float = 0, 
            existing_hod_path: str = None, 
            target_nz_filename: str = None,
            full_sky: bool = None):
        """
        Sample HOD galaxies from the snapshots and build a cutsky catalog.
        This does not yet apply the angular or radial masks, which should be done
        separately after this method.

        Parameters
        ----------
        hod_params : dict
            Dictionary containing the HOD parameters to use for sampling.
        nthreads : int, optional
            Number of threads to use for sampling, by default 1.
        seed : float, optional
            HOD random seed for reproducibility, by default 0.
        existing_hod_path : str, optional
            Path to an existing HOD catalog file. If provided, this will be loaded
            instead of sampling from the simulation.
        release : str, optional
            The DESI data release, e.g., 'Y1', 'Y3', or 'Y5, by default 'Y1'.
        target_nz_filename : str, optional
            Path to an n(z) filename that can be used as a reference to estimate what is the maximum
            number density that the HOD boxes require to allow for a radial mask to be applied later.
        full_sky: bool
            If True, the survey volunme is scaled to the full sky rather than an octant
        Returns
        -------
        dict
            The cutsky catalog containing positions, velocities, and other properties of the galaxies.
        """  
        self.catalog = self.init_lightcone()

        boxsize = 2*self.boxsize if self.sim_type == 'base' else self.boxsize
        
        if seed == 0: seed = None

        for i, (zsnap, ball) in enumerate(zip(self.snapshots, self.balls)):  

            self.logger.info(f'Processing snapshot at z = {zsnap}')
            
            if self.load_existing_hod:
                box_positions, box_velocities = self.load_hod(mock_path=existing_hod_path)
            else:
                ball  = self.balls[i]
                box_positions, box_velocities = self._sample_hod(ball, hod_params, nthreads=nthreads,
                                                                 target_nbar=None, seed=seed)
            #recenter box
            box_positions += 990
            #remove outbounds
            mask = (box_positions[:,0]>0)*(box_positions[:,1]>0)*(box_positions[:,2]>0)
            box_positions = box_positions[mask]
            box_velocities = box_velocities[mask]
            if full_sky:
                box_positions, box_velocities = self.make_full_sky(box_positions, box_velocities)
            
            box = mockfactory.BoxCatalog(data={'Position': box_positions, 'Velocity': box_velocities},
                                              position='Position', velocity='Velocity',
                                              boxsize=boxsize, boxcenter=[boxsize/2, boxsize/2, boxsize/2])
            lightcone_shell = self.box_to_cutsky(box=box, zmin=self.zrange[0], zmax=self.zrange[1], 
                                          zrsd=zsnap, apply_rsd=True)
            for key in self.keys_lightcone:
                self.catalog[key].extend(lightcone_shell[key])
            del box_positions, box_velocities, box, lightcone_shell
        for key in self.keys_lightcone:
            self.catalog[key] = np.array(self.catalog[key])
        return self.catalog
                        
    def make_full_sky(self, box_positions, box_velocities):
        x = box_positions[0]
        y = box_positions[1]
        z = box_positions[2]
        pos = np.c_[x, y, z]
        pos = np.concatenate(
            [pos,
             np.c_[-x, y, z],
             np.c_[x, -y, z],
                np.c_[x, y, -z],
                np.c_[-x, -y, z],
                np.c_[-x, y, -z],
                np.c_[x, -y, -z],
                np.c_[-x, -y, -z]
            ]
        )
        x = box_velocities[0]
        y = box_velocities[1]
        z = box_velocities[2]
        vel = np.c_[x, y, z]
        vel = np.concatenate(
            [pos,
             np.c_[-x, y, z],
             np.c_[x, -y, z],
                np.c_[x, y, -z],
                np.c_[-x, -y, z],
                np.c_[-x, y, -z],
                np.c_[x, -y, -z],
                np.c_[-x, -y, -z]
            ]
        )
        
        return pos.T, vel.T

    """
    def apply_angular_mask(self):
        # TODO: determine args
        # See BaseLightconeCatalog.apply_angular_mask for TODO details
        BaseLightconeCatalog.apply_angular_mask(self)
    """
    
    def apply_radial_mask(self, nz_filename: str, shape_only: bool = False, full_sky: bool = False, ):
        """
        Applies the radial selection function to a lightcone catalog based on 
        an input n(z) file (number desity as a function of redshift).

        Parameters
        ----------
        nz_filename : str
            Path to the n(z) file containing the target number density. Columns
            (1, 2, 3) are zbin_min, zbin_max, and target_nz respectively.
        shape_only : bool, optional
            If True, match only the shape of the n(z), disregarding the amplitude.
        full_sky: bool
            If True, the survey volunme is scaled to the full sky rather than an octant
        Returns
        -------
        None
            The lightcone catalog is modified in place.
        """
        BaseLightconeCatalog.apply_radial_mask(self, nz_filename, shape_only, full_sky)
        
    
class LightconeRandoms(CutskyRandoms, BaseLightconeCatalog):
    """
    Class to generate randoms in a lightcone region.
    """
    def __init__(self, full_sky = False, zrange=(0.4, 0.6),
        csize=None, nbar=None, seed=None, cosmo_idx=0, sim_type='base'):
        """
        Initialize the CutskyRandoms class. This generates randoms in a cutsky region
        that has a certain right ascension, declination and redshift range, but 
        the proper angular and radial mask needs to be applied later with the dedicated methods.
        Parameters
        ----------
        rarange : tuple, optional
            Range of right ascension in degrees, by default (0., 360.).
        decrange : tuple, optional
            Range of declination in degrees, by default (-90., 90.).
        zrange : tuple, optional
            Range of redshift, by default (0.4, 0.6).
        csize : int, optional
            Number of randoms to generate if `nbar` is not specified.
        nbar : float, optional
            Surface area density of randoms in (deg^2)^-1, by default None.
        seed : int, optional
            Random seed for reproducibility, by default None.
        cosmo_idx : int, optional
            Index of the AbacusSummit cosmology. Used for the redshift-distance relation.
        sim_type : str, optional
            Type of simulation to use for the HOD sampling. Defaults to 'base' (2 Gpc/h).
            Other valid options include 'huge' (7.5 Gpc/h)
            TODO: huge not yet supported with BoxHOD
            """
        BaseLightconeCatalog.__init__(self)
        self.logger = logging.getLogger('LightconeRandoms')
        self.rarange = (0., 360.) if full_sky else (0., 90.)
        self.decrange = (-90., 90.) if full_sky else (0., 90.)
        self.zrange = zrange
        self.nbar = nbar
        self.keys_cutsky = ['RA', 'DEC', 'Z', 'Distance', 'Position']
        self.cosmo = AbacusSummit(cosmo_idx)
        r2d = self.cosmo.comoving_radial_distance
        self.drange = (r2d(zrange[0]), r2d(zrange[1]))
        self.catalog = RandomCutskyCatalog(rarange=self.rarange, decrange=self.decrange,
                                          drange=self.drange, csize=csize, nbar=nbar, seed=seed)
        d2r = mockfactory.DistanceToRedshift(distance=self.cosmo.comoving_radial_distance)
        self.catalog['Z'] = d2r(self.catalog['Distance'])
        self.catalog = {key: self.catalog[key] for key in self.keys_cutsky}
        self.raw_nbar = self.calculate_raw_nbar()
        self.sim_type = sim_type
        self.boxsize = 7500 if self.sim_type == 'huge' else 2000
        self.monte_carlo_sampling_count = 10000

        if self.sim_type == 'base':
            xyz_pos = self.catalog['Position']
            mask = (xyz_pos[:,0] < 2000) * (xyz_pos[:,1] < 4000) * (xyz_pos[:,2] < 4000)
            mask = mask * ( (xyz_pos[:,1] < 2000) + (xyz_pos[:,2] < 2000) )

            for key in self.catalog.keys():
                self.catalog[key] = self.catalog[key][mask]
    """    
    def apply_angular_mask(self):
        # TODO: determine args
        BaseLightconeCatalog.apply_angular_mask(self)
    """
    
    def apply_radial_mask(self, nz_filename: str, shape_only: bool = False, full_sky: bool = False, ):
        """
        Applies the radial selection function to a lightcone catalog based on 
        an input n(z) file (number desity as a function of redshift).

        Parameters
        ----------
        nz_filename : str
            Path to the n(z) file containing the target number density. Columns
            (1, 2, 3) are zbin_min, zbin_max, and target_nz respectively.
        shape_only : bool, optional
            If True, match only the shape of the n(z), disregarding the amplitude.
        full_sky: bool
            If True, the survey volunme is scaled to the full sky rather than an octant
        Returns
        -------
        None
            The lightcone catalog is modified in place.
        """
        BaseLightconeCatalog.apply_radial_mask(self, nz_filename, shape_only, full_sky)