from  abc import ABC

import numpy as np

# cosmodesi/acm
import mockfactory
from mockfactory.desi import is_in_desi_footprint
from cosmoprimo.fiducial import AbacusSummit
from .box import BoxHOD
from .footprint import *
import fitsio
import logging
import warnings
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

from acm.utils.paths import get_Abacus_dirs
LRG_Abacus_DM = get_Abacus_dirs(tracer='LRG', simtype='box')

#TODO : add docstrings !


class BaseCutskyCatalog(ABC):
    def __init__(self):
        pass

    def apply_angular_mask(self, region='N+SNGC', release='Y1', npasses=None, program='dark'):
        """
        Applies the angular mask to the cutsky catalog based on the specified region.

        Parameters
        ----------
        region : str
            The region to apply the angular mask. Options include 'N', 'DN', 'DS', 'N+SNGC', 'SNGC', 'SSGC', 'DES', 'NGC', 'SGC'.
        release : str
            The release of the data, e.g., 'Y1'.
        npasses : int, optional
            The number of passes for the mask. If None, defaults to 1.
        program : str
            The program to use for the mask, e.g., 'dark'.

        Returns
        -------
        None
            The cutsky catalog is modified in place.
        """
        self.logger.info('Applying angular mask.')
        is_in_desi = is_in_desi_footprint(self.catalog['RA'], self.catalog['DEC'], release=release, program=program, npasses=npasses)
        self.catalog['HPX'], is_in_photo = self.is_in_photometric_region(self.catalog['RA'], self.catalog['DEC'], region=region)
        for key in self.catalog.keys():
            self.catalog[key] = self.catalog[key][is_in_desi & is_in_photo]

    def apply_radial_mask(self, nz_filename: str, shape_only: bool = False):
        """
        Applies the radial selection function to a cutsky catalog based on 
        an input n(z) file (number desity as a function of redshift).

        Parameters
        ----------
        nz_filename : str
            Path to the n(z) file containing the target number density. Columns
            (1, 2, 3) are zbin_min, zbin_max, and target_nz respectively.
        shape_only : bool, optional
            If True, match only the shape of the n(z), disregarding the amplitude.

        Returns
        -------
        None
            The cutsky catalog is modified in place.
        """
        from scipy.interpolate import InterpolatedUnivariateSpline
        self.logger.info('Applying radial mask.')
        zbin_min, zbin_max, target_nz = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
        zbin_mid = (zbin_min + zbin_max) / 2
        nz_spline = InterpolatedUnivariateSpline(zbin_mid, target_nz, k=1, ext=3)
        ratio = target_nz / self.raw_nbar
        if shape_only:
            max_ratio = np.max(ratio[~np.isinf(ratio)])
            ratio /= max_ratio
        ratio_spline = InterpolatedUnivariateSpline(zbin_mid, ratio, k=1, ext=3)
        # use the spline to get the number density at the redshift of every galaxy
        # then assign a random number to each and compare it to the ratio to determine
        # if the galaxy should be kept or not
        data_nz = nz_spline(self.catalog['Z'])
        select_mask = np.random.uniform(size=len(self.catalog['Z'])) < ratio_spline(self.catalog['Z'])
        for key in self.catalog.keys():
            self.catalog[key] = self.catalog[key][select_mask]
        self.catalog['NZ'] = data_nz[select_mask]
        if shape_only:
            self.catalog['NZ'] /= max_ratio


    def compute_nz(self, zedges=None):
        """
        Computes the comoving number density n(z) of the cutsky catalog.

        Parameters
        ----------
        zedges : np.ndarray, optional
            The edges of the redshift bins. If None, defaults to the range of redshifts in the catalog.

        Returns
        -------
        zbin_mid : np.ndarray
            Midpoints of the redshift bins.
        data_nz : np.ndarray
            Number density of galaxies in each redshift bin.
        """
        if zedges is None:
            zedges = np.linspace(self.catalog['Z'].min(), self.catalog['Z'].max(), 100)
        data_nz, _ = np.histogram(self.catalog['Z'], bins=zedges)
        zbin_mid = (zedges[:-1] + zedges[1:]) / 2
        return zbin_mid, data_nz / np.diff(zedges)

    def is_in_photometric_region(self, ra, dec, region, rank=0):
        """DN=NNGC and DS = SNGC"""
        region = region.upper()
        assert region in ['N', 'DN', 'DS', 'N+SNGC', 'SNGC', 'SSGC', 'DES', 'NGC', 'SGC']

        DR9Footprint = None
        try:
            from regressis import DR9Footprint
        except ImportError:
            if rank == 0: self.logger.info('Regressis not found, falling back to RA/Dec cuts')

        if DR9Footprint is None:
            mask = np.ones_like(ra, dtype='?')
            if region == 'DES':
                raise ValueError('Do not know DES cuts, install regressis')
            dec_cut = 32.375
            if region == 'N':
                mask &= dec > dec_cut
            else:  # S
                mask &= dec < dec_cut
            if region in ['DN', 'DS', 'SNGC', 'SSGC']:
                mask_ra = (ra > 100 - dec)
                mask_ra &= (ra < 280 + dec)
                if region in ['DN', 'SNGC']:
                    mask &= mask_ra
                else:  # DS
                    mask &= dec > -25
                    mask &= ~mask_ra
            return np.nan * np.ones(ra.size), mask
        else:
            from regressis.utils import build_healpix_map
            # Precompute the healpix number
            nside = 256
            _, pixels = build_healpix_map(nside, ra, dec, return_pix=True)

            # Load DR9 footprint and create corresponding mask
            dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=False,
                                         mask_around_des=False, cut_desi=False,
                                         verbose=(rank == 0))
            convert_dict = {
                'N': 'north',
                'DN': 'south_mid_ngc',
                'N+SNGC': 'ngc', 'SNGC': 'south_mid_ngc',
                'DS': 'south_mid_sgc',
                'SSGC': 'south_mid_sgc',
                'DES': 'des',
                'NGC': 'ngc',
                'SGC': 'south_mid_sgc',
            }
            return pixels, dr9_footprint(convert_dict[region])[pixels]

        

class CutskyHOD(BaseCutskyCatalog):
    """
    Patch together cubic boxes to form a pseudo-lightcone.
    """
    def __init__(
            self, varied_params, config_file: str = None, cosmo_idx: int = 0, 
            phase_idx: int = 0, zranges: list[list] = [[0.4, 0.6]], 
            snapshots: list = [0.5], DM_DICT: dict = LRG_Abacus_DM,
            load_existing_hod: bool = False):
        self.logger = logging.getLogger('CutskyHOD')
        self.load_existing_hod = load_existing_hod
        self.varied_params = varied_params
        self.cosmo_idx = cosmo_idx
        self.phase_idx = phase_idx
        self.sim_type = 'base'
        if len(zranges) != len(snapshots):
            raise ValueError('Number of redshift ranges must match number of snapshots.')
        self.zranges = zranges
        self.snapshots = snapshots
        self.boxsize_snapshot = 2000  # Mpc/h
        self.boxpad = 1000  # Mpc/h
        self.boxcenter = np.array([0, 0, 0])  # Mpc/h
        if self.load_existing_hod:
            self.cosmo = AbacusSummit(self.cosmo_idx)
            self.logger.info('Load existing hod instead of generating new ones.')
        else:
            self.setup_hod(DM_DICT=DM_DICT)
        self.keys_cutsky = ['RA', 'DEC', 'Z', 'RSDPosition', 'Distance', 'Position']

    def setup_hod(self, DM_DICT: dict):
        """
        Initialize AbacusHOD objects for each snapshot.

        Parameters
        ----------
        DM_DICT : dict
            Dictionary containing the DM fields for the HOD sampling.
        """
        self.balls = []
        for zsnap in self.snapshots:
            ball = BoxHOD(varied_params=self.varied_params,
                          DM_DICT=DM_DICT, sim_type=self.sim_type,
                          redshift=zsnap, cosmo_idx=self.cosmo_idx,
                          phase_idx=self.phase_idx)
            self.balls += [ball]
        self.cosmo = AbacusSummit(self.cosmo_idx)

    def _sample_hod(self, ball, hod_params: dict, nthreads: int = 1, seed: float = 0,
        target_nbar: float = None):
        """
        Sample HOD galaxies from the given ball object using the provided HOD parameters.
        
        Parameters
        ----------
        ball : BoxHOD
            The BoxHOD object to sample from.
        hod_params : dict
            Dictionary containing the HOD parameters to use for sampling.
        nthreads : int, optional
            Number of threads to use for sampling, by default 1.
        seed : float, optional
            Random seed for reproducibility, by default 0.
        target_nbar : float, optional
            Number density to downsample the HOD catalog to, in (Mpc/h)^-3.
        Returns
        -------
        tuple
            Tuple containing positions and velocities of the sampled galaxies.
        """
        hod_dict = ball.run(hod_params, seed=seed, nthreads=nthreads,
                            tracer_density_mean=target_nbar)['LRG']
        pos = np.c_[hod_dict['X'], hod_dict['Y'], hod_dict['Z']]
        vel = np.c_[hod_dict['VX'], hod_dict['VY'], hod_dict['VZ']]
        return pos.astype(np.float32), vel.astype(np.float32)

    def load_hod(self, mock_path=None):
        """
        Load an existing HOD catalog from a file to speed up debugging.

        Parameters
        ----------
        mock_path : str, optional
            Path to the HOD catalog file. If None, uses a default path.
        Returns
        -------
        tuple
            Tuple containing positions and velocities of the galaxies.
        """
        if mock_path is None:
            base_path = '/global/cfs/projectdirs/desi/cosmosim/SecondGenMocks/CubicBox/LRG/z0.500/AbacusSummit_base_c000_ph000'
            mock_path = base_path +'/LRG_real_space.fits'
            # data_dir = '/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z0.5/yuan23_prior/c000_ph000/seed0'
            # data_fn = Path(data_dir) / 'hod030_raw.fits'
        data  = fitsio.read(mock_path)
        pos = np.vstack([data['x'],data['y'],data['z']]).T
        vel = np.vstack([data['vx'],data['vy'],data['vz']]).T
        return pos.astype(np.float32), vel.astype(np.float32)

    def init_cutsky(self):
        """Initialize the catalog dictionary."""
        cutsky = {}
        for key in self.keys_cutsky:
            cutsky[key] = []
        return cutsky

    def sample_hod(
            self, hod_params: dict, nthreads: int = 1, seed: float = 0, 
            box_margin: float = 300, existing_hod_path: str = None, 
            region: str ='NGC', release: str ='Y1', program: str ='dark'):
        """
        Sample HOD galaxies from the snapshots and build a cutsky catalog.
        This does not yet apply the angular or radial masks, which should be done
        separately after this method.
        """
        self.catalog = self.init_cutsky()

        # construct one redshift shell at a time from the snapshots
        for i, (zsnap, zranges) in enumerate(zip(self.snapshots, self.zranges)):
            self.logger.info(f'Processing snapshot at z = {zsnap} for redshift range {zranges}')
            target_nbar = self.get_target_nbar(zmin=zranges[0], zmax=zranges[1])
            if self.load_existing_hod:
                box_positions, box_velocities = self.load_hod(mock_path=existing_hod_path)
            else:
                ball  = self.balls[i]
                box_positions, box_velocities = self._sample_hod(ball, hod_params, nthreads=nthreads,
                                                                 target_nbar=target_nbar, seed=seed)
            self.raw_nbar = len(box_positions) / (self.boxsize_snapshot**3)
            # replicate the box along each axis to cover more volume
            pos_min, pos_max = self.get_reference_borders(zranges, region=region, release=release)
            shifts = self.get_box_shifts(pos_min, pos_max)
            box_positions, box_velocities = self.get_box_replications(box_positions, box_velocities,
                                                                      pos_min, pos_max, target_nbar,
                                                                      shifts=shifts)
            box = mockfactory.BoxCatalog(data={'Position': box_positions, 'Velocity': box_velocities},
                                              position='Position', velocity='Velocity',
                                              boxsize=pos_max-pos_min, boxcenter=(pos_max+pos_min)/2,)
            cutsky_shell = self.box_to_cutsky(box=box, zmin=zranges[0], zmax=zranges[1], 
                                          zrsd=zsnap, apply_rsd=True)
            for key in self.keys_cutsky:
                self.catalog[key].extend(cutsky_shell[key])
            del box_positions, box_velocities, box, cutsky_shell
        for key in self.keys_cutsky:
            self.catalog[key] = np.array(self.catalog[key])
        return self.catalog

    def get_box_shifts(self, pos_min, pos_max):
        """
        Get the shifts that need to be applied to replicate the box along
        one or more axes of the simulation.
        Parameters
        ----------
        pos_min, pos_max: both should be 1-d array, the minimum and maximum of postion from the reference mock.
        Returns
        -------
        list
            List of shifts to be applied to the box positions.
        """
        mappings_max = np.int32(np.ceil((pos_max - self.boxpad)/self.boxsize_snapshot))
        mappings_min = np.int32(np.floor((pos_min + self.boxpad)/self.boxsize_snapshot))
        shifts = []
        mappings = [np.arange(mappings_min[i],mappings_max[i]+1) for i in range(3)]
        for i in mappings[0]:
            for j in mappings[1]:
                for k in mappings[2]:
                    shifts.append([self.boxsize_snapshot * np.array([i, j, k])])
        return shifts

    def get_box_replications(self, position, velocity, pos_min, pos_max, target_nbar, shifts: list = None):
        """
        Get the positions, velocities, and box centers of the replications of the simulations,
        obtained by applying the input shift values.

        Parameters
        ----------
        position : np.ndarray
            Positions of the particles in the original box.
        velocity : np.ndarray
            Velocities of the particles in the original box.
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
            - new_center: np.ndarray of centers of the replicated boxes.
        """
        if shifts is None:
            shifts = self.get_box_shifts()
        new_pos = []
        new_vel = []
        for shift in shifts:
            temp_pos,temp_vel = self.get_pos_within_borders(position + shift, velocity,
                                                            pos_min, pos_max, target_nbar)
            new_pos.append(temp_pos)
            new_vel.append(temp_vel)
        new_pos = np.concatenate(new_pos)
        new_vel = np.concatenate(new_vel)
        return new_pos, new_vel

    def box_to_cutsky(self, box, zmin: float, zmax: float, apply_rsd: bool = False, zrsd: float = None):
        """
        Convert a box catalog with cartesian positions and velocities to a cutsky catalog
        with sky coordinates and redshifts.

        Parameters
        ----------
        box : mockfactory.BoxCatalog
            The box catalog containing positions and velocities.
        zmin : float
            Minimum redshift for the cutsky.
        zmax : float
            Maximum redshift for the cutsky.
        apply_rsd : bool, optional
            Whether to apply RSD to the positions, by default False.
        zrsd : float, optional
            Redshift at which to evaluate the cosmology to apply the RSD, by default None.
        Returns
        -------
        cutsky : dict
            Dictionary containing the cutsky catalog with keys 'Distance', 'RA', 'DEC', and 'Z'.
        """
        cutsky = {}
        if apply_rsd: cutsky = self.apply_rsd(box, zrsd)
        d2r = mockfactory.DistanceToRedshift(distance=self.cosmo.comoving_radial_distance)
        pos = 'RSDPosition' if apply_rsd else 'Position'
        cutsky['Distance'], cutsky['RA'], cutsky['DEC'] = mockfactory.cartesian_to_sky(box[pos])
        cutsky['Z'] = d2r(cutsky['Distance'])
        cutsky = cutsky[(cutsky['Z'] >= zmin) & (cutsky['Z'] <= zmax)]
        return cutsky

    def apply_rsd(self, catalog, redshift: float):
        """
        Apply the redshift-space distortions (RSD) to the positions in the catalog.

        Parameters
        ----------
        catalog : mockfactory.BoxCatalog
            The catalog containing positions and velocities.
        redshift : float
            Redshift at which we evaluate the cosmology to apply the RSD.
        Returns
        -------
        catalog : mockfactory.BoxCatalog
            The catalog with RSD applied to the positions.
        """
        self.logger.info('Applying RSD.')
        a = 1 / (1 + redshift) # scale factor
        H = 100.0 * self.cosmo.efunc(redshift)  # Hubble parameter in km/s/Mpc
        rsd_factor = 1 / (a * H)  # multiply velocities by this factor to convert to Mpc/h
        catalog['RSDPosition'] = catalog.rsd_position(f=rsd_factor)
        return catalog

    def get_reference_borders(self, zranges, region='NGC', release='Y1'):
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
        Returns
        -------
        tuple
            A tuple containing the minimum and maximum positions in each dimension (x, y, z).
            If boxpad > 1, add a padding value in Mpc/h. If boxpad <= 1, add it as a fracton
            of the base box size.
        """
        boxpad = self.boxpad
        assert boxpad > 0
        pos_min, pos_max = minmax_xyz_desi(zranges, region=region, release=release, tracer='LRG') 
        if boxpad > 1:
            return pos_min - boxpad, pos_max + boxpad
        else:
            return pos_min - boxpad * self.boxsize_snapshot, pos_max + boxpad * self.boxsize_snapshot

    def get_target_nbar(self, zmin: float = 0., zmax: float = 6., nzpad=1.1, region='NGC'):
        """
        Get the maximum number density associated to a given tracer in a given redshift range
        from the observed n(z) file. This is to know what the number density of the created
        HOD should be if we later want to apply a radial mask to the cutsky catalog.

        Parameters
        ----------
        zmin : float
            Minimum redshift for the target number density.
        zmax : float
            Maximum redshift for the target number density.
        nzpad : float, optional
            Padding factor for the number density, by default 1.1. This is to leave
            a little extra room for downsampling the catalog later.
        region : str, optional
            The DESI photometric region, e.g., 'NGC', 'SGC', etc. By default 'NGC'.
        Returns
        -------
        float
            The maximum number density in the specified redshift range, multiplied by nzpad.
        """
        nz_filename = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/LRG_{region}_nz.txt'
        zbin_min, zbin_max, n_z = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
        chosen = np.logical_and(zbin_min >= zmin, zbin_max <= zmax)
        return nzpad * np.max(n_z[chosen])

    def get_pos_within_borders(self, pos, vel, pos_min, pos_max, target_nbar):
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
        target_nbar : float
            Target number density of the particles.
        Returns
        -------
        pos : np.ndarray
            Filtered positions of the particles within the specified borders.
        """
        # target_ngal = int(target_nbar*self.boxsize_snapshot**3)
        # chosen = np.random.choice(len(pos),target_ngal,replace=False)
        # pos = pos[chosen]
        # vel = vel[chosen]
        for i in range(3):
            chosen = np.logical_and(pos[:,i] > pos_min[i], pos[:,i] < pos_max[i])
            pos = pos[chosen]
            vel = vel[chosen]
        return pos,vel


class CutskyRandoms(BaseCutskyCatalog):
    """
    Class to generate randoms in a cutsky region.
    """
    def __init__(self, rarange=(0., 360.), decrange=(-90., 90.), zrange=(0.4, 0.6),
        csize=None, nbar=None, seed=None, cosmo_idx=0):
        from mockfactory import RandomCutskyCatalog
        from mockfactory.utils import radecbox_area
        self.logger = logging.getLogger('CutskyRandoms')
        self.rarange = rarange
        self.decrange = decrange
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

    def calculate_raw_nbar(self):
        """
        Calculate the comoving number density as a function of redshift, in (Mpc/h)^-3.

        Returns
        -------
        float
            The number density of the randoms.
        """
        from mockfactory.utils import radecbox_area
        area = radecbox_area(self.rarange, self.decrange)  # in square degrees
        fsky = area / 41253.0  # sky fraction covered by the randoms
        volume = 4/3 * np.pi * (self.drange[1]**3 - self.drange[0]**3) * fsky  # in (Mpc/h)^3
        return len(self.catalog['Z']) / volume
