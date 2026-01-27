from abc import ABC
import logging
import fitsio
import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline

# cosmodesi/acm
import mockfactory
from mockfactory.desi import is_in_desi_footprint
from mockfactory import RandomCutskyCatalog
from mockfactory.utils import radecbox_area
from cosmoprimo.fiducial import AbacusSummit
from desitarget.targetmask import desi_mask, obsconditions

from .box import BoxHOD
from .footprint import *
from acm.utils.paths import get_Abacus_dirs

# Optional imports with better error handling
try:
    from regressis import DR9Footprint
    from regressis.utils import build_healpix_map
    HAS_REGRESSIS = True
except ImportError:
    DR9Footprint = None
    build_healpix_map = None
    HAS_REGRESSIS = False

LRG_Abacus_DM = get_Abacus_dirs(tracer='LRG', simtype='box')

# Valid DESI photometric regions
# N = North, DN = Dark North, DS = Dark South, SNGC = South NGC, SSGC = South SGC
# DES = Dark Energy Survey, NGC = North Galactic Cap, SGC = South Galactic Cap
VALID_REGIONS = ['N', 'DN', 'DS', 'N+SNGC', 'SNGC', 'SSGC', 'DES', 'NGC', 'SGC']

#TODO : add docstrings !


class BaseCutskyCatalog(ABC):
    """
    Base class for cutsky catalogs. This class provides methods to apply angular and radial masks,
    compute number density, and check if coordinates are within a specified photometric region.
    It is intended to be subclassed for specific cutsky catalog implementations.
    """
    def __init__(self):
        pass

    def apply_angular_mask(
        self,
        region: str = 'N+SNGC',
        release: str = 'Y1',
        npasses: int | None = None,
        program: str = 'dark',
        custom_mask_path: str | None = None
    ) -> None:
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
        custom_mask_path : str
            If not set to None, a custom mask file is read for applying the angular mask. The file should be in FITS format
            and should include a column named IN_MASK that corresponds to a boolean healpix mask

        Returns
        -------
        None
            The cutsky catalog is modified in place.
        """
        self.logger.info('Applying angular mask.')
        if custom_mask_path is None:
            is_in_desi = is_in_desi_footprint(
                self.catalog['RA'],
                self.catalog['DEC'],
                release=release,
                program=program,
                npasses=npasses
            )
            self.catalog['HPX'], is_in_photo = self.is_in_photometric_region(
                self.catalog['RA'],
                self.catalog['DEC'],
                region=region
            )
            for key in self.catalog.keys():
                self.catalog[key] = self.catalog[key][is_in_desi & is_in_photo]
        else:
            mask = fitsio.read(custom_mask_path)
            nside = hp.npix2nside(len(mask['IN_MASK']))
            phi = np.radians(self.catalog['RA'])
            theta = np.radians(90 - self.catalog['DEC'])
            target_pixels = hp.ang2pix(nside, theta, phi)
            is_in_mask = mask['IN_MASK'][target_pixels]
            for key in self.catalog.keys():
                self.catalog[key] = self.catalog[key][is_in_mask]
            

    def apply_radial_mask(self, nz_filename: str, shape_only: bool = False) -> None:
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
        self.logger.info('Applying radial mask.')
        zbin_min, zbin_max, target_nz = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
        zbin_mid = (zbin_min + zbin_max) / 2
        nz_spline = InterpolatedUnivariateSpline(zbin_mid, target_nz, k=1, ext=3)
        raw_nbar = self.get_raw_nbar_at_z(zbin_mid)
        ratio = target_nz / raw_nbar
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

    def compute_nz(self, zedges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
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

    def is_in_photometric_region(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        region: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine if the given RA/Dec coordinates are within the specified photometric region.

        Parameters
        ----------
        ra : np.ndarray
            Array of right ascension values in degrees.
        dec : np.ndarray
            Array of declination values in degrees.
        region : str
            The photometric region to check. Options include 'N', 'DN', 'DS', 'N+SNGC',
            'SNGC', 'SSGC', 'DES', 'NGC', 'SGC'.

        Returns
        -------
        pixels : np.ndarray
            Healpix pixel numbers corresponding to the RA/Dec coordinates.
        mask : np.ndarray
            Boolean mask indicating whether each RA/Dec coordinate is within the specified region.
        """
        region = region.upper()
        if region not in VALID_REGIONS:
            raise ValueError(f"Invalid region '{region}'. Must be one of: {', '.join(VALID_REGIONS)}")

        if not HAS_REGRESSIS:
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
            # Precompute the healpix number
            nside = 256
            _, pixels = build_healpix_map(nside, ra, dec, return_pix=True)

            # Load DR9 footprint and create corresponding mask
            dr9_footprint = DR9Footprint(
                nside,
                mask_lmc=False,
                clear_south=False,
                mask_around_des=False,
                cut_desi=False
            )
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

    def add_columns_fiberassign(self, seed: int = 0) -> None:
        """
        Add columns to the catalog that are needed for fiber assignment.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Defaults to 0.
        """
        self.logger.info('Adding columns for fiber assignment.')

        priority = {'LRG': 3200, 'QSO': 3400, 'ELG': 3100, 'BGS': 2100}
        tile = {'LRG': 'DARK', 'QSO': 'DARK', 'ELG': 'DARK', 'BGS': 'BRIGHT'}

        csize = len(self.catalog['RA'])
        self.catalog['DESI_TARGET'] = desi_mask[self.tracer] * np.ones(csize, dtype='i8')
        self.catalog['PRIORITY'] = priority[self.tracer] * np.ones(csize, dtype='i8')
        self.catalog['SUBPRIORITY'] = np.random.uniform(size=csize, low=0, high=1).astype('f8')
        self.catalog['OBSCONDITIONS'] = obsconditions.mask(tile[self.tracer]) * np.ones(csize, dtype='i8')
        self.catalog['NUMOBS_MORE'] = np.ones(csize, dtype='i8')
        self.catalog['TARGETID'] = np.arange(csize, dtype='i8')

    def save(self, filename: str) -> None:
        """
        Save the cutsky catalog to a file. Supports .fits and .npy formats.

        Parameters
        ----------
        filename : str
            The path to the output file.
        """
        self.logger.info(f'Saving cutsky catalog to {filename}')
        if filename.endswith('.fits'):
            table = Table(self.catalog)
            myfits = fits.BinTableHDU(data=table)
            myfits.writeto(filename, overwrite=True)
        elif filename.endswith('.npy'):
            np.save(filename, self.catalog)
        else:
            raise ValueError('Unsupported file format. Use .fits or .npy.')
        

class CutskyHOD(BaseCutskyCatalog):
    """
    Patch together cubic boxes to form a pseudo-lightcone.
    """
    def __init__(
        self,
        varied_params: list[str],
        config_file: str | None = None,
        cosmo_idx: int = 0,
        phase_idx: int = 0,
        zranges: list[list[float]] = [[0.4, 0.6]],
        snapshots: list[float] = [0.5],
        DM_DICT: dict = LRG_Abacus_DM,
        load_existing_hod: bool = False,
        sim_type: str = 'base',
        tracer: str = 'LRG'
    ):
        """
        Initialize the CutskyHOD class. This checks the HOD parameters and 
        loads the relevant simulation data that will be used to sample the
        HOD galaxies from the snapshots later.

        Parameters
        ----------
        varied_params : list[str]
            List of HOD parameters that will be varied.
        config_file : str, optional
            Path to the configuration file for HOD parameters. If None,
            it will read the default file stored in the package.
        cosmo_idx : int, optional
            Index of the cosmology to use for the AbacusSummit simulation.
        phase_idx : int, optional
            Index of the phase to use for the AbacusSummit simulation.
        zranges : list[list[float]], optional
            List of redshift ranges for which to build the cutsky catalog.
            Each sublist should contain two elements: [zmin, zmax].
        snapshots : list[float], optional
            List of snapshots (redshifts) to use for building each redshift range
            specified in `zranges`.
        DM_DICT : dict, optional
            Dictionary containing the DM fields for the HOD sampling.
            Defaults to LRG_Abacus_DM, which is defined in utils.paths.
        load_existing_hod : bool, optional
            Flag to allow loading an existing HOD catalog in the `sample_hod` method.
            When True, prevents the Dark Matter catalog from being loaded and allows
            the use of pre-generated HOD catalogs (useful for quick debugging). 
            Defaults to False.
        sim_type : str, optional
            Type of simulation to use for the HOD sampling. Defaults to 'base' (2 Gpc/h).
        tracer : str, optional
            The type of tracer to use for the HOD sampling. Defaults to 'LRG'.
        """
        super().__init__()
        self.logger = logging.getLogger('CutskyHOD')
        self.config_file = config_file
        self.load_existing_hod = load_existing_hod
        self.varied_params = varied_params
        self.cosmo_idx = cosmo_idx
        self.phase_idx = phase_idx
        self.sim_type = sim_type
        self.tracer = tracer
        if len(zranges) != len(snapshots):
            raise ValueError('Number of redshift ranges must match number of snapshots.')
        self.zranges = zranges
        self.snapshots = snapshots
        self.boxsize = 500 if sim_type == 'small' else 2000
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
            ball = BoxHOD(
                varied_params=self.varied_params,
                DM_DICT=DM_DICT,
                sim_type=self.sim_type,
                redshift=zsnap,
                cosmo_idx=self.cosmo_idx,
                phase_idx=self.phase_idx,
                config_file=self.config_file
            )
            self.balls += [ball]
        self.cosmo = AbacusSummit(self.cosmo_idx)

    def _sample_hod(
        self,
        ball,
        hod_params: dict,
        nthreads: int = 1,
        seed: float = 0,
        target_nbar: float = None
    ):
        """
        Internal function to sample HOD galaxies from the given ball object
        using the provided HOD parameters.
        
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
        hod_dict = ball.run(
            hod_params,
            seed=seed,
            nthreads=nthreads,
            tracer_density_mean=target_nbar
        )[self.tracer]
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
            base_path = f'/global/cfs/projectdirs/desi/cosmosim/SecondGenMocks/CubicBox/{self.tracer}/z0.500/AbacusSummit_base_c000_ph000'
            mock_path = base_path + f'/{self.tracer}_real_space.fits'
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
        self,
        hod_params: dict,
        nthreads: int = 1,
        seed: float = 0,
        existing_hod_path: str | None = None,
        region: str = 'NGC',
        release: str = 'Y1',
        target_nz_filename: str | None = None,
        custom_xyz_file: str | None = None
    ) -> dict:
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
        region : str, optional
            The DESI photometric region, e.g., 'NGC', or 'SGC', by default 'NGC'.
        release : str, optional
            The DESI data release, e.g., 'Y1', 'Y3', or 'Y5, by default 'Y1'.
        target_nz_filename : str, optional
            Path to an n(z) filename that can be used as a reference to estimate what is the maximum
            number density that the HOD boxes require to allow for a radial mask to be applied later.
        custom_xyz_file : str
            If not None, a custom file is read for the positions of the tracers that define
            the survey volume bounds

        Returns
        -------
        dict
            The cutsky catalog containing positions, velocities, and other properties of the galaxies.
        """
        self.catalog = self.init_cutsky()

        self.raw_nbar_snapshots = []
        
        # construct one redshift shell at a time from the snapshots
        for i, (zsnap, zranges) in enumerate(zip(self.snapshots, self.zranges)):
            self.logger.info(f'Processing snapshot at z = {zsnap} for redshift range {zranges}')
            target_nbar = self.get_target_nbar(
                nz_filename=target_nz_filename,
                zmin=zranges[0],
                zmax=zranges[1],
                region=region
            )
            if self.load_existing_hod:
                box_positions, box_velocities = self.load_hod(mock_path=existing_hod_path)
            else:
                ball  = self.balls[i]
                box_positions, box_velocities = self._sample_hod(
                    ball,
                    hod_params,
                    nthreads=nthreads,
                    target_nbar=target_nbar,
                    seed=seed
                )
            self.raw_nbar_snapshots.append( len(box_positions) / (self.boxsize**3) )
            # replicate the box along each axis to cover more volume
            pos_min, pos_max = self.get_reference_borders(
                zranges,
                region=region,
                release=release,
                custom_xyz_file=custom_xyz_file
            )
            shifts = self.get_box_shifts(pos_min, pos_max)
            box_positions, box_velocities = self.get_box_replications(
                box_positions,
                box_velocities,
                pos_min,
                pos_max,
                target_nbar,
                shifts=shifts
            )
            box = mockfactory.BoxCatalog(
                data={'Position': box_positions, 'Velocity': box_velocities},
                position='Position',
                velocity='Velocity',
                boxsize=pos_max-pos_min,
                boxcenter=(pos_max+pos_min)/2,
            )
            cutsky_shell = self.box_to_cutsky(
                box=box,
                zmin=zranges[0],
                zmax=zranges[1],
                zrsd=zsnap,
                apply_rsd=True
            )
            for key in self.keys_cutsky:
                self.catalog[key].extend(cutsky_shell[key])
            del box_positions, box_velocities, box, cutsky_shell
        for key in self.keys_cutsky:
            self.catalog[key] = np.array(self.catalog[key])
        return self.catalog

    def get_box_shifts(
        self,
        pos_min: np.ndarray,
        pos_max: np.ndarray
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
        mappings_max = np.int32(np.ceil((pos_max - self.boxpad)/self.boxsize))
        mappings_min = np.int32(np.floor((pos_min + self.boxpad)/self.boxsize))
        shifts = []
        mappings = [np.arange(mappings_min[i],mappings_max[i]+1) for i in range(3)]
        for i in mappings[0]:
            for j in mappings[1]:
                for k in mappings[2]:
                    shifts.append([self.boxsize * np.array([i, j, k])])
        return shifts

    def get_box_replications(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        pos_min: np.ndarray,
        pos_max: np.ndarray,
        target_nbar: float,
        shifts: list | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
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
        """
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
                target_nbar
            )
            new_pos.append(temp_pos)
            new_vel.append(temp_vel)
        new_pos = np.concatenate(new_pos)
        new_vel = np.concatenate(new_vel)
        return new_pos, new_vel

    def box_to_cutsky(
        self,
        box,
        zmin: float,
        zmax: float,
        apply_rsd: bool = False,
        zrsd: float = None
    ):
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
        else: cutsky = box
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

    def get_reference_borders(
        self,
        zranges: list,
        region: str = 'NGC',
        release: str = 'Y1',
        custom_xyz_file: str | None = None
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
            region=region,
            release=release,
            tracer=self.tracer,
            custom_xyz_file=custom_xyz_file
        ) 
        if boxpad > 1:
            return pos_min - boxpad, pos_max + boxpad
        else:
            return pos_min - boxpad * self.boxsize, pos_max + boxpad * self.boxsize

    def get_target_nbar(
        self,
        nz_filename: str | None = None,
        zmin: float = 0.,
        zmax: float = 6.,
        nzpad: float = 1.1,
        region: str = 'NGC'
    ) -> float:
        """
        Get the maximum number density associated to a given tracer in a given redshift range
        from the observed n(z) file. This is to know what the number density of the created
        HOD should be if we later want to apply a radial mask to the cutsky catalog.

        Parameters
        ----------
        nz_filename : str, optional
            Path to the n(z) file. If None, it defaults to the Y1 LRG n(z) file.
            This file should contain columns (1, 2, 3) as zbin_min, zbin_max, and target_nz.
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
        if nz_filename is None:
            nz_filename = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/{self.tracer}_{region}_nz.txt'
        zbin_min, zbin_max, n_z = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
        chosen = np.logical_and(zbin_min >= zmin, zbin_max <= zmax)
        return nzpad * np.max(n_z[chosen])

    def get_pos_within_borders(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        pos_min: np.ndarray,
        pos_max: np.ndarray,
        target_nbar: float
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
        target_nbar : float
            Target number density of the particles.

        Returns
        -------
        pos : np.ndarray
            Filtered positions of the particles within the specified borders.
        """
        # target_ngal = int(target_nbar*self.boxsize**3)
        # chosen = np.random.choice(len(pos),target_ngal,replace=False)
        # pos = pos[chosen]
        # vel = vel[chosen]
        for i in range(3):
            chosen = np.logical_and(pos[:,i] > pos_min[i], pos[:,i] < pos_max[i])
            pos = pos[chosen]
            vel = vel[chosen]
        return pos,vel

    def get_raw_nbar_at_z(self, redshift: np.ndarray) -> np.ndarray | float:
        """
        Obtains the correct raw_nbar value for a given redshift input

        Parameters
        ----------
        redshift : np.ndarray
            Array of input redshifts

        Returns
        -------
        combined_raw_nbar : np.ndarray
            The raw_nbar values for each redshift
        """

        if len(self.raw_nbar_snapshots) == 1:
            return self.raw_nbar_snapshots[0]

        combined_raw_nbar = np.zeros_like(redshift)
        
        for raw_nbar, zrange in zip(self.raw_nbar_snapshots, self.zranges):

            select_targets = np.ones_like(redshift, dtype = bool)

            if zrange[0] != self.zranges[0][0]:
                select_targets *= (redshift > zrange[0] ) 
            if zrange[1] != self.zranges[-1][1]:
                select_targets *= (redshift < zrange[1] ) 
            
            combined_raw_nbar[select_targets] = raw_nbar

        return combined_raw_nbar
        
class CutskyRandoms(BaseCutskyCatalog):
    """
    Class to generate randoms in a cutsky region.
    """
    def __init__(
        self,
        rarange: tuple = (0., 360.),
        decrange: tuple = (-90., 90.),
        zrange: tuple = (0.4, 0.6),
        csize: int | None = None,
        nbar: float | None = None,
        seed: int | None = None,
        cosmo_idx: int = 0
    ):
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
        """
        super().__init__()
        self.logger = logging.getLogger('CutskyRandoms')
        self.rarange = rarange
        self.decrange = decrange
        self.zrange = zrange
        self.nbar = nbar
        self.keys_cutsky = ['RA', 'DEC', 'Z', 'Distance', 'Position']
        self.cosmo = AbacusSummit(cosmo_idx)
        r2d = self.cosmo.comoving_radial_distance
        self.drange = (r2d(zrange[0]), r2d(zrange[1]))
        self.catalog = RandomCutskyCatalog(
            rarange=self.rarange,
            decrange=self.decrange,
            drange=self.drange,
            csize=csize,
            nbar=nbar,
            seed=seed
        )
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
        area = radecbox_area(self.rarange, self.decrange)  # in square degrees
        fsky = area / 41253.0  # sky fraction covered by the randoms
        volume = 4/3 * np.pi * (self.drange[1]**3 - self.drange[0]**3) * fsky  # in (Mpc/h)^3
        return len(self.catalog['Z']) / volume

    def get_raw_nbar_at_z(self, *args):
        """
        Obtains the correct raw_nbar value for a randoms catalog

        Returns
        -------
        self.raw_nbar : float
            The raw_nbar values for the randoms
        """
        return self.raw_nbar
