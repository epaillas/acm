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


class CutskyHOD:
    """
    Patch together cubic boxes to form a pseudo-lightcone.
    """
    def __init__(
            self, varied_params, config_file: str = None, cosmo_idx: int = 0, 
            phase_idx: int = 0, zranges: list[list] = [[0.41, 0.6]], 
            snapshots: list = [0.5], DM_DICT: dict = LRG_Abacus_DM,
            load_existing_hod: bool = False):
        self.logger = logging.getLogger('CutskyHOD')
        self.load_existing_hod = load_existing_hod
        self.varied_params = varied_params
        self.cosmo_idx = cosmo_idx
        self.phase_idx = phase_idx
        self.sim_type = 'base'
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

    def setup_hod(self, DM_DICT: dict):
        """
        Initialize AbacusHOD objects for each snapshot.
        """
        self.balls = []
        for zsnap in self.snapshots:
            ball = BoxHOD(varied_params=self.varied_params,
                          DM_DICT=DM_DICT, sim_type=self.sim_type,
                          redshift=zsnap, cosmo_idx=self.cosmo_idx,
                          phase_idx=self.phase_idx)
            self.logger.info(f'Processing {ball.abacus_simname()} at z = {ball.redshift}')
            self.balls += [ball]
        self.cosmo = AbacusSummit(self.cosmo_idx)

    def sample_hod(self, ball, hod_params: dict, nthreads: int = 1, seed: float = 0):
        """
        Sample HOD galaxies from the given ball object using the provided HOD parameters.
        """
        hod_dict = ball.run(hod_params, seed=seed, nthreads=nthreads)['LRG']
        pos = np.c_[hod_dict['X'], hod_dict['Y'], hod_dict['Z']]
        vel = np.c_[hod_dict['VX'], hod_dict['VY'], hod_dict['VZ']]
        return pos.astype(np.float32), vel.astype(np.float32)

    def load_hod(self, mock_path='None'):
        if mock_path is None:
            base_path = '/global/cfs/projectdirs/desi/cosmosim/SecondGenMocks/CubicBox/LRG/z0.500/AbacusSummit_base_c000_ph000'
            mock_path = base_path +'/LRG_real_space.fits'
            # data_dir = '/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z0.5/yuan23_prior/c000_ph000/seed0'
            # data_fn = Path(data_dir) / 'hod030_raw.fits'
        data  = fitsio.read(mock_path)
        pos = np.vstack([data['x'],data['y'],data['z']]).T
        vel = np.vstack([data['vx'],data['vy'],data['vz']]).T
        return pos.astype(np.float32), vel.astype(np.float32)

    def init_data(self):
        """
        Initialize the data dictionary.
        Returns
        """
        self.keys_data = ['RA', 'DEC', 'Z', 'RSDPosition', 'Distance', 'Position']
        data = {}
        for key in self.keys_data:
            data[key] = []
        return data

    def init_randoms(self):
        """
        Initialize the randoms dictionary.
        """
        self.keys_randoms = ['RA', 'DEC', 'Z', 'Distance', 'Position']
        randoms = {}
        for key in self.keys_randoms:
            randoms[key] = []
        return randoms

    def run(
            self, hod_params: dict, nthreads: int = 1, seed: float = 0, 
            generate_randoms: bool = False,  box_margin: float = 300, alpha_randoms: int = 5, 
            randoms_seed: float = 42, existing_mock_path: str = 'None', 
            region: str ='NGC', release: str ='Y1', program: str ='dark'):

        data = self.init_data()
        randoms = self.init_randoms() if generate_randoms else None

        # construct one redshift shell at a time from the snapshots
        for i in range(len(self.snapshots)):
            zsnap   = self.snapshots[i]
            zranges = self.zranges[i]
            self.logger.info(f'Processing snapshot at z = {zsnap} with redshift range {zranges}')
            if self.load_existing_hod:
                pos_box, vel_box = self.load_hod(mock_path=existing_hod_path)
            else:
                ball    = self.balls[i]
                pos_box, vel_box = self.sample_hod(ball, hod_params, nthreads=nthreads, seed=seed)

            target_nbar = self.get_target_nbar(zmin=zranges[0], zmax=zranges[1])

            pos_min,pos_max = self.get_reference_borders(zranges, region=region, release=release)
            # replicate the box along each axis to cover more volume
            shifts = self.get_box_shifts(pos_min, pos_max)  # shifts to translate particle positions

            pos_rep, vel_rep = self.get_box_replications(pos_box, vel_box,
                                                         pos_min, pos_max, target_nbar, shifts=shifts)
            # data_nbar = len(pos_box) / (self.boxsize_snapshot ** 3)
            data_nbar = target_nbar
            data_rep = mockfactory.BoxCatalog(data={'Position': pos_rep, 'Velocity': vel_rep},
                                              position='Position', velocity='Velocity',
                                              boxsize=pos_max-pos_min, boxcenter=(pos_max+pos_min)/2,)
            data_rep = self.box_to_cutsky(box=data_rep, zmin=zranges[0], zmax=zranges[1], 
                                          zrsd=zsnap, radial_mask_norm=1/data_nbar, apply_rsd=True,
                                          apply_radial_mask=True, apply_footprint_mask=True,
                                          region=region, release=release, program=program)
            for key in self.keys_data:
                data[key].extend(data_rep[key])

            if generate_randoms:
                self.logger.info('Generating randoms.')
                nbar_randoms = data_nbar * alpha_randoms
                randoms_rep =  mockfactory.RandomBoxCatalog(nbar=nbar_randoms, boxsize=pos_max-pos_min,
                                                            boxcenter=(pos_max+pos_min)/2, seed=randoms_seed)
                randoms_rep = self.box_to_cutsky(randoms_rep, zmin=zranges[0], zmax=zranges[1],
                                                 apply_rsd=False, apply_radial_mask=True,
                                                 radial_mask_norm=1/data_nbar,
                                                 apply_footprint_mask=True,
                                                 region=region, release=release, program=program)
                for key in self.keys_randoms:
                    randoms[key].extend(randoms_rep[key])

        for key in self.keys_data:
            data[key] = np.array(data[key])
        if generate_randoms:
            for key in self.keys_randoms:
                randoms[key] = np.array(randoms[key])
            return data, randoms
        return data

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

    def box_to_cutsky(
            self, box, zmin: float, zmax: float, zrsd: float = None, 
            apply_rsd: bool = False, apply_radial_mask: bool = False,
            apply_footprint_mask: bool = False, radial_mask_norm: bool = None,
            region: str = 'NGC', release: str = 'Y1', program: str = 'dark',):
        """
        Convert a box catalog to a cutsky catalog by applying geometric cuts, RSD, and masks.
        """
        if apply_rsd: 
            cutsky = self.apply_rsd(box, zrsd)
        else:
            cutsky = box
        cutsky = self._get_sky_positions(cutsky, apply_rsd)
        if apply_radial_mask:
            cutsky = self.apply_radial_mask(cutsky, zmin=zmin, zmax=zmax,
                                             norm=radial_mask_norm,
                                             region=region, release=release)
        if apply_footprint_mask:
            cutsky = self.apply_footprint_mask(cutsky, region=region, release=release, program=program)
        return cutsky

    def apply_geometric_cuts(self, catalog, zmin, zmax, region='NGC', release='Y1'):
        self.logger.info('Applying geometric cuts.')
        # largest (RA, Dec) range we can achieve for a maximum distance of dist + boxsize / 2.
        drange, rarange, decrange = self.get_reference_skyrange((zmin, zmax), region=region, release=release)
        # returned isometry corresponds to a displacement of the box along the x-axis to match drange, then a rotation to match rarange and decrange
        isometry, mask_radial, mask_angular = catalog.isometry_for_cutsky(drange=drange, rarange=rarange, decrange=decrange)
        return catalog.cutsky_from_isometry(isometry, rdd=None)

    def apply_rsd(self, catalog, zsnap: float):
        self.logger.info('Applying RSD.')
        a = 1 / (1 + zsnap) # scale factor
        H = 100.0 * self.cosmo.efunc(zsnap)  # Hubble parameter in km/s/Mpc
        rsd_factor = 1 / (a * H)  # multiply velocities by this factor to convert to Mpc/h
        catalog['RSDPosition'] = catalog.rsd_position(f=rsd_factor)
        return catalog

    def _get_sky_positions(self, catalog, apply_rsd: bool = False):
        self.logger.info('Converting to sky positions.')
        distance_to_redshift = mockfactory.DistanceToRedshift(distance=self.cosmo.comoving_radial_distance)
        pos = 'RSDPosition' if apply_rsd else 'Position'
        catalog['Distance'], catalog['RA'], catalog['DEC'] = mockfactory.cartesian_to_sky(catalog[pos])
        catalog['Z'] = distance_to_redshift(catalog['Distance'])
        return catalog

    def apply_radial_mask(self, catalog, zmin: float = 0., zmax: float = 6., seed: float = 42, norm=None,
        region='NGC', release='Y1'):
        # TODO need to make this work for other releases
        self.logger.info('Applying radial mask.')
        nz_filename = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/LRG_{region}_nz.txt'
        zbin_min, zbin_max, n_z = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
        zbin_mid = (zbin_min + zbin_max) / 2
        zedges = np.insert(zbin_max, 0, zbin_min[0])
        dedges = self.cosmo.comoving_radial_distance(zedges)
        volume = dedges[1:]**3 - dedges[:-1]**3
        mask_radial = mockfactory.TabulatedRadialMask(z=zbin_mid, nbar=n_z, interp_order=2,
                                                      zrange=(zmin, zmax), norm=norm)
        return catalog[mask_radial(catalog['Z'], seed=seed)]

    def apply_footprint_mask(self, catalog, region='N+SNGC', release='Y1', npasses=None, program='dark'):
        self.logger.info('Applying footprint mask.')
        is_in_desi = is_in_desi_footprint(catalog['RA'], catalog['DEC'], release=release, program=program, npasses=npasses)
        catalog['HPX'], is_in_photo = self.is_in_photometric_region(catalog['RA'], catalog['DEC'], region=region)
        return catalog[is_in_desi & is_in_photo]

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

    def get_reference_borders(self, zranges, region='NGC', release='Y1'):
        """
        Gets the reference borders in cartesian borders to apply to the box replications.
        """
        boxpad = self.boxpad
        assert boxpad > 0
        pos_min, pos_max = minmax_xyz_desi(zranges, region=region, release=release, tracer='LRG') 
        if boxpad > 1:
            return pos_min - boxpad, pos_max + boxpad
        else:
            return pos_min - boxpad * self.boxsize_snapshot, pos_max + boxpad * self.boxsize_snapshot

    def get_reference_skyrange(self, zranges, region='NGC', release='Y1'):
        """
        Gets the reference ra, min, distance ranges in degrees and Mpc/h.
        """
        return minmax_skycoord_desi(zranges, region=region, release=release, tracer='LRG')
            
    def get_target_nbar(self, zmin: float = 0., zmax: float = 6., nzpad=1.1, region='NGC', release='Y1'):
        # TODO need to make this work for other releases
        self.logger.info('Applying radial mask.')
        nz_filename = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/LRG_{region}_nz.txt'
        zbin_min, zbin_max, n_z = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
        chosen = np.logical_and(zbin_min>=zmin,zbin_max<=zmax)
        return nzpad * np.max(n_z[chosen])

    def get_pos_within_borders(self,pos,vel,pos_min,pos_max,target_nbar):
        target_ngal = int(target_nbar*self.boxsize_snapshot**3)
        chosen = np.random.choice(len(pos),target_ngal,replace=False)
        pos = pos[chosen]
        vel = vel[chosen]
        for i in range(3):
            chosen = np.logical_and(pos[:,i]>pos_min[i],pos[:,i]<pos_max[i])
            pos = pos[chosen]
            vel = vel[chosen]
        return pos,vel