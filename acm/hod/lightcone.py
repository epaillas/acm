import os
import yaml
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.table import Table

# cosmodesi/acm
import mockfactory
from abacusnbody.hod import abacus_hod
from cosmoprimo.fiducial import AbacusSummit

from acm.utils.paths import get_Abacus_dirs
LRG_Abacus_DM = get_Abacus_dirs(tracer='LRG', simtype='lightcone')


import logging
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# TODO : add docstrings !
class LightconeHOD:
    def __init__(
        self, 
        varied_params, 
        config_file: str = None, 
        cosmo_idx: int = 0, 
        phase_idx: int = 0,
        zrange: list = [0.4, 0.8],
        DM_DICT: dict = LRG_Abacus_DM,
        use_abacus_base: bool = False,
        ):
        self.logger = logging.getLogger('LightconeHOD')
        self.cosmo_idx = cosmo_idx
        self.phase_idx = phase_idx
        self.sim_type = 'base'
        self.zrange = zrange
        self.boxsize = 2000
        if config_file is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = Path(config_dir) /  'lightcone.yaml'
        config = yaml.safe_load(open(config_file))
        self.setup(config, DM_DICT)
        self.check_params(varied_params)
        self.use_abacus_base = use_abacus_base
        self.monte_carlo_sampling_count = 10000

    @property
    def snap_redshifts(self):
        return [0.400, 0.450, 0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950, 1.025, 1.100]

    def snap_in_zrange(self):
        snap_min = np.abs(np.array(self.snap_redshifts) - self.zrange[0]).argmin()
        snap_max = np.abs(np.array(self.snap_redshifts) - self.zrange[1]).argmin()
        snaps = self.snap_redshifts[snap_min:snap_max+2]  # Include an extra snapshot at high-z to avoid edge effects
        self.logger.info(f'Lightcone composed of snapshots at z: {snaps}.')
        return snaps
        # return [z for z in self.snap_redshifts if z >= self.zrange[0] and z <= self.zrange[1]]

    def abacus_simdirs(self, DM_DICT: dict):
        sim_dir = DM_DICT[self.sim_type]['sim_dir']
        subsample_dir = DM_DICT[self.sim_type]['subsample_dir']
        return sim_dir, subsample_dir

    def abacus_simname(self):
        return f'AbacusSummit_{self.sim_type}_c{self.cosmo_idx:03}_ph{self.phase_idx:03}'

    def setup(self, config: dict, DM_DICT: dict):
        sim_params = config['sim_params']
        sim_dir, subsample_dir = self.abacus_simdirs(DM_DICT)
        sim_params['sim_dir'] = sim_dir
        sim_params['subsample_dir'] = subsample_dir
        sim_params['sim_name'] = self.abacus_simname()
        HOD_params = config['HOD_params']
        self.balls = []
        for znap in self.snap_in_zrange():
            sim_params['z_mock'] = znap
            self.balls += [abacus_hod.AbacusHOD(sim_params, HOD_params)]
        self.cosmo = AbacusSummit(self.cosmo_idx)
        # self.az = 1 / (1 + self.redshift)
        # self.hubble = 100 * self.cosmo.efunc(self.redshift)
        self.logger.info(f'Processing {self.abacus_simname()} at zrange = {self.zrange}')

    def check_params(self, params):
        params = list(params)
        params = self.param_mapping(params)
        for param in params:
            if param not in self.balls[0].tracers['LRG'].keys():
                raise ValueError(f'Invalid parameter: {param}. Valid list '
                                 f'of parameters include: {list(self.balls[0].tracers["LRG"].keys())}')
        self.logger.info(f'Varied parameters: {params}.')
        self.varied_params = params
        default = {key: value for key, value in self.balls[0].tracers['LRG'].items() if key not in params}
        self.logger.info(f'Default parameters: {default}.')

    def run(
        self, 
        hod_params, 
        nthreads: int = 1, 
        tracer: str = 'LRG', 
        make_randoms: bool = False, 
        add_weights: bool = False,
        seed: float = None, 
        save_fn: str = None, 
        full_sky: bool = False, 
        alpha_rand: int = 1, 
        apply_radial_mask: bool = False,
        ):
        if seed == 0: seed = None
        if tracer not in ['LRG']:
            raise ValueError('Only LRGs are currently supported.')
        hod_params = self.param_mapping(hod_params)
        if set(hod_params.keys()) != set(self.varied_params):
            raise ValueError('Invalid HOD parameters. Must match the varied parameters.')
        for i, ball in enumerate(self.balls):
            for key in hod_params.keys():
                if key == 'sigma' and tracer == 'LRG':
                    ball.tracers[tracer][key] = 10**hod_params[key]
                else:
                    ball.tracers[tracer][key] = hod_params[key]
            ball.tracers[tracer]['ic'] = 1
            if i == 0:
                hod_dict = ball.run_hod(ball.tracers, ball.want_rsd, Nthread=nthreads, reseed=seed)
            else:
                hod_dict_i = ball.run_hod(ball.tracers, ball.want_rsd, Nthread=nthreads, reseed=seed)
                for key in hod_dict_i[tracer].keys():
                    if key == 'Ncent': 
                        hod_dict[tracer][key] += hod_dict_i[tracer][key]
                    else:
                        hod_dict[tracer][key] = np.concatenate([hod_dict[tracer][key], hod_dict_i[tracer][key]])
            # positions_dict = self.get_positions(hod_dict, tracer)
        self.format_catalog(hod_dict, save_fn, tracer, full_sky, apply_radial_mask)
        if make_randoms:
            zmin = self.zrange[0]
            zmax = self.zrange[1]
            nbar = self.get_data_nbar(hod_dict, tracer, full_sky)
            randoms_dict = self._make_randoms(nbar=nbar, zmin=zmin, zmax=zmax, apply_radial_mask=apply_radial_mask,
                                              full_sky=full_sky, alpha=alpha_rand, tracer=tracer)
            return hod_dict, randoms_dict
        return hod_dict

    def recenter_box(self, hod_dict, tracer: str = 'LRG'):
        data = hod_dict[tracer]
        data['X'] += 990
        data['Y'] += 990
        data['Z'] += 990

    def remove_outbounds(self, hod_dict, tracer: str = 'LRG'):
        mask = (hod_dict[tracer]['X'] >= 0) & (hod_dict[tracer]['Y'] >= 0) & (hod_dict[tracer]['Z'] >= 0)
        for key in hod_dict[tracer].keys():
            hod_dict[tracer][key] = hod_dict[tracer][key][mask]

    def apply_zcut(self, hod_dict, zmin: float, zmax: float, tracer: str = 'LRG'):
        self.logger.info(f'Applying redshift cut: {zmin} < z < {zmax}.')
        mask = (hod_dict[tracer]['Z'] >= zmin) & (hod_dict[tracer]['Z'] <= zmax)
        for key in hod_dict[tracer].keys():
            hod_dict[tracer][key] = hod_dict[tracer][key][mask]

    def get_sky_coordinates(self, hod_dict, tracer: str = 'LRG'):
        from mockfactory import cartesian_to_sky, DistanceToRedshift
        data = hod_dict[tracer]
        dist, ra, dec = cartesian_to_sky(np.c_[data['X'], data['Y'], data['Z']])
        d2z = DistanceToRedshift(self.cosmo.comoving_radial_distance)
        redshift = d2z(dist)
        hod_dict[tracer]['RA'] = ra
        hod_dict[tracer]['DEC'] = dec
        hod_dict[tracer]['Redshift'] = redshift

    def make_full_sky(self, hod_dict, tracer: str = 'LRG'):
        data = hod_dict[tracer]
        x = data['X']
        y = data['Y']
        z = data['Z']
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
        hod_dict[tracer]['X'] = pos[:, 0]
        hod_dict[tracer]['Y'] = pos[:, 1]
        hod_dict[tracer]['Z'] = pos[:, 2]
        for key in ['MASS', 'ID', 'IS_CENT']:
            if key in hod_dict[tracer]:
                hod_dict[tracer][key] = np.tile(hod_dict[tracer][key], 8)

    def get_data_nbar(self, hod_dict, tracer: str = 'LRG', full_sky: bool = False):
        """
        Compute the number density of the data catalog, which is defined
        by an octant of the spherical shell delimited by the redshift cuts.
        """
        data = hod_dict[tracer]
        dmin, dmax = self.cosmo.comoving_radial_distance(self.zrange)
        
        if self.use_abacus_base:
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
            nbar = len(data['Z']) / volume
            
        else:
            # Abacus huge (assumes that shells are entirely contained within the survey box)
            # Monte carlo sampling would be more accurate if this is not the case
            volume = 4/3 * np.pi * (dmax**3 - dmin**3)
            correction = 1 if full_sky else 8  # divide by 8 if only using a sky octant
            nbar = len(data['Z']) / (volume / correction)
        return nbar
        
    def format_catalog(self, hod_dict, save_fn: str = False, tracer: str = 'LRG', full_sky: str = False, apply_radial_mask: str = False):
        Ncent = hod_dict[tracer]['Ncent']
        hod_dict[tracer].pop('Ncent', None)
        is_central = np.zeros(len(hod_dict[tracer]['x']))
        is_central[:Ncent] += 1
        hod_dict[tracer]['is_cent'] = is_central
        hod_dict[tracer] = {k.upper():v  for k, v in hod_dict[tracer].items()}
        self.recenter_box(hod_dict, tracer)
        self.remove_outbounds(hod_dict, tracer)
        if full_sky: self.make_full_sky(hod_dict)
        self.get_sky_coordinates(hod_dict)
        self.drop_cartesian(hod_dict)
        self.apply_zcut(hod_dict, self.zrange[0], self.zrange[1])
        data_nbar = self.get_data_nbar(hod_dict, tracer, full_sky)
        self.logger.info(f'Raw data nbar: {data_nbar}' )
        if apply_radial_mask:
            nz_filename = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/{tracer}_NGC_nz.txt'
            #print(1/data_nbar)
            self.apply_radial_mask(hod_dict, nz_filename)
            self.logger.info(f'Downsampled data nbar: {self.get_data_nbar(hod_dict, tracer, full_sky)}' )
        if save_fn:
            table = Table(hod_dict[tracer])
            header = fits.Header({'N_cent': Ncent, 'gal_type': tracer, **self.ball.tracers[tracer]})
            myfits = fits.BinTableHDU(data=table, header=header)
            myfits.writeto(save_fn, overwrite=True)
            self.logger.info(f'Saving {save_fn}.')

    def drop_cartesian(self, hod_dict, tracer: str = 'LRG'):
        hod_dict[tracer].pop('X')
        hod_dict[tracer].pop('Y')
        hod_dict[tracer].pop('Z')
        if 'VX' in hod_dict[tracer]: hod_dict[tracer].pop('VX')
        if 'VY' in hod_dict[tracer]: hod_dict[tracer].pop('VY')
        if 'VZ' in hod_dict[tracer]: hod_dict[tracer].pop('VZ')
        hod_dict[tracer]['Z'] = hod_dict[tracer].pop('Redshift')

    def param_mapping(self, hod_params: dict | list):
        """
        Map custom HOD parameters to Abacus HOD parameters. 

        Parameters
        ----------
        hod_params : dict or list
            Dictionary or list of HOD parameters.

        Returns
        -------
        dict or list
            Dictionary or list of AbacusHOD parameters.
        
        Raises
        ------
        ValueError
            If the type of hod_params is not dict or list.
        """
        
        # Add custom keys here if needed. 
        # Be careful to the one-to-one position mapping in the list !!
        abacus_keys = ['logM1', 'Acent', 'Asat', 'Bcent', 'Bsat']
        custom_keys = ['logM_1', 'A_cen', 'A_sat', 'B_cen', 'B_sat']
        
        # Check if custom keys are used
        if any(key in hod_params for key in custom_keys): # Same syntax for dict and list :)
            for abacus_key, custom_key in zip(abacus_keys, custom_keys): 
                if custom_key in hod_params: # Just in case not all custom keys are used
                    # Replace custom keys with Abacus keys
                    if type(hod_params) is dict:
                        hod_params[abacus_key] = hod_params.pop(custom_key)
                    elif type(hod_params) is list:
                        hod_params[hod_params.index(custom_key)] = abacus_key
                    else:
                        raise ValueError('Invalid type for hod_params. Must be either dict or list.')
        return hod_params

    def _make_randoms(
        self, 
        nbar: float, 
        zmin: float, 
        zmax: float, 
        alpha: float=1, 
        full_sky: bool = False, 
        apply_radial_mask: bool = False, 
        tracer: str = 'LRG'
        ):
        from mockfactory import RandomBoxCatalog
        self.logger.info(f'Generating random catalog.')
        # nbar = 6e-4  # hardcoded for LRG
        if self.use_abacus_base:
            # Abacus base
            pos = RandomBoxCatalog(
                boxsize=self.boxsize * 2, boxcenter=self.boxsize, nbar=nbar*alpha, seed=42
            )['Position']
            in_vol = (pos[:, 0]<self.boxsize)*~((pos[:, 1]>self.boxsize)*(pos[:, 2]>self.boxsize))
            randoms = {tracer: {'X': pos[:, 0][in_vol], 'Y': pos[:, 1][in_vol], 'Z': pos[:, 2][in_vol]}}
        else:
            # Abacus huge
            pos = RandomBoxCatalog(
                boxsize=self.boxsize, boxcenter=self.boxsize/2, nbar=nbar*alpha, seed=42
            )['Position']
            randoms = {tracer: {'X': pos[:, 0], 'Y': pos[:, 1], 'Z': pos[:, 2]}}
        if full_sky: self.make_full_sky(randoms)
        self.get_sky_coordinates(randoms)
        self.drop_cartesian(randoms)
        zmask = (randoms[tracer]['Z'] >= zmin) & (randoms[tracer]['Z'] <= zmax)
        for key in randoms[tracer].keys():
            randoms[tracer][key] = randoms[tracer][key][zmask]
        if apply_radial_mask:
            nz_filename = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/{tracer}_NGC_nz.txt'
            #randoms_nbar = self.get_data_nbar(randoms, tracer, full_sky)
            self.apply_radial_mask(randoms, nz_filename, shape_only=True)
        dmin, dmax = self.cosmo.comoving_radial_distance(self.zrange)
        return randoms

    def apply_radial_mask(self, hod_dict, nz_filename: str, tracer: str = 'LRG', shape_only: bool = False):
        # TODO need to make this work for other releases
        self.logger.info('Applying radial mask.')
        nz_filename = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/LRG_NGC_nz.txt'
        zbin_min, zbin_max, target_nz = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
        zbin_mid = (zbin_min + zbin_max) / 2
        zedges = np.insert(zbin_max, 0, zbin_min[0])
        dedges = self.cosmo.comoving_radial_distance(zedges)
        volume = 4/3 * np.pi * (dedges[1:]**3 - dedges[:-1]**3) / 8 
        
        # Handle shells exterior to the Abacus boxes
        if np.any(dedges > self.boxsize):
            
            exterior_indices = np.where(dedges > self.boxsize)[0]
            exterior_shells = (dedges[exterior_indices] + dedges[exterior_indices - 1]) / 2

            # fraction of each shell inside the simulation volume
            filling_fractions = self.shell_filling_fraction(exterior_shells)
                    
            # Adjust volumes using volume filling fractions
            volume[exterior_indices-1] = volume[exterior_indices-1] * filling_fractions
                
        data_nz = np.histogram(hod_dict[tracer]['Z'], bins=zedges)[0] / volume
        zmin_data = hod_dict[tracer]['Z'].min()
        zmax_data = hod_dict[tracer]['Z'].max()
        ratio = target_nz / data_nz
        if shape_only:
            ratio /= np.max(ratio[~np.isinf(ratio)])
        select_mask = np.zeros_like(hod_dict[tracer]['Z'], dtype=bool)
        for i in range(len(zbin_mid) - 1):
            if zbin_mid[i] < zmin_data or zbin_mid[i + 1] > zmax_data:
                continue
            z_mask = (hod_dict[tracer]['Z'] >= zbin_mid[i]) & (hod_dict[tracer]['Z'] < zbin_mid[i + 1])
            n_galaxies = np.sum(z_mask)
            if n_galaxies > 0:
                n_select = int(np.round(ratio[i] * n_galaxies))
                select_indices = np.random.choice(np.where(z_mask)[0], size=n_select, replace=False)
                select_mask[select_indices] = True
        for key in hod_dict[tracer].keys():
            hod_dict[tracer][key] = hod_dict[tracer][key][select_mask]

    def shell_filling_fraction(self, shells):
    
        # Muller-Marsaglia octant sampling
        # evenly distribute points along an octant of a unit sphere
        num_samples = self.monte_carlo_sampling_count
        sample_points = np.abs(np.random.normal(0,1, (3,num_samples)))
        sample_points /= np.linalg.norm(sample_points, axis=0)
    
        # fraction of each shell inside the simulation volume
        vol_fractions = []
    
        if self.use_abacus_base:
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
            # Abacus huge
            for shell_radius in shells:
                # count how many points in the shell are in the box
                vol_fractions.append(
                    np.sum((sample_points[0]<self.boxsize/shell_radius)*\
                           (sample_points[1]<self.boxsize/shell_radius)*\
                           (sample_points[2]<self.boxsize/shell_radius)\
                          )/len(sample_points[0])
                ) 

        return np.array(vol_fractions)