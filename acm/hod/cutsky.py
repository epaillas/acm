import os
from pathlib import Path
import yaml
import numpy as np
from abacusnbody.hod import abacus_hod
from cosmoprimo.fiducial import AbacusSummit
import mockfactory
from astropy.io import fits
from astropy.table import Table
import logging
import warnings
import sys
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class CutskyHOD:
    """
    Patch together cubic boxes to form a pseudo-lightcone.
    """
    def __init__(self, varied_params, config_file=None, cosmo_idx=0, phase_idx=0,
        zranges=[[0.41, 0.6]], snapshots=[0.5]):
        self.logger = logging.getLogger('CutskyHOD')
        self.varied_params = varied_params
        self.cosmo_idx = cosmo_idx
        self.phase_idx = phase_idx
        self.sim_type = 'base'
        self.zranges = zranges
        self.snapshots = snapshots
        self.boxsize = 2000
        self.boxcenter = 0
        self.setup()

    def setup(self):
        self.balls = []
        for zsnap in self.snapshots:
            # self.logger.info(f'Processing {self.abacus_simname()} at z = {self.redshift}')
            ball = BoxHOD(varied_params=self.varied_params, sim_type=self.sim_type,
                          redshift=zsnap, cosmo_idx=self.cosmo_idx, phase_idx=self.phase_idx)
            self.balls += [ball]
        self.cosmo = AbacusSummit(self.cosmo_idx)

    def run(self, hod_params, nthreads=1, seed=0, generate_randoms=False, alpha_randoms=5,
            randoms_seed=42):
        data_cutsky = {}
        randoms_cutsky = {}
        for ball, zsnap, zranges in zip(self.balls, self.snapshots, self.zranges):
            hod_dict_i = ball.run(hod_params, seed=seed, nthreads=nthreads)['LRG']
            pos = np.c_[hod_dict_i['X'], hod_dict_i['Y'], hod_dict_i['Z']]
            vel = np.c_[hod_dict_i['VX'], hod_dict_i['VY'], hod_dict_i['VZ']]
            print('Generating data')
            data = mockfactory.BoxCatalog(
                data={'Position': pos, 'Velocity': vel},
                position='Position',
                velocity='Velocity',
                boxsize=self.boxsize,
                boxcenter=self.boxcenter,
            )
            data.recenter()
            data_nbar = len(data) / (self.boxsize**3)
            tmp_data_cutsky = self._to_cutsky(data, *zranges, zsnap, 
                                          apply_rsd=True,
                                          apply_radial_mask=True,
                                          radial_mask_norm=1/data_nbar,
                                          apply_footprint_mask=True)
            if generate_randoms:
                print('Generating randoms.')
                nbar_randoms = data_nbar * alpha_randoms
                randoms =  mockfactory.RandomBoxCatalog(
                    nbar=nbar_randoms, boxsize=self.boxsize,
                    boxcenter=self.boxcenter, seed=randoms_seed,
                )
                tmp_randoms_cutsky = self._to_cutsky(randoms, *zranges, zsnap,
                                                 apply_rsd=False,
                                                 apply_radial_mask=True,
                                                 radial_mask_norm=1/data_nbar,
                                                 apply_footprint_mask=True)
            # concatenate to previous shell, if any
            data_keys = ['RA', 'DEC', 'Z', 'RSDPosition', 'Distance', 'Position']
            randoms_keys = ['RA', 'DEC', 'Z', 'Position', 'Distance']
            if data_cutsky:
                for key in data_keys:
                    data_cutsky[key] = np.concatenate([data_cutsky[key], tmp_data_cutsky[key]])
                if generate_randoms:
                    for key in randoms_keys:
                        randoms_cutsky[key] = np.concatenate([randoms_cutsky[key], tmp_randoms_cutsky[key]])
            else:
                for key in data_keys:
                    data_cutsky[key] = tmp_data_cutsky[key]
                if generate_randoms:
                    for key in randoms_keys:
                        randoms_cutsky[key] = tmp_randoms_cutsky[key]
        if generate_randoms:
            return data_cutsky, randoms_cutsky
        return data_cutsky

    def _to_cutsky(self, catalog, zmin, zmax, zsnap, apply_rsd=False, apply_radial_mask=False,
        apply_footprint_mask=False, radial_mask_norm=None):
        nbar = len(catalog) / (self.boxsize**3)
        dist = self.cosmo.comoving_radial_distance((zmin + zmax) / 2)
        cutsky = self._apply_geometric_cuts(catalog, self.boxsize, dist)
        if apply_rsd: 
            cutsky = self._apply_rsd(cutsky, zsnap)
        cutsky = self._get_sky_positions(cutsky, apply_rsd)
        if apply_radial_mask:
            cutsky = self._apply_radial_mask(cutsky, zmin=zmin, zmax=zmax,
                                             norm=radial_mask_norm)
        if apply_footprint_mask:
            cutsky = self._apply_footprint_mask(cutsky)
        return cutsky

    def _apply_geometric_cuts(self, catalog, boxsize, dist):
        print('Applying geometric cuts.')
        # largest (RA, Dec) range we can achieve for a maximum distance of dist + boxsize / 2.
        drange, rarange, decrange = mockfactory.box_to_cutsky(boxsize=boxsize, dmax=dist + boxsize / 2.)
        rarange = np.array(rarange) + 192
        decrange = np.array(decrange) + 35
        # returned isometry corresponds to a displacement of the box along the x-axis to match drange, then a rotation to match rarange and decrange
        isometry, mask_radial, mask_angular = catalog.isometry_for_cutsky(drange=drange, rarange=rarange, decrange=decrange)
        return catalog.cutsky_from_isometry(isometry, rdd=None)

    def _apply_rsd(self, catalog, zsnap):
        print('Applying RSD.')
        a = 1 / (1 + zsnap) # scale factor
        H = 100.0 * self.cosmo.efunc(zsnap)  # Hubble parameter in km/s/Mpc
        rsd_factor = 1 / (a * H)  # multiply velocities by this factor to convert to Mpc/h
        catalog['RSDPosition'] = catalog.rsd_position(f=rsd_factor)
        return catalog

    def _get_sky_positions(self, catalog, apply_rsd=False):
        print('Converting to sky positions.')
        distance_to_redshift = mockfactory.DistanceToRedshift(distance=self.cosmo.comoving_radial_distance)
        pos = 'RSDPosition' if apply_rsd else 'Position'
        catalog['Distance'], catalog['RA'], catalog['DEC'] = mockfactory.cartesian_to_sky(catalog[pos])
        catalog['Z'] = distance_to_redshift(catalog['Distance'])
        return catalog

    def _apply_radial_mask(self, catalog, zmin=0., zmax=6., seed=42, norm=None):
        print('Applying radial mask.')
        from mockfactory import TabulatedRadialMask
        nz_filename = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/LRG_NGC_nz.txt'
        zbin_min, zbin_max, n_z = np.genfromtxt(nz_filename, usecols=(1, 2, 3)).T
        zbin_mid = (zbin_min + zbin_max) / 2
        zedges = np.insert(zbin_max, 0, zbin_min[0])
        dedges = self.cosmo.comoving_radial_distance(zedges)
        volume = dedges[1:]**3 - dedges[:-1]**3
        mask_radial = mockfactory.TabulatedRadialMask(z=zbin_mid, nbar=n_z, interp_order=2,
                                                      zrange=(zmin, zmax), norm=norm)
        return catalog[mask_radial(catalog['Z'], seed=seed)]

    def _apply_footprint_mask(self, catalog):
        print('Applying footprint mask.')
        from mockfactory.desi import is_in_desi_footprint
        is_in_desi = is_in_desi_footprint(catalog['RA'], catalog['DEC'], release='y1', program='dark', npasses=None)
        return catalog[is_in_desi]