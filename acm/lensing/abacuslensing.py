# Based on the AbacusLensing products
# presented in https://arxiv.org/abs/2305.11935

from abc import ABC, abstractmethod
import asdf
import healpy as hp
import numpy as np
from pathlib import Path


BASE_DIR = Path("/global/cfs/cdirs/desi/public/cosmosim/AbacusLensing/v1")

class AbacusLensingMap(ABC):
    """
    Abstract base class for Abacus Lensing Maps.
    """
    def __init__(self, snap_idx=47, cosmo_idx=0, phase_idx=0, sim_type='base'):
        """
        Initialize the AbacusLensingMap with snapshot index, cosmology index, phase index, and simulation type.
        :param snap_idx: Snapshot index (maps to source redshift). Default is 47 (CMB lensing)
        :param cosmo_idx: Cosmology index (default is 0)
        :param phase_idx: Phase index (default is 0)
        :param sim_type: Simulation type ('base' or 'huge')
        """
        self.snap_idx = snap_idx
        self.cosmo_idx = cosmo_idx
        self.phase_idx = phase_idx
        if sim_type == 'huge':
            assert phase_idx in [201, 202], "Phase index for 'huge' simulation must be 201 or 202."
        self.sim_type = sim_type
        self.nside = 16384
        self.map = self.read_map()

    @abstractmethod
    def read_map(self):
        """
        Abstract method to read the map data from an ASDF file.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def sample_mask(self):
        """
        Read the binary mask that tells us whether we are in a region of the sky with data
        and calculates the right ascension and declination of the unmasked pixels.
        """
        map_dir = f"AbacusSummit_{self.sim_type}_c{self.cosmo_idx:03}_ph{self.phase_idx:03}"
        f = asdf.open(BASE_DIR / map_dir / f"mask_{self.snap_idx:05d}.asdf")
        mask = f['data']['mask'] == 1
        npix = hp.nside2npix(self.nside)
        m = np.arange(npix)
        ra, dec = hp.pix2ang(nside=self.nside, ipix=m, lonlat=True)
        self.ra = ra[mask]
        self.dec = dec[mask]
        # TODO the code below should go somewhere else
        if hasattr(self, 'kappa'):
            self.kappa = self.kappa[mask]
        if hasattr(self, 'gamma1'):
            self.gamma1 = self.gamma1[mask]
            self.gamma2 = self.gamma2[mask]

    def plot_mollview(self, save_fn=None):
        """
        Plot the (kappa/shear) convergence map using Mollweide projection.
        """
        import matplotlib.pyplot as plt
        import healpy as hp
        toplot = self.kappa if self.map_type == 'kappa' else self.gamma1
        hp.mollview(toplot, badcolor='w', cmap='coolwarm')
        if save_fn:
            plt.savefig(save_fn)
        plt.close()


class AbacusConvergenceMap(AbacusLensingMap):
    """
    Class for AbacusLensing convergence Maps.
    """
    def __init__(self, snap_idx=47, cosmo_idx=0, phase_idx=0, sim_type='base'):
        self.map_type = 'kappa'
        super().__init__(snap_idx, cosmo_idx, phase_idx, sim_type)

    def read_map(self):
        """
        Read the convergence map from the ASDF file.
        """
        map_dir = f"AbacusSummit_{self.sim_type}_c{self.cosmo_idx:03}_ph{self.phase_idx:03}"
        f = asdf.open(BASE_DIR / map_dir / f"{self.map_type}_{self.snap_idx:05d}.asdf")
        self.header = f['header']
        self.data = f['data']
        self.kappa = hp.ma(f['data']['kappa'], badval=0)
        self.kappa.mask = self.kappa == 0

    def to_treecorr(self):
        """
        Convert the map to a TreeCorr catalog.
        """
        import treecorr
        self.treecorr = treecorr.Catalog(
            ra=self.ra, dec=self.dec, k=self.kappa,
            ra_units='deg', dec_units='deg'
        )

class AbacusShearMap(AbacusLensingMap):
    """
    Class for AbacusLensing shear Maps.
    """
    def __init__(self, snap_idx=47, cosmo_idx=0, phase_idx=0, sim_type='base'):
        self.map_type = 'gamma'
        super().__init__(snap_idx, cosmo_idx, phase_idx, sim_type)

    def read_map(self):
        """
        Read the convergence map from the ASDF file.
        """
        map_dir = f"AbacusSummit_{self.sim_type}_c{self.cosmo_idx:03}_ph{self.phase_idx:03}"
        f = asdf.open(BASE_DIR / map_dir / f"{self.map_type}_{self.snap_idx:05d}.asdf")
        self.header = f['header']
        self.data = f['data']
        self.gamma1 = hp.ma(f['data']['gamma1'], badval=0)
        self.gamma2 = hp.ma(f['data']['gamma2'], badval=0)
        self.gamma1.mask = self.gamma1 == 0
        self.gamma2.mask = self.gamma2 == 0

    def to_treecorr(self):
        """
        Convert the map to a TreeCorr catalog.
        """
        import treecorr
        self.treecorr = treecorr.Catalog(
            ra=self.ra, dec=self.dec,
            g1=self.gamma1, g2=self.gamma2,
            ra_units='deg', dec_units='deg'
        )


if __name__ == "__main__":
    kappa = AbacusConvergenceMap(snap_idx=45, cosmo_idx=0, phase_idx=201, sim_type='huge')
    kappa.plot_mollview(save_fn="kappa_map.png")
    kappa.sample_mask()
    kappa.to_treecorr()

    gamma = AbacusShearMap(snap_idx=45, cosmo_idx=0, phase_idx=201, sim_type='huge')
    gamma.plot_mollview(save_fn="gamma_map.png")
    kappa.sample_mask()
    kappa.to_treecorr()