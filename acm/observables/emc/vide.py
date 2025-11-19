import xarray
import numpy as np
import glob
from pathlib import Path
from .base import BaseObservableEMC
import matplotlib.pyplot as plt
from jaxpower import read
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.xarray_data import dataset_to_dict
from acm.utils.plotting import set_plot_style


class VIDEVoidGalaxyCorrelationFunctionMultipoles(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge galaxy correlation
    function multipoles.
    """
    def __init__(self, **kwargs):
        super().__init__(stat_name='vide_ccf', **kwargs)
    
    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return 'none'
    
    def compress_covariance(
        self,
        save_to: str = None,
        ells: list = [0, 2],
    ) -> xarray.DataArray:
        """
        Compress the covariance array from the raw measurement files.
        
        Parameters
        ----------
        save_to : str
            Path of the directory where to save the compressed covariance and bin_values. If None, it is not saved.
            Default is None.
        ells : list
            List of multipoles to compute the statistics for. Default is [0, 2, 4].
            
        Returns
        -------
        xarray.DataArray
            Covariance array. 
        """
        raise NotImplementedError("Covariance compression not implemented yet for VIDE CCF.")

    def compress_data(
        self, 
        add_covariance: bool = False,
        save_to: str = None,
        cosmos: list = cosmo_list,
        n_hod: int = 500,
        ells: list = [0, 2, 4],
        phase_idx: int = 0,
        seed_idx: int = 0,
    ) -> dict:
        """
        Compress the data from the tpcf raw measurement files.
        
        Parameters
        ----------
        add_covariance : bool
            If True, add the covariance to the compressed data. Default is False.
        save_to : str
            Path of the directory where to save the compressed file. If None, it is not saved.
            Default is None.
        cosmos : list
            List of cosmological parameters to use. If None, use all cosmological parameters.
            Default is None.
        n_hod : int
            Number of HOD parameters to use. Default is 100.
        ells : list
            List of multipoles to compute the statistics for. Default is [0, 2, 4].
        phase_idx : int
            TODO
        seed_idx : int
            TODO
            
        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'x' and 'y' DataArrays. 
            If add_covariance is True, also contains 'covariance_y' DataArray.
        """
        base_dir = Path(self.paths['measurements_dir'],  f'base/vide/')

        filename = base_dir / 'multipoles_85cosmologies_100HODs_4bins_0.0-1.0_rv0.3-2.5.npz'
        data = np.load(filename, allow_pickle=True)

        # read multipole data
        xi0 = data['AP_Mock_monopole_runs']
        xi2 = data['AP_Mock_quadrupole_runs']
        xi4 = data['AP_Mock_hexadecapole_runs']

        n_stacked_bins = 4

        xi0 = xi0.reshape(len(xi0), n_stacked_bins, -1)
        xi2 = xi2.reshape(len(xi2), n_stacked_bins, -1)
        xi4 = xi4.reshape(len(xi4), n_stacked_bins, -1)

        # concatenate along the last dimension
        y = np.concatenate([xi0, xi2, xi4], axis=-1)

        rv = data['r_bin_centers']

        # get hod indices
        hods = {}
        for cosmo_idx in cosmos:
            hods[cosmo_idx] = self.get_raw_hod_idx(cosmo_idx)[:n_hod]

        y = xarray.DataArray(
            data = y.reshape(len(cosmos), n_hod, n_stacked_bins, len(ells), -1),
            coords = {
                'cosmo_idx': cosmos,
                'hod_idx': list(range(n_hod)),
                'stacked_bins': list(range(n_stacked_bins)),
                'multipoles': ells,
                'rv': rv,
            },
            attrs = {
                'sample': ['cosmo_idx', 'hod_idx'],
                'features': ['stacked_bins', 'multipoles', 'rv'],
            },
            name = 'y',
        )
        x = self.compress_x(hods=hods, cosmos=cosmos)
        
        self.logger.info(f'Loaded data with shape: {x.shape}, {y.shape}')
        
        cout = xarray.Dataset(
            data_vars = {
                'x': x,
                'y': y,
            },
        )
        if add_covariance:
            cov_y = self.compress_covariance(rebin=rebin, ells=ells)
            cout = xarray.merge([cout, cov_y])
        
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed data to {save_fn}')
        return cout