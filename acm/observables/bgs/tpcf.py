import xarray
import numpy as np
from pathlib import Path
from pycorr import TwoPointCorrelationFunction
from .base import BaseObservableBGS
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.xarray import dataset_to_dict

class GalaxyCorrelationFunctionMultipoles(BaseObservableBGS):
    """
    Class for the application of the Two-point correlation function statistic of the ACM pipeline 
    to the BGS dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(stat_name='tpcf', **kwargs)
        
    #%% Compressed files creation
    def compress_covariance(
        self, 
        save_to: str = None, 
        rebin: int = 1, 
        ells: list = [0, 2], 
        overwrite_s: np.ndarray = None
    ) -> xarray.DataArray:
        """
        Compress the covariance array from the raw measurement files.
        
        Parameters
        ----------
        save_to : str
            Path of the directory where to save the compressed covariance and bin_values. If None, it is not saved.
            Default is None.
        rebin : int
            Rebinning factor for the statistics. Default is 1.
        ells : list
            List of multipoles to compute the statistics for. Default is [0, 2].
        overwrite_s : np.ndarray
            If not None, overwrite the final separation values with this array. 
            This is primarily useful to ensure consistency between the covariance and the data dims.
            Default is None.
        
        Returns
        -------
        xarray.DataArray
            Covariance array. 
        """
        base_dir = Path(self.paths['measurements_dir']) / 'small' / self.stat_name
        outliers_path = base_dir / 'outliers_idx.npy' # NOTE: Hardcoded !
        outliers_phases = np.load(outliers_path)
        
        y = []
        for phase in range(3000, 5000):
            data_fn = Path(base_dir) / f'tpcf_c000_ph{phase:04}_hod096.npy' # NOTE: Hardcoded !
            if not data_fn.exists() or phase in outliers_phases:
                continue # Skip missing files or outliers
            # data = TwoPointCorrelationFunction.load(data_fn)[::rebin]
            data = np.load(data_fn, allow_pickle=True).item()[::rebin] # FIXME: remove this on next computation of measurements 
            s, multipoles = data(ells=ells, return_sep=True)
            y.append(np.concatenate(multipoles))
        y = np.array(y)
        s = overwrite_s if overwrite_s is not None else s
        
        self.logger.info(f'Loaded covariance with shape: {y.shape}')
        
        cout = xarray.DataArray(
            data = y.reshape(y.shape[0], len(ells), -1),
            coords = {
                "phase_idx": list(range(y.shape[0])),
                "ells": ells,
                "s": s,
            },
            attrs = {
                "sample": ["phase_idx"],
                "features": ["ells", "s"],
            },
            name = "covariance_y",
        )
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed covariance file to {save_fn}')
        return cout
    
    def compress_data(
        self, 
        add_covariance: bool = False,
        save_to: str = None,
        rebin: int = 1, 
        ells: list = [0, 2],
        cosmos: list = cosmo_list,
        n_hod: int = 100,
    ) -> xarray.Dataset:
        """
        Compress the data from the tpcf raw measurement files.
        
        Parameters
        ----------
        add_covariance : bool
            If True, add the covariance to the compressed data. Default is False.
        save_to : str
            Path of the directory where to save the compressed file. If None, it is not saved.
            Default is None.
        rebin : int
            Rebinning factor for the statistics. Default is 1.
        ells : list
            List of multipoles to compute the statistics for. Default is [0, 2].
        cosmos : list
            List of cosmological parameters to use. If None, use all cosmological parameters.
            Default is cosmo_list.
        n_hod : int
            Number of HOD parameters to use. Default is 100.
            
        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'x' and 'y' DataArrays. 
            If add_covariance is True, also contains 'covariance_y' DataArray.
        """
        base_dir = Path(self.paths['measurements_dir']) / 'base' / self.stat_name
        
        y = []
        for cosmo_idx in cosmos:
            for hod in range(n_hod):
                data_fn = Path(base_dir) / f'{self.stat_name}_c{cosmo_idx:03d}_hod{hod:03}.npy' # NOTE: Hardcoded !
                # data = TwoPointCorrelationFunction.load(data_fn)[::rebin]
                data = np.load(data_fn, allow_pickle=True).item()[::rebin] # FIXME: remove this on next computation of measurements 
                s, multipoles = data(ells=ells, return_sep=True)
                y.append(np.concatenate(multipoles))
        y = np.array(y)
        y = xarray.DataArray(
            data = y.reshape(len(cosmos), n_hod, len(ells), -1),
            coords = {
                'cosmo_idx': cosmos,
                'hod_idx': list(range(n_hod)),
                'ells': ells,
                's': s,
            },
            attrs = {
                'sample': ['cosmo_idx', 'hod_idx'],
                'features': ['ells', 's'],
            },
            name = 'y',
        )
        x = self.compress_x(cosmos=cosmos, n_hod=n_hod)
        
        self.logger.info(f'Loaded data with shape: {x.shape}, {y.shape}')

        cout = xarray.Dataset(
            data_vars = {
                'x': x,
                'y': y,
            },
        )
        if add_covariance:
            cov_y = self.compress_covariance(rebin=rebin, ells=ells, overwrite_s=s)
            cout = xarray.merge([cout, cov_y])
        
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed data to {save_fn}')
        return cout