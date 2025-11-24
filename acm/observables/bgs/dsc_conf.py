import xarray
import numpy as np
from pathlib import Path
from .base import BaseObservableBGS
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.xarray import dataset_to_dict

class DensitySplitCorrelationFunctionMultipoles(BaseObservableBGS):
    """
    Class for the application of the densitysplit statistic of the ACM pipeline to the BGS dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(stat_name='dsc_conf', **kwargs)

    #%% Compressed files creation
    def compress_covariance(
        self, 
        save_to: str = None, 
        rebin: int = 1, 
        ells: list = [0, 2],
        quantiles: list = [0, 1, 3, 4],
        overwrite_s : np.ndarray = None,
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
        quantiles : list
            List of quantiles to compute the statistics for. Default is [0, 1, 3, 4].
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
            multipoles_stat = []
            for stat in ['ccf', 'acf']: # FIXME: Change this in the next computation !
                data_fn = Path(base_dir) / f'{stat}_c000_ph{phase:03}_hod096.npy' # NOTE: Hardcoded !
                if not data_fn.exists() or phase in outliers_phases:
                    break # Skip missing files or outliers
                data = np.load(data_fn, allow_pickle=True)
                multipoles_quantiles = []
                for q in quantiles:
                    result = data[q][::rebin]
                    s, multipoles = result(ells=ells, return_sep=True)
                    multipoles_quantiles.append(np.concatenate(multipoles))
                multipoles_stat.append(np.concatenate(multipoles_quantiles))
            else: # If the loop is not broken
                y.append(np.concatenate(multipoles_stat))
        y = np.array(y)
        s = overwrite_s if overwrite_s is not None else s
        
        self.logger.info(f'Loaded covariance with shape: {y.shape}')
        
        cout = xarray.DataArray(
            data = y.reshape(y.shape[0], 2, len(quantiles), len(ells), -1),
            coords = {
                "phase_idx": list(range(y.shape[0])),
                "statistics": ['quantile_data_correlation', 'quantile_correlation'],
                "quantiles": quantiles,
                "ells": ells,
                "s": s,
            },
            attrs = {
                "sample": ["phase_idx"],
                "features": ["statistics", "quantiles", "ells", "s"],
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
        quantiles: list = [0, 1, 3, 4],
        cosmos: list = cosmo_list,
        n_hod: int = 100,
    ) -> xarray.Dataset:
        """
        Compress the data from the densitysplit raw measurement files.
        
        Parameters
        ----------
        add_covariance : bool
            If True, add the covariance to the compressed data. Default is False.
        save_to : str
            Path of the directory where to save the compressed file. If None, it is not saved.
            Default is None.
        rebin : int
            Rebinning factor for the statistics. Default is 4.
        ells : list
            List of multipoles to compute the statistics for. Default is [0, 2, 4].
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
                multipoles_stat = []
                for stat in ['ccf', 'acf']: # NOTE: Hardcoded !
                    data_fn = Path(base_dir) / f'{stat}_c{cosmo_idx:03d}_hod{hod:03}.npy' # NOTE: Hardcoded !
                    data = np.load(data_fn, allow_pickle=True)
                    multipoles_quantiles = []
                    for q in quantiles:
                        result = data[q]
                        s, multipoles = result(ells=ells, return_sep=True)
                        multipoles_quantiles.append(np.concatenate(multipoles))
                    multipoles_stat.append(np.concatenate(multipoles_quantiles))
                y.append(np.concatenate(multipoles_stat))
        y = np.array(y)
        y = xarray.DataArray(
            data = y.reshape(len(cosmos), n_hod, 2, len(quantiles), len(ells), -1),
            coords = {
                'cosmo_idx': cosmos,
                'hod_idx': list(range(n_hod)),
                'statistics': ['quantile_data_correlation', 'quantile_correlation'],
                'quantiles': quantiles,
                'ells': ells,
                's': s,
            },
            attrs = {
                "sample": ["cosmo_idx", "hod_idx"],
                "features": ["statistics", "quantiles", "ells", "s"],
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
            cov_y = self.compress_covariance(rebin=rebin, ells=ells, quantiles=quantiles, overwrite_s=s)
            cout = xarray.merge([cout, cov_y])
            
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed data to {save_fn}')
        return cout