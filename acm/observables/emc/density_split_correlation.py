import xarray
import numpy as np
from pathlib import Path
from .base import BaseObservableEMC
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.xarray_data import dataset_to_dict

class DensitySplitCorrelationFunctionMultipoles(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge density-split correlation
    function multipoles.
    """
    def __init__(self, **kwargs):
        super().__init__(stat_name='dsc_conf', **kwargs)
        self.n_test = 6*100 #Override default number of test samples 
    
    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return '/pscratch/sd/e/epaillas/emc/trained_models/dsc_conf/cosmo+hod/aug9/last-v1.ckpt' #FIXME: Update this path to the correct checkpoint file format
    
    def compress_covariance(
        self, 
        save_to: str = None, 
        rebin: int = 4, 
        ells: list = [0, 2],
        quantiles: list = [0, 1, 3, 4],
        statistics: list = ['quantile_data_correlation', 'quantile_correlation'],
        overwrite_s : np.ndarray = None,
    ):
        """
        Compress the covariance array from the raw measurement files.
        
        Parameters
        ----------
        save_to : str
            Path of the directory where to save the compressed covariance and bin_values. If None, it is not saved.
            Default is None.
        rebin : int
            Rebinning factor for the statistics. Default is 4.
        ells : list
            List of multipoles to compute the statistics for. Default is [0, 2, 4].
        quantiles : list
            List of quantiles to compute the statistics for. Default is [0, 1, 3, 4].
        statistics : list
            List of statistics to compute the statistics for. Used in the filenames.
            Default is ['quantile_data_correlation', 'quantile_correlation'].
        overwrite_s : np.ndarray
            If not None, overwrite the final separation values with this array. 
            This is primarily useful to ensure consistency between the covariance and the data dims.
            Default is None.
            
        Returns
        -------
        xarray.DataArray
            Covariance array. 
        """
        base_dir = self.paths['covariance_statistic_dir']
        
        y = []
        for phase in range(3000, 5000):
            multipoles_stat = []
            for stat in statistics:
                data_dir = Path(base_dir) / f'{stat}/z0.5/yuan23_prior/' # NOTE: Hardcoded !
                data_fn = data_dir / f'{stat}_ph{phase:03}_hod466.npy' # NOTE: Hardcoded !
                if not data_fn.exists():
                    break
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
            data = y.reshape(y.shape[0], len(statistics), len(quantiles), len(ells), -1),
            coords = {
                "phase_idx": list(range(y.shape[0])),
                "statistics": statistics,
                "quantiles": quantiles,
                "multipoles": ells,
                "s": s,
            },
            attrs = {
                "sample": ["phase_idx"],
                "features": ["statistics", "quantiles", "multipoles", "s"],
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
        rebin: int = 4, 
        ells: list = [0, 2, 4],
        quantiles: list = [0, 1, 3, 4],
        statistics: list = ['quantile_data_correlation', 'quantile_correlation'],
        cosmos: list = cosmo_list,
        n_hod: int = 100,
        phase_idx: int = 0,
        seed_idx: int = 0,
    ):
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
            Default is None.
        n_hod : int
            Number of HOD parameters to use. Default is 100.
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
        base_dir = self.paths['statistic_dir']
        
        y = []
        for cosmo_idx in cosmos:
            data_dir = base_dir + f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx:01}/'
            for hod_idx in range(n_hod):
                multipoles_stat = []
                for stat in statistics:
                    data_fn = Path(data_dir) / f'{stat}_hod{hod_idx:03}.npy' # NOTE: File name format hardcoded !
                    data = np.load(data_fn, allow_pickle=True)
                    multipoles_quantiles = []
                    for q in quantiles:
                        result = data[q][::rebin]
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
                'statistics': statistics,
                'quantiles': quantiles,
                'multipoles': ells,
                's': s,
            },
            attrs = {
                "sample": ["cosmo_idx", "hod_idx"],
                "features": ["statistics", "quantiles", "multipoles", "s"],
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