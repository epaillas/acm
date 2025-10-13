import xarray
import numpy as np
from pathlib import Path
from pycorr import TwoPointCorrelationFunction
from .base import BaseObservableBGS
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.xarray_data import dataset_to_dict

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
        cosmo_idx: int = 0,
        hod_idx: int = 96,
        seed: int = 0,
        save_to: str = None, 
        rebin: int = 1, 
        ells: list = [0, 2], 
        overwrite_s: np.ndarray = None
    ) -> xarray.DataArray:
        """
        Compress the covariance array from the raw measurement files.
        
        Parameters
        ----------
        cosmo_idx : int
            Index of the cosmology to use. Default is 0.
        hod_idx : int
            Index of the HOD to use. Default is 96.
        seed : int
            Seed index to use. Default is 0.
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
        base_dir = Path(self.paths['measurements_dir']) / 'small' 
        
        # NOTE : this is kept there just in case, but should not be used anymore, if next run works fine, will be removed (TODO)
        outliers_path = base_dir / 'outliers_idx.npy' # NOTE: Hardcoded !
        if outliers_path.exists():
            outliers_phases = np.load(outliers_path)
            self.logger.warning(f'Excluding outlier phases: {outliers_phases}')
        else:
            outliers_phases = []
        
        y = []
        phases = [int(fn.stem.split('_ph')[-1]) for fn in sorted(base_dir.glob(f'c{cosmo_idx:03d}_ph*'))]
        for phase in phases:
            data_fn = Path(base_dir) / f'c{cosmo_idx:03d}_ph{phase:03d}' / f'seed{seed}' / self.stat_name / f'hod{hod_idx:03d}.npy' # NOTE: Hardcoded !
            if not data_fn.exists() or phase in outliers_phases:
                continue # Skip missing files or outliers
            data = TwoPointCorrelationFunction.load(data_fn)[::rebin]
            s, multipoles = data(ells=ells, return_sep=True)
            y.append(np.concatenate(multipoles))
        y = np.array(y)
        s = overwrite_s if overwrite_s is not None else s
        
        self.logger.info(f'Loaded covariance with shape: {y.shape}')
        
        cout = xarray.DataArray(
            data = y.reshape(y.shape[0], len(ells), -1),
            coords = {
                "phase_idx": list(range(y.shape[0])),
                "multipoles": ells,
                "s": s,
            },
            attrs = {
                "sample": ["phase_idx"],
                "features": ["multipoles", "s"],
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
        phase: int = 0,
        seed: int = 0,
        add_covariance: bool = False,
        save_to: str = None,
        rebin: int = 1, 
        ells: list = [0, 2],
        cosmos: list = cosmo_list,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Compress the data from the tpcf raw measurement files.
        
        Parameters
        ----------
        phase : int, optional
            Phase index to read the data from. Default is 0.
        seed : int, optional
            Seed index to read the data from. Default is 0.
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
        **kwargs
            Extra arguments to pass to `compress_covariance` (`cosmo_idx` or `hod_idx`), or to `compress_x` (`statistic`) if needed.
            See their documentation for details and default values.
            
        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'x' and 'y' DataArrays. 
            If add_covariance is True, also contains 'covariance_y' DataArray.
        """
        base_dir = Path(self.paths['measurements_dir']) / 'base'
        
        statistic = kwargs.pop('statistic', 'density') # To avoid conflict with the arguments of compress_covariance
        x = self.compress_x(cosmos=cosmos, phase=phase, seed=seed, statistic=statistic)
        n_hod = len(x.hod_idx)
        
        y = []
        for cosmo_idx in cosmos:
            hod_idx = self.get_raw_hod_idx(cosmo_idx, phase=phase, seed=seed, statistic=statistic) # Get the HODs available for this cosmology
            for hod in hod_idx:
                data_fn = Path(base_dir) / f'c{cosmo_idx:03d}_ph{phase:03d}' / f'seed{seed}' / self.stat_name / f'hod{hod:03d}.npy' # NOTE: Hardcoded !
                data = TwoPointCorrelationFunction.load(data_fn)[::rebin]
                s, multipoles = data(ells=ells, return_sep=True)
                y.append(np.concatenate(multipoles))
        y = np.array(y)
        y = xarray.DataArray(
            # NOTE: Should crash if n_hod is not consistent with the hod number from the statistics, this is intended
            data = y.reshape(len(cosmos), n_hod, len(ells), -1),
            coords = {
                'cosmo_idx': cosmos,
                'hod_idx': list(range(n_hod)), # re-index HODs to be continuous
                'multipoles': ells,
                's': s,
            },
            attrs = {
                'sample': ['cosmo_idx', 'hod_idx'],
                'features': ['multipoles', 's'],
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
            cov_y = self.compress_covariance(rebin=rebin, ells=ells, overwrite_s=s, seed=seed, **kwargs)
            cout = xarray.merge([cout, cov_y])
        
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed data to {save_fn}')
        return cout