import xarray
import numpy as np
from pathlib import Path
from .base import BaseObservableBGS
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.xarray_data import dataset_to_dict

class DensitySplitBaseClass(BaseObservableBGS):
    """
    Base class for densitysplit observables in the ACM pipeline for the BGS dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    #%% Compressed files creation
    def compress_covariance(
        self, 
        cosmo_idx: int = 0,
        hod_idx: int = 157,
        seed: int = 0,
        los: list[str] = ['x', 'y', 'z'],
        save_to: str = None, 
        rebin: int = 1, 
        ells: list = [0, 2], 
        quantiles: list = [0, 1, 3, 4],
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
        los : list[str]
            List of line-of-sight directions to use. Default is ['x', 'y', 'z'].
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
        small_dir = Path(self.paths['measurements_dir']) / 'small' 
        
        y = []
        phases = [int(fn.stem.split('_ph')[-1]) for fn in sorted(small_dir.glob(f'c{cosmo_idx:03d}_ph*'))] # List of available phases from files
        
        for phase in phases:
            y_quantiles = []
            for q in quantiles:
                fn_dir = small_dir / f'c{cosmo_idx:03d}_ph{phase:03d}' / f'seed{seed}' / f'hod{hod_idx:03d}'
                fns = [fn_dir / f'{self.measurement_root}_los_{l}.npy' for l in los] # NOTE: Hardcoded !
                existing_fns = [fn for fn in fns if fn.exists()]
                if len(existing_fns) == 0:
                    # NOTE: This will crash the process later, but at least we log it
                    self.logger.warning(f'No measurement files found in {fn_dir} for quantile {q}, skipping.')
                    continue
                data = sum([np.load(fn, allow_pickle=True)[q].normalize() for fn in existing_fns])
                s, multipoles = data[::rebin](ells=ells, return_sep=True)
                y_quantiles.append(multipoles)
            y.append(y_quantiles)
        y = np.array(y)
        s = overwrite_s if overwrite_s is not None else s
        
        y = xarray.DataArray(
            data = y.reshape(len(phases), len(quantiles), len(ells), -1),
            coords = {
                "phase_idx": phases, # TODO: continuous phase indexing ?
                "quantiles": quantiles,
                "ells": ells,
                "s": s,
            },
            attrs = {
                "sample": ["phase_idx"],
                "features": ["quantiles", "ells", "s"],
            },
            name = "covariance_y",
        )
        
        self.logger.info(f'Loaded covariance with shape: {y.shape}')
        
        cout = xarray.Dataset(data_vars = {'covariance_y': y})
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
        los: list[str] = ['x', 'y', 'z'],
        rebin: int = 1, 
        ells: list = [0, 2],
        quantiles: list = [0, 1, 3, 4],
        cosmos: list = cosmo_list,
        n_hod: int = None,
        density_threshold: float = None,
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
        los : list[str]
            List of line-of-sight directions to use. Default is ['x', 'y', 'z'].
        rebin : int
            Rebinning factor for the statistics. Default is 1.
        ells : list
            List of multipoles to compute the statistics for. Default is [0, 2].
        quantiles : list
            List of quantiles to compute the statistics for. Default is [0, 1, 3, 4].
        cosmos : list
            List of cosmological parameters to use. If None, use all cosmological parameters.
            Default is cosmo_list.
        n_hod : int, optional
            Number of HODs to consider per cosmology. 
            If None, it is determined from the first cosmology and restricted to that number for all cosmologies. 
            Defaults to None.
        density_threshold : float, optional
            Density threshold to use for the HOD mocks selection. If None, use all available HOD mocks.
            Default is None.
        **kwargs
            Extra arguments to pass to `compress_covariance` (`cosmo_idx` or `hod_idx`) if needed.
            See their documentation for details and default values.
            
        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'x' and 'y' DataArrays. 
            If add_covariance is True, also contains 'covariance_y' DataArray.
        """
        x = self.compress_x(cosmos=cosmos, phase=phase, seed=seed, n_hod=n_hod)
        n_hod = len(x.hod_idx) # Edge case if n_hod was None
        
        y = []
        for cosmo_idx in cosmos:
            # Get the HODs folders available for this cosmology
            hod_fns = self.get_hod_from_files(
                cosmo_idx, 
                phase=phase, 
                seed=seed, 
                density_threshold=density_threshold, 
                return_fn=True,
            )[:n_hod] # Restrict to n_hod if needed
            
            for fn_dir in hod_fns:
                y_quantiles = []
                for q in quantiles:
                    fns = [fn_dir / f'{self.measurement_root}_los_{l}.npy' for l in los] # NOTE: Hardcoded !
                    data = sum([np.load(fn, allow_pickle=True)[q].normalize() for fn in fns if fn.exists()])
                    if data == 0:
                        # NOTE: This will crash the process later, but at least we log it
                        self.logger.warning(f'No measurement files found in {fn_dir} for quantile {q}, skipping.')
                        continue
                    s, multipoles = data[::rebin](ells=ells, return_sep=True)
                    y_quantiles.append(multipoles)
                y.append(y_quantiles)
        y = np.array(y)
        
        y = xarray.DataArray(
            # NOTE: Should crash if n_hod is not consistent with the hod number from the statistics, this is intended
            data = y.reshape(len(cosmos), n_hod, len(quantiles), len(ells), -1),
            coords = {
                'cosmo_idx': cosmos,
                'hod_idx': list(range(n_hod)), # re-index HODs to be continuous
                'quantiles': quantiles,
                'ells': ells,
                's': s,
            },
            attrs = {
                'sample': ['cosmo_idx', 'hod_idx'],
                'features': ['quantiles', 'ells', 's'],
            },
            name = 'y',
        )
        
        self.logger.info(f'Loaded data with shape: {x.shape}, {y.shape}')

        cout = xarray.Dataset(
            data_vars = {
                'x': x,
                'y': y,
            },
        )
        if add_covariance:
            cov_y = self.compress_covariance(rebin=rebin, ells=ells, overwrite_s=s, seed=seed, los=los, **kwargs)
            cout = xarray.merge([cout, cov_y])
        
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed data to {save_fn}')
        return cout


class DensitySplitQuantileGalaxyCorrelationFunctionMultipoles(DensitySplitBaseClass):
    """
    Class for the application of the densitysplit cross-correlation statistic of the ACM pipeline to the BGS dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(stat_name='ds_xiqg', **kwargs)
        self.measurement_root = 'quantile_data_correlation'


class DensitySplitGalaxyCorrelationFunctionMultipoles(DensitySplitBaseClass):
    """
    Class for the application of the densitysplit auto-correlation statistic of the ACM pipeline to the BGS dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(stat_name='ds_xigg', **kwargs)
        self.measurement_root = 'quantile_correlation'

# Aliases
ds_xigg = DensitySplitGalaxyCorrelationFunctionMultipoles
ds_xiqg = DensitySplitQuantileGalaxyCorrelationFunctionMultipoles