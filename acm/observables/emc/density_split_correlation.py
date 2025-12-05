import xarray
import numpy as np
from pathlib import Path
from .base import BaseObservableEMC
import matplotlib.pyplot as plt
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.plotting import set_plot_style
from acm.utils.decorators import temporary_class_state
from acm.utils.xarray import dataset_to_dict, split_vars


class DensitySplitBaseClass(BaseObservableEMC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def compress_covariance(
        self, 
        save_to: str = None,
        smin: float = 0.0,
        smax: float = 150,
        rebin: int = 4, 
        ells: list = [0, 2],
        quantiles: list = [0, 1, 3, 4],
        overwrite_s : np.ndarray = None,
    ):
        """
        Compress the covariance array from the raw measurement files.
        
        Parameters
        ----------
        save_to : str
            Path of the directory where to save the compressed covariance and bin_values. If None, it is not saved.
            Default is None.
        smin : float
            Minimum separation value to consider, in Mpc/h. Default is 0.0.
        smax : float
            Maximum separation value to consider, in Mpc/h. Default is 150.
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
        # Directories
        base_dir = Path(self.paths['measurements_dir']) / 'small' / 'density_split'
        data_fns = list(base_dir.glob(f'{self.measurement_root}_poles_ph*.npy')) # NOTE: File name format hardcoded !
        n_sims = len(data_fns)
        
        y = []
        for data_fn in data_fns:
            data = np.load(data_fn, allow_pickle=True)
            for q in quantiles:
                result = data[q][::rebin].select((smin, smax))
                s, multipoles = result(ells=ells, return_sep=True)
                y.append(np.concatenate(multipoles))
        y = np.array(y)
        y = y.reshape(n_sims, len(quantiles), len(ells), -1)
        s = overwrite_s if overwrite_s is not None else s
        
        y = xarray.DataArray(
            data = y,
            coords = {
                "phase_idx": list(range(y.shape[0])),
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
        add_covariance: bool = False,
        save_to: str = None,
        rebin: int = 4,
        smin: float = 0.0,
        smax: float = 150,
        ells: list = [0, 2],
        quantiles: list = [0, 1, 3, 4],
        cosmos: list = cosmo_list,
        n_hod: int = 100,
        phase_idx: int = 0,
        seed_idx: int = 0,
        test_filters: dict = None,
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
        test_filters : dict, optional
            Dictionary of filters to split the dataset into training and test sets.
            Keys are the dimension names and values are the values to filter on for the test set.
            If None, no splitting is done. Default is None.
            
        Returns
        -------
        xarray.Dataset
            Compressed dataset containing 'x' and 'y' DataArrays.
            If add_covariance is True, also contains 'covariance_y' DataArray.
        """
        base_dir = Path(self.paths['measurements_dir']) / 'base' / 'density_split'
        
        y = []
        hods = {}
        for cosmo_idx in cosmos:
            self.logger.info(f'Compressing c{cosmo_idx:03}')
            handle = f'c{cosmo_idx:03}_ph{phase_idx:03d}/seed{seed_idx}/{self.measurement_root}_poles_c{cosmo_idx:03}_hod*.npy'
            filenames = sorted(base_dir.glob(handle))[:n_hod]
            hods[cosmo_idx] = [int(f.stem.split('hod')[-1]) for f in filenames]
            self.logger.info(f'Number of HODs: {len(hods[cosmo_idx])}')
            for filename in filenames:
                data = np.load(filename, allow_pickle=True)
                for q in quantiles:
                    result = data[q][::rebin]
                    result.select((smin, smax))
                    s, multipoles = result(ells=ells, return_sep=True)
                    y.append(np.concatenate(multipoles))
        y = np.array(y)
        y = xarray.DataArray(
            data = y.reshape(len(cosmos), n_hod, len(quantiles), len(ells), -1),
            coords = {
                'cosmo_idx': cosmos,
                'hod_idx': list(range(n_hod)),
                'quantiles': quantiles,
                'ells': ells,
                's': s,
            },
            attrs = {
                "sample": ["cosmo_idx", "hod_idx"],
                "features": ["quantiles", "ells", "s"],
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
            cov_y = self.compress_covariance(rebin=rebin, ells=ells, quantiles=quantiles, overwrite_s=s)
            cout = xarray.merge([cout, cov_y])
            
        if test_filters is not None:
            for v_in, v_out in split_vars(cout.x, cout.y, **test_filters):
                v_in.name = v_in.name + '_test'
                v_out.name = v_out.name + '_train'
                v_in.attrs['nan_dims'] = list(test_filters.keys()) # Mark filtered dimensions that will be filled with NaNs
                v_out.attrs['nan_dims'] = list(test_filters.keys())
                cout = xarray.merge([cout, v_in, v_out])
        
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed data to {save_fn}')
        return cout

    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_training_set(self, save_fn: str = None):
        ells = self._dataset.y.coords['ells'].values.tolist()
        quantiles = self._dataset.y.coords['quantiles'].values.tolist()

        fig, lax = plt.subplots(len(ells), 1, figsize=(4, 5), sharex=True)

        for ell in ells:
            self.select_filters.update({'ells': ell})
            s = self.s

            for i, quantile in enumerate(quantiles):
                self.select_filters.update({'quantiles': quantile})

                for data in self.y:
                    lax[ell//2].plot(s, s**2 * data, ls='-', color=f'C{i}', lw=0.1, alpha=0.5)

            lax[ell//2].set_ylabel(r'$s^2\xi_{\ell}(s)\,[h^{-2}{\rm Mpc}^2]$')
        lax[-1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$')

        plt.tight_layout()
        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving plot to {save_fn}')
        return fig, lax

class DensitySplitQuantileGalaxyCorrelationFunctionMultipoles(DensitySplitBaseClass):
    """
    Class for the Emulator's Mock Challenge density-split correlation
    function multipoles.
    """
    def __init__(self, n_test=6*200, **kwargs):
        super().__init__(stat_name='ds_xiqg', n_test=n_test, **kwargs)
        self.measurement_root = 'dsc_xiqg'
    
    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return f'/pscratch/sd/e/epaillas/emc/v1.2/trained_models/best/{self.stat_name}/last.ckpt'
    
class DensitySplitGalaxyCorrelationFunctionMultipoles(DensitySplitBaseClass):
    """
    Class for the application of the densitysplit auto-correlation statistic of the ACM pipeline to the BGS dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(stat_name='ds_xigg', **kwargs)
        self.measurement_root = 'dsc_xigg'
        
        
# Aliases
ds_xiqg = DensitySplitQuantileGalaxyCorrelationFunctionMultipoles
ds_xigg = DensitySplitGalaxyCorrelationFunctionMultipoles