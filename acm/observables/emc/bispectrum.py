import xarray
import numpy as np
from pathlib import Path
from .base import BaseObservableEMC
import matplotlib.pyplot as plt
from jaxpower import read
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.plotting import set_plot_style
from acm.utils.decorators import temporary_class_state
from acm.utils.xarray import dataset_to_dict, split_vars

class GalaxyBispectrumMultipoles(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge galaxy correlation
    function multipoles.
    """
    def __init__(self, n_test=6*500, **kwargs):
        super().__init__(stat_name='bispectrum', n_test=n_test, **kwargs)
    
    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return f'/pscratch/sd/e/epaillas/emc/v1.2/trained_models/best/{self.stat_name}/last.ckpt'
    
    def compress_covariance(
        self,
        save_to: str = None,
        kmin: float = 0.016,
        kmax: float = 0.285, 
        rebin: int = 3,
        ells: list = [0, 2],
    ) -> xarray.DataArray:
        """
        Compress the covariance array from the raw measurement files.
        
        Parameters
        ----------
        save_to : str
            Path of the directory where to save the compressed covariance and bin_values. If None, it is not saved.
            Default is None.
        kmin : float
            Minimum k value to consider. Default is 0.01.
        kmax : float
            Maximum k value to consider. Default is 0.7.
        rebin : int
            Rebinning factor for the statistics. Default is 4.
        ells : list
            List of multipoles to compute the statistics for. Default is [0, 2, 4].
            
        Returns
        -------
        xarray.DataArray
            Covariance array. 
        """
        # Directories
        base_dir = Path(self.paths['measurements_dir']) / 'small' / self.stat_name
        data_fns = list(base_dir.glob('mesh3_spectrum_poles_ph*.h5')) # NOTE: File name format hardcoded !
        
        y = []
        for data_fn in data_fns:
            data = read(data_fn)
            data = data.select(k=slice(0, None, rebin)).select(k=(kmin, kmax))
            poles = [data.get(ell) for ell in ells]
            k = poles[0].coords('k')
            weights = k.prod(axis=1) / 1e5
            y.append(np.concatenate([weights * pole.value().real for pole in poles]))
        y = np.array(y)
        bin_idx = np.arange(len(k))
        
        y = xarray.DataArray(
            data = y.reshape(y.shape[0], len(ells), -1),
            coords = {
                "phase_idx": list(range(y.shape[0])),
                "ells": ells,
                "bin_idx": bin_idx,
            },
            attrs = {
                "sample": ["phase_idx"],
                "features": ["ells", "bin_idx"],
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
        kmin: float = 0.016,
        kmax: float = 0.285, 
        rebin: int = 3,
        ells: list = [0, 2],
        cosmos: list = cosmo_list,
        n_hod: int = 500,
        phase: int = 0,
        seed: int = 0,
        test_filters: dict = None,
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
        kmin : float
            Minimum k value to consider. Default is 0.01.
        kmax : float
            Maximum k value to consider. Default is 0.27.
        rebin : int
            Rebinning factor for the statistics. Default is 4.
        ells : list
            List of multipoles to compute the statistics for. Default is [0, 2, 4].
        cosmos : list
            List of cosmological parameters to use. If None, use all cosmological parameters.
            Default is None.
        n_hod : int
            Number of HOD parameters to use. Default is 100.
        phase : int, optional
            Phase index to read the data from. Default is 0.
        seed : int, optional
            Seed index to read the data from. Default is 0.
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
        base_dir = Path(self.paths['measurements_dir'],  f'base/{self.stat_name}/')
        
        y = []
        hods = {}
        for cosmo_idx in cosmos:
            hods[cosmo_idx] = []
            self.logger.info(f'Compressing c{cosmo_idx:03d}')
            handle = f'c{cosmo_idx:03d}_ph{phase:03d}/seed{seed}/mesh3_spectrum_poles_c{cosmo_idx:03d}_hod*.h5'
            filenames = sorted(base_dir.glob(handle))[:n_hod]
            hods[cosmo_idx] = [int(f.stem.split('hod')[-1]) for f in filenames]
            self.logger.info(f'Number of HODs: {len(hods[cosmo_idx])}')
            for filename in filenames:
                data = read(filename)
                data = data.select(k=slice(0, None, rebin)).select(k=(kmin, kmax))
                poles = [data.get(ell) for ell in (0, 2)]
                k = poles[0].coords('k')
                weights = k.prod(axis=1) / 1e5
                y.append(np.concatenate([weights * pole.value().real for pole in poles]))
        y = np.array(y)
        bin_idx = np.arange(len(k))
        y = xarray.DataArray(
            data = y.reshape(len(cosmos), n_hod, len(ells), -1),
            coords = {
                'cosmo_idx': cosmos,
                'hod_idx': list(range(n_hod)),
                'ells': ells,
                'bin_idx': bin_idx,
            },
            attrs = {
                'sample': ['cosmo_idx', 'hod_idx'],
                'features': ['ells', 'bin_idx'],
            },
            name = 'y',
        )
        x = self.compress_x(cosmos=cosmos, n_hod=n_hod, phase=phase, seed=seed)
        
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
    def plot_observable(self, model_params: dict, save_fn: str = None):
        """
        Plot the reconstructed galaxy bispectrum multipoles data, model, and residuals.

        Parameters
        ----------
        model_params : dict
            Dictionary of model parameters to use for the prediction.
        save_fn : str
            Filename to save the plot. If None, the plot is not saved.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, numpy.ndarray
            Figure and axes of the plot.
        """
        ells = self._dataset.y.coords['ells'].values.tolist()

        height_ratios = [max(len(ells), 3)] + [1] * len(ells)
        figsize = (6, 1.5 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False,
            gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0.1)
        show_legend = True

        for i, ell in enumerate(ells):
            lax[-1].set_xlabel(r'$\textrm{bin index}$]', fontsize=15)
            lax[0].set_ylabel(r'$k_1k_2k_3 B_\ell(k)$ [$h^3\,\mathrm{{Mpc}}^{{-3}}$]', fontsize=15)

            self.select_filters.update({'multipoles': ell})

            bin_idx = self.bin_idx.values
            data = self.y
            model = self.get_model_prediction(model_params)

            cov = self.get_covariance_matrix(volume_factor=64)
            error = np.sqrt(np.diag(cov))

            lax[0].errorbar(bin_idx, data, error, marker='o', ms=4, ls='', 
                color=f'C{i}', elinewidth=1.0, capsize=None, label=f'$\ell={ell}$')
            lax[0].plot(bin_idx, model, ls='-', color=f'C{i}')
            lax[i + 1].plot(bin_idx, (data - model) / error, ls='-', color=f'C{i}')

            for offset in [-2, 2]: lax[i + 1].axhline(offset, color='k', ls='--')
            lax[i + 1].set_ylabel(rf'$\Delta B_{{{ell:d}}} / \sigma_{{ B_{{{ell:d}}} }}$', fontsize=15)
            lax[i + 1].set_ylim(-4, 4)
        for ax in lax:
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=14)
        if show_legend: lax[0].legend(fontsize=15)

        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving plot to {save_fn}')
        return fig, lax
    
# Aliases
bispectrum = GalaxyBispectrumMultipoles