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


class MinimumSpanningTree(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge Minimum Spanning Tree.
    """
    def __init__(self, **kwargs):
        super().__init__(stat_name='mst', n_test=6*500, **kwargs)
    
    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return f'/pscratch/sd/e/epaillas/emc/v1.2/trained_models/best/{self.stat_name}/last-v25.ckpt'
    
    def compress_covariance(
        self,
        save_to: str = None,
    ) -> xarray.DataArray:
        """
        Compress the covariance array from the raw measurement files.
        
        Parameters
        ----------
        save_to : str
            Path of the directory where to save the compressed covariance and bin_values. If None, it is not saved.
            Default is None.
            
        Returns
        -------
        xarray.DataArray
            Covariance array. 
        """
        # Directories
        base_dir = Path('/pscratch/sd/k/knaidoo/ACM/MockChallenge/data_v2')
        data_fns = sorted(base_dir.glob('covariance_c000_ph*_seed0_hod000_smooth_3.0.npz'))

        y = []
        for data_fn in data_fns:
            data = np.load(data_fn)
            _y = np.concatenate(
                [data['mst1pt'], data['mst2pt'], data['end2pt'], data['mst3pt'], data['end3pt'], 
                 data['mst4pt'], data['end4pt'], data['mst5pt'], data['end5pt']]
            )
            y.append(_y)
        y = np.array(y)
        
        self.logger.info(f'Loaded covariance with shape: {y.shape}')
        
        cout = xarray.DataArray(
            data = y.reshape(y.shape[0], -1),
            coords = {
                "phase_idx": list(range(y.shape[0])),
                "bin_idx": np.arange(y.shape[1]),
            },
            attrs = {
                "sample": ["phase_idx"],
                "features": ["bin_idx"],
            },
            name = "covariance_y",
        )

        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed covariance file to {save_fn}')
        return cout
        

    def compress_data(
        self, 
        add_covariance: bool = False,
        save_to: str = None,
        cosmos: list = cosmo_list,
        n_hod: int = 300,
        phase_idx: int = 0,
        seed_idx: int = 0,
    ) -> dict:
        """
        Compress the data from raw measurement files.
        
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
        base_dir = Path('/pscratch/sd/k/knaidoo/ACM/MockChallenge/data_v2')

        y = []
        hods = {}
        for cosmo_idx in cosmos:
            self.logger.info(f'Compressing c{cosmo_idx:03}')
            handle = f'emulator_c{cosmo_idx:03}_ph000_seed0_hod*_smooth_3.0.npz'
            filenames = sorted(base_dir.glob(handle))[:n_hod]
            hods[cosmo_idx] = [int(f.stem.split('hod')[-1].split('_smooth_3.0')[0]) for f in filenames]
            self.logger.info(f'Number of HODs: {len(hods[cosmo_idx])}')
            for filename in filenames:
                data = np.load(filename, allow_pickle=True)
                _y = np.concatenate(
                    [data['mst1pt'], data['mst2pt'], data['end2pt'], data['mst3pt'], data['end3pt'], 
                     data['mst4pt'], data['end4pt'], data['mst5pt'], data['end5pt']]
                )
                y.append(_y/64) 
                # factor of 64 is because the mst was computed on 64 subcubes (i.e. cubes 
                # of approx 500 Mpc/h. These statistics were then combined but not correctly normalised.
        y = np.array(y)

        y = xarray.DataArray(
            data = y.reshape(len(cosmos), n_hod, -1),
            coords = {
                'cosmo_idx': cosmos,
                'hod_idx': list(range(n_hod)),
                'bin_idx': np.arange(y.shape[-1]),
            },
            attrs = {
                'sample': ['cosmo_idx', 'hod_idx'],
                'features': ['bin_idx'],
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
            cov_y = self.compress_covariance()
            cout = xarray.merge([cout, cov_y])
        
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed data to {save_fn}')
        return cout


    @set_plot_style
    def plot_training_set(self, save_fn: str = None):
        """
        Plot the training set for the observable.
        
        Parameters
        ----------
        save_fn : str
            Path to save the figure. If None, the figure is not saved.
            Default is None.
        """

        fig, ax = plt.subplots(figsize=(5, 4))

        for data in self.y:
            ax.plot(data, color='gray', alpha=0.5, lw=0.1)

        ax.set_xlabel('bin index')
        ax.set_ylabel('MST coefficient')

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving training set figure to {save_fn}')
            
        return fig, ax

    @set_plot_style
    def plot_observable(self, model_params: dict, save_fn: str = None):
        """
        Plot Minimum Spanning Tree predictions against data.

        Parameters
        ----------
        model_params : dict
            Dictionary of model parameters to use for the prediction.
        save_fn : str
            Filename to save the plot. If None, the plot is not saved.

        Returns
        -------
        fig, lax : matplotlib.figure.Figure, np.ndarray
            Figure and axes array of the plot.
        """
        height_ratios = [3, 1]
        figsize = (6, 1.5 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False,
            gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0.1)
        show_legend = False

        lax[-1].set_xlabel(r'$\textrm{bin index}$]', fontsize=15)
        lax[0].set_ylabel(r'$\textrm{MST coefficient}$', fontsize=15)

        bin_idx = self.bin_idx.values
        data = self.y
        model = self.get_model_prediction(model_params)

        cov = self.get_covariance_matrix(volume_factor=64)
        error = np.sqrt(np.diag(cov))

        lax[0].errorbar(bin_idx, data, error, marker='o', ms=3, ls='', 
            color=f'C0', elinewidth=1.0, capsize=None)
        lax[0].plot(bin_idx, model, ls='-', color=f'C1')
        lax[1].plot(bin_idx, (data - model) / error, ls='-', color=f'C0')

        for offset in [-2, 2]: lax[1].axhline(offset, color='k', ls='--')
        lax[1].set_ylabel(r'$\Delta \textrm{MST} / \sigma_\textrm{MST}$', fontsize=15)
        lax[1].set_ylim(-4, 4)

        for ax in lax:
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=14)
        if show_legend: lax[0].legend(fontsize=15)

        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving plot to {save_fn}')
        return fig, lax

    @set_plot_style
    def plot_covariance_set(self, save_fn: str = None):
        """
        Plot the covariance matrix for the observable.

        Parameters
        ----------
        save_fn : str
            Filename to save the plot. If None, the plot is not saved.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
            Figure and axes of the plot.
        """
        fig, ax = plt.subplots(figsize=(5, 4))

        for data in self.covariance_y:
            ax.plot(data, color='gray', alpha=0.5, lw=0.1)

        mean = np.mean(self.covariance_y, axis=0)
        ax.plot(mean, color='k', lw=1.0)

        ax.set_xlabel('bin index')
        ax.set_ylabel('MST coefficient')

        cov = np.cov(self.covariance_y, rowvar=False)
        prec = np.linalg.inv(cov)

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving training set figure to {save_fn}')
            
        return fig, ax