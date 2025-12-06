import xarray
import numpy as np
import glob
from pathlib import Path
from .base import BaseObservableEMC
import matplotlib.pyplot as plt
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.plotting import set_plot_style
from acm.utils.xarray import dataset_to_dict, split_vars


class DDkNN(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge 2D DD-kNN statistic
    """
    def __init__(self, **kwargs):
        super().__init__(stat_name='dd_knn', n_test=6*100, **kwargs)
        self.paths['measurements_dir'] = '/pscratch/sd/p/pd2487/knns/dd_knn_measurements/'
    
    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return f'/pscratch/sd/p/pd2487/knns/trained_models/best/{self.stat_name}/last.ckpt'
        #return f'/pscratch/sd/e/epaillas/emc/v1.2/trained_models/best/{self.stat_name}/last-v25.ckpt'

    def make_mask(self, train_y):
        pass
    
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
        base_dir = Path(self.paths['measurements_dir']) / 'small' / self.stat_name
        data_fns = list(base_dir.glob('dd_knn_ph*.npy')) # NOTE: File name format hardcoded !
        
        y = []
        for data_fn in data_fns:
            data = np.load(data_fn, allow_pickle=True)
            y.append(data)
        y = np.array(y)

        # Additional step: use the mask obtained from big boxes to filter out noisy bins
        # Check that mask is present in the class (compress_data() should have been called for that)
        if not hasattr(self, 'mask'):
            raise AttributeError('To determine covariance of kNNs, compress_data() should be called first')
        y = y.reshape(y.shape[0], -1)
        y = y[:,self.mask]
        
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
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, dataset_to_dict(cout))
            self.logger.info(f'Saving compressed covariance file to {save_fn}')
        return cout

    def compress_data(
        self, 
        add_covariance: bool = False,
        save_to: str = None,
        cosmos: list = cosmo_list,
        n_hod: int = 500,
        phase: int = 0,
        seed: int = 0,
        cdf_floor: float = 0.05,
        test_filters: dict = None,
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
        base_dir = Path(self.paths['measurements_dir'],  f'base/dd_knn')
        
        y = []
        hods = {}
        for cosmo_idx in cosmos:
            self.logger.info(f'Compressing c{cosmo_idx:03}')
            handle = f'c{cosmo_idx:03}_ph{phase:03d}/seed{seed}/dd_knn_c{cosmo_idx:03}_hod*.npy'
            filenames = sorted(base_dir.glob(handle))[:n_hod]
            hods[cosmo_idx] = [int(f.stem.split('hod')[-1]) for f in filenames]
            self.logger.info(f'Number of HODs: {len(hods[cosmo_idx])}')
            for filename in filenames:
                data = np.load(filename, allow_pickle=True)
                y.append(data)
        y = np.array(y)
        y = y.reshape(len(cosmos), n_hod, -1)

        # Additional processing step for kNNs: filter out noisy bins and flatten using fiducial cosmology
        y_fid = y[0]

        # Make a mask
        mask1     = np.mean(y_fid, axis=0) > cdf_floor
        mask2     = np.mean(y_fid, axis=0) < (1 - cdf_floor)
        self.mask = np.logical_and(mask1, mask2)

        # Apply it
        y = y[:,:,self.mask]

        # Make xarrays
        y = xarray.DataArray(
            data = y,
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
        
        self.logger.info(f'Loaded data with shape: {x.shape}, {y.shape} (after bins filtering)')
        
        cout = xarray.Dataset(
            data_vars = {
                'x': x,
                'y': y,
            },
        )

        if add_covariance:
            cov_y = self.compress_covariance()
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
        ax.set_ylabel('2D DD-kNN CDF value')

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving training set figure to {save_fn}')
            
        return fig, ax

    @set_plot_style
    def plot_observable(self, model_params: dict, save_fn: str = None):
        """
        Plot DD-kNN CDFs  predictions against data.

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
        lax[0].set_ylabel(r'$\textrm{2D DD-kNN CDF value}$', fontsize=15)

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
        lax[1].set_ylabel(r'$\Delta \textrm{DDkNN} / \sigma_\textrm{DDkNN}$', fontsize=15)
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
        ax.set_ylabel('2D DD-kNN value')

        cov = np.cov(self.covariance_y, rowvar=False)
        prec = np.linalg.inv(cov)

        if save_fn is not None:
            fig.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving training set figure to {save_fn}')
            
        return fig, ax