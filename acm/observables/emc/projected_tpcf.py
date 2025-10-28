import xarray
import numpy as np
import glob
from pathlib import Path
from .base import BaseObservableEMC
from acm.utils.default import cosmo_list # List of cosmologies in AbacusSummit
from acm.utils.xarray_data import dataset_to_dict
from acm.utils.plotting import set_plot_style

class ProjectedGalaxyCorrelationFunction(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge galaxy correlation
    function multipoles.
    """
    def __init__(self, **kwargs):
        super().__init__(stat_name='projected_tpcf', n_test=6*200, **kwargs)
        self.paths['statistic_dir'] = f'/pscratch/sd/e/epaillas/emc/training_sets/spectrum/cosmo+hod_bugfix/z0.5/yuan23_prior/'
        self.paths['statistic_covariance_dir'] = f'/pscratch/sd/e/epaillas/emc/covariance_sets/tpcf/z0.5/yuan23_prior/'
    
    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return f'/pscratch/sd/e/epaillas/emc/v1.2/trained_models/best/{self.stat_name}/last.ckpt'
    
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
        from pycorr import TwoPointCorrelationFunction
        # Directories
        base_dir = Path(self.paths['measurements_dir']) / 'small' / self.stat_name
        data_fns = list(base_dir.glob('tpcf_rppi_ph*.npy')) # NOTE: File name format hardcoded !
        
        y = []
        for data_fn in data_fns:
            data = TwoPointCorrelationFunction.load(data_fn)
            r_p, w_p = data(pimax=None, return_sep=True)
            y.append(w_p)
        y = np.array(y)
        
        self.logger.info(f'Loaded covariance with shape: {y.shape}')
        
        cout = xarray.DataArray(
            data = y.reshape(y.shape[0], -1),
            coords = {
                "phase_idx": list(range(y.shape[0])),
                "r_p": r_p,
            },
            attrs = {
                "sample": ["phase_idx"],
                "features": ["r_p"],
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
        from pycorr import TwoPointCorrelationFunction
        base_dir = Path(self.paths['measurements_dir'],  f'base/{self.stat_name}/')
        
        y = []
        hods = {}
        for cosmo_idx in cosmos:
            hods[cosmo_idx] = []
            self.logger.info(f'Compressing c{cosmo_idx:03}')
            handle = f'c{cosmo_idx:03}_ph000/seed0/tpcf_rppi_c{cosmo_idx:03}_hod???.npy'
            filenames = sorted(base_dir.glob(handle))[:n_hod]
            for filename in filenames:
                data = TwoPointCorrelationFunction.load(filename)
                r_p, w_p = data(pimax=None, return_sep=True)
                y.append(w_p)
                hod_idx = int(str(filename).split('hod')[1].split('.')[0])
                hods[cosmo_idx].append(hod_idx)
            self.logger.info(f'HOD indices: {hods[cosmo_idx]}')
        y = np.array(y)
        y = xarray.DataArray(
            data = y.reshape(len(cosmos), n_hod, -1),
            coords = {
                'cosmo_idx': cosmos,
                'hod_idx': list(range(n_hod)),
                'r_p': r_p,
            },
            attrs = {
                'sample': ['cosmo_idx', 'hod_idx'],
                'features': ['r_p'],
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
    
    def compute_phase_correction(self, rebin: int = 4, ells: list = [0, 2, 4]):
        """
        Correction factor to bring the fixed phase precictions (p000) to the ensemble average.
        
        Parameters
        ----------
        rebin : int
            Rebinning factor for the statistics. Default is 4.
        ells : list
            List of multipoles to compute the correction for. Default is [0, 2, 4].
        
        Returns
        -------
        np.ndarray
            Correction factor for the fixed phase predictions.
        """
        from pathlib import Path
        import numpy as np
        from pycorr import TwoPointCorrelationFunction
        
        base_dir = self.paths['measurements_dir'] + f'base/{self.stat_name}/'
        # base_dir = '/pscratch/sd/e/epaillas/emc/training_sets/tpcf/cosmo+hod_bugfix/z0.5/yuan23_prior/' # Old FIXME : remove it later
        
        multipoles_mean = []
        for phase in range(25): # NOTE: Hardcoded !
            data_dir = f'{base_dir}/c000_ph{phase:03}/seed0' # NOTE: Hardcoded !
            multipoles_hods = []
            for hod in range(50): # NOTE: Hardcoded !
                data_fn = Path(data_dir) / f'tpcf_hod{hod:03}.npy' # NOTE: File name format hardcoded !
                data = TwoPointCorrelationFunction.load(data_fn)[::rebin]
                s, multipoles = data(ells=ells, return_sep=True) 
                multipoles_hods.append(multipoles)
            multipoles_hods = np.array(multipoles_hods).mean(axis=0)
            multipoles_mean.append(multipoles_hods)
        multipoles_mean = np.array(multipoles_mean).mean(axis=0)

        data_dir = f'{base_dir}/c000_ph000/seed0'  # NOTE: Hardcoded !
        multipoles_ph0 = []
        for hod in range(50): # NOTE: Hardcoded !
            data_fn = Path(data_dir) / f'tpcf_hod{hod:03}.npy' # NOTE: File name format hardcoded !
            data = TwoPointCorrelationFunction.load(data_fn)[::4]
            s, multipoles = data(ells=ells, return_sep=True) 
            multipoles_ph0.append(multipoles)
        multipoles_ph0 = np.array(multipoles_ph0).mean(axis=0)
        delta = ((multipoles_mean + 1) - (multipoles_ph0 + 1))/(multipoles_ph0 + 1)
        return delta.reshape(-1)

    def apply_phase_correction(self, prediction):
        """
        Apply the phase correction to the predictions.
        We apply this to (1 + prediction) to avoid zero-crossings.

        Parameters
        ----------
        prediction : np.ndarray
            Array of predictions.

        Returns
        -------
        np.ndarray
            Corrected predictions.
        """
        return (1 + prediction) * (1 + self.phase_correction) - 1

    @set_plot_style
    def plot_observable(self, model_params: dict, save_fn: str = None, show: bool = False):
        """
        Plot the reconstructed galaxy power spectrum multipoles data, model, and residuals.

        Parameters
        ----------
        model_params : dict
            Dictionary of model parameters to use for the prediction.
        save_fn : str
            Filename to save the plot. If None, the plot is not saved.
        show : bool
            If True, display the plot. Default is False.
        """
        import matplotlib.pyplot as plt

        height_ratios = [3, 1]
        figsize = (6, 1.5 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False,
            gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0.1)
        show_legend = False

        lax[-1].set_xlabel(r'$r_p$ [$h^{-1}\,\mathrm{Mpc}$]', fontsize=15)
        lax[0].set_ylabel(r'$r_p w_p(r_p)$ [$h^{-1}\,\mathrm{Mpc}$]', fontsize=15)

        rp = self.r_p
        data = self.y[0]
        model = self.get_model_prediction(model_params)[0]
        cov = self.get_covariance_matrix(volume_factor=64)
        error = np.sqrt(np.diag(cov))

        lax[0].errorbar(rp, rp * data, rp * error, marker='o', ms=4, ls='', 
            color=f'C0', elinewidth=1.0, capsize=None)
        lax[0].plot(rp, rp * model, ls='-', color=f'C0')
        lax[1].plot(rp, (data - model) / error, ls='-', color=f'C0')

        for offset in [-2, 2]: lax[1].axhline(offset, color='k', ls='--')
        lax[1].set_ylabel(rf'$\Delta w_p / \sigma_{{w_p}}$', fontsize=15)
        lax[1].set_ylim(-4, 4)

        for ax in lax:
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=14)
        if show_legend: lax[0].legend(fontsize=15)

        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving plot to {save_fn}')
        if show:
            plt.show()