from .base import BaseObservableEMC

class GalaxyCorrelationFunctionMultipoles(BaseObservableEMC):
    """
    Class for the Emulator's Mock Challenge galaxy correlation
    function multipoles.
    """
    
    stat_name = 'tpcf'
    
    _data_features = {
        'multipoles': [0, 2, 4],
    }
    
    @property
    def summary_coords_dict(self) -> dict:
        return {
            **self._summary_coords_dict,
            'data_features': self._data_features,
        }
    
    @property
    def paths(self) -> dict:
        paths = super().paths
        paths['statistic_dir'] = f'/pscratch/sd/e/epaillas/emc/training_sets/tpcf/cosmo+hod_bugfix/z0.5/yuan23_prior/'
        paths['statistic_covariance_dir'] = f'/pscratch/sd/e/epaillas/emc/covariance_sets/tpcf/z0.5/yuan23_prior/'
        return paths
    
    @property
    def checkpoint_fn(self) -> str:
        """
        Override checkpoint_fn to point to the correct checkpoint file.
        """
        return '/pscratch/sd/e/epaillas/emc/v1.1/trained_models/best/GalaxyCorrelationFunctionMultipoles/last.ckpt'
    
    def compress_covariance(self, save_to: str = None, rebin: int = 4, ells: list = [0, 2, 4]):
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
        
        Returns
        -------
        np.ndarray
            Covariance array. 
        """
        from pycorr import TwoPointCorrelationFunction
        from pathlib import Path
        import numpy as np
        
        # Directories
        base_dir = Path(self.paths['measurements_dir']) / 'small' / self.stat_name
        # base_dir = Path(f'/pscratch/sd/e/epaillas/emc/covariance_sets/tpcf/z0.5/yuan23_prior/') # Old FIXME : remove it later
        data_fns = list(base_dir.glob('tpcf_ph*_hod466.npy')) # NOTE: File name format hardcoded !
        
        y = []
        for data_fn in data_fns:
            data = TwoPointCorrelationFunction.load(data_fn)[::rebin]
            s, multipoles = data(ells=ells, return_sep=True) 
            y.append(np.concatenate(multipoles))
        y = np.array(y)
        bin_values = s
        
        self.logger.info(f'Loaded covariance with shape: {y.shape}')
        
        cout = {'bin_values': bin_values, 'cov_y': y}
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, cout)
            self.logger.info(f'Saving compressed covariance file to {save_fn}')
            
        return y
    
    def compress_data(
        self, 
        add_covariance: bool = False,
        save_to: str = None,
        rebin: int = 4, 
        ells: list = [0, 2, 4],
        cosmos: list = None,
        n_hod: int = 100,
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
        dict
            Dictionary containing the compressed data with the following keys:
            - 'bin_values' : Array of the bin values.
            - 'x' : Array of the parameters used to generate the simulations.
            - 'y' : Array of the statistics values.
            - 'x_names' : List of the names of the parameters.
            - 'cov_y' : Array of the covariance matrix of the statistics values
        """
        from pycorr import TwoPointCorrelationFunction
        from pathlib import Path
        import numpy as np
        if cosmos is None:
            cosmos = self.summary_coords_dict['sample_features']['cosmo_idx']
        
        # Directories
        base_dir = self.paths['measurements_dir'] + f'base/{self.stat_name}/'
        # base_dir = '/pscratch/sd/e/epaillas/emc/training_sets/tpcf/cosmo+hod_bugfix/z0.5/yuan23_prior/' # Old FIXME : remove it later
        
        y = []
        for cosmo_idx in cosmos:
            data_dir = base_dir + f'c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx:01}/'
            for hod_idx in range(n_hod):
                data_fn = f"{data_dir}/tpcf_hod{hod_idx:03}.npy" # NOTE: File name format hardcoded !
                data = TwoPointCorrelationFunction.load(data_fn)[::rebin]
                s, multipoles = data(ells=ells, return_sep=True) 
                y.append(np.concatenate(multipoles))
        y = np.array(y)
        bin_values = s
        x, x_names = self.compress_x(cosmos=cosmos, n_hod=n_hod)
        
        self.logger.info(f'Loaded data with shape: {x.shape}, {y.shape}')
        
        cout = {'bin_values': bin_values, 'x': x, 'x_names': x_names, 'y': y}
        if add_covariance:
            cov_y = self.compress_covariance(rebin=rebin, ells=ells)
            cout['cov_y'] = cov_y
        
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, cout)
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
        for phase in range(25):
            data_dir = f'{base_dir}/c000_ph{phase:03}/seed0'
            multipoles_hods = []
            for hod in range(50):
                data_fn = Path(data_dir) / f'tpcf_hod{hod:03}.npy' # NOTE: File name format hardcoded !
                data = TwoPointCorrelationFunction.load(data_fn)[::rebin]
                s, multipoles = data(ells=ells, return_sep=True) 
                multipoles_hods.append(multipoles)
            multipoles_hods = np.array(multipoles_hods).mean(axis=0)
            multipoles_mean.append(multipoles_hods)
        multipoles_mean = np.array(multipoles_mean).mean(axis=0)

        data_dir = f'{base_dir}/c000_ph000/seed0'
        multipoles_ph0 = []
        for hod in range(50):
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