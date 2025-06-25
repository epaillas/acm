from .base import BaseObservableBGS

class DensitySplitCorrelationFunctionMultipoles(BaseObservableBGS):
    """
    Class for the application of the densitysplit statistic of the ACM pipeline to the BGS dataset.
    """
    
    stat_name = 'dsc_conf'
    
    _data_features = {
        'statistics': ['quantile_data_correlation', 'quantile_correlation'],
        'quantiles': [0, 1, 3, 4],
        'multipoles': [0, 2],
    }
    
    @property
    def summary_coords_dict(self):
        return {
            **self._summary_coords_dict,
            'data_features': self._data_features,
        }

    #%% Compressed files creation
    def compress_covariance(
        self, 
        save_to: str = None, 
        rebin: int = 1, 
        ells: list = [0, 2],
        quantiles: list = [0, 1, 3, 4],
    ):
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
        
        Returns
        -------
        np.ndarray
            Covariance array. 
        """
        import numpy as np
        from pathlib import Path
        
        base_dir = Path(self.paths['measurements_dir']) / 'small' / self.stat_name
        outliers_path = base_dir / 'outliers_idx.npy' # NOTE: Hardcoded !
        outliers_phases = np.load(outliers_path)
        
        y = []
        for phase in range(3000, 5000):
            multipoles_stat = []
            for stat in ['ccf', 'acf']: # NOTE: Hardcoded !
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
        rebin: int = 1, 
        ells: list = [0, 2],
        quantiles: list = [0, 1, 3, 4],
        cosmos: list = None,
        n_hod: int = 100,
    ) -> dict:
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
        from pathlib import Path
        import numpy as np
        if cosmos is None:
            cosmos = self.summary_coords_dict['sample_features']['cosmo_idx']
            
        # Directories
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
        bin_values = s
        x, x_names = self.compress_x()
        
        self.logger.info(f'Loaded data with shape: {x.shape}, {y.shape}')
        
        cout = {'bin_values': bin_values, 'x': x, 'x_names': x_names, 'y': y}
        if add_covariance:
            cov_y = self.compress_covariance(rebin=rebin, ells=ells, quantiles=quantiles)
            cout['cov_y'] = cov_y
            
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            save_fn = Path(save_to) / f'{self.stat_name}.npy'
            np.save(save_fn, cout)
            self.logger.info(f'Saving compressed data to {save_fn}')
        
        return cout