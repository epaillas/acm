from .base import BaseObservable
import logging


class GalaxyCorrelationFunctionMultipoles(BaseObservable):
    """
    Class for the Emulator's Mock Challenge galaxy correlation
    function multipoles.
    """
    def __init__(
        self,
        select_mocks: dict = None,
        select_indices: list = None,
        select_coordinates: dict = None,
        slice_coordinates: dict = None,
        phase_correction: bool = False,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stat_name = 'tpcf'
        self.sep_name = 's'

        if phase_correction and hasattr(self, 'compute_phase_correction'):
            self.logger.info('Computing phase correction.')
            self.phase_correction = self.compute_phase_correction()

        self.select_mocks = select_mocks
        self.select_coordinates = select_coordinates
        self.slice_coordinates = slice_coordinates
        self.select_indices = {'bin_idx': select_indices} if select_indices else {}
        super().__init__()

    @property
    def lhc_indices(self):
        """
        Indices of the Latin hypercube samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(350)),
        }

    @property
    def test_set_indices(self):
        """
        Indices of the test set samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)),
            'hod_idx': list(range(350)),
        }

    @property
    def small_box_indices(self):
        """
        Indices of the covariance samples, including variations in phase and HOD parameters.
        """
        return {
            'phase_idx': list(range(1786)),
        }

    @property
    def coordinates(self):
        """
        Coordinates of the data and model vectors.
        """
        return{
            'multipoles': [0, 2, 4],
            's': self.separation,
        }
    
    @property
    def coordinates_indices(self):
        """
        Indices of the (flat) coordinates of the data and model vectors.
        """
        return{'bin_idx': list(range(3 * len(self.separation)))}
        
    @property
    def model_fn(self):
        """
        Path to the trained emulator checkpoint file.
        """
        return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/best/GalaxyCorrelationFunctionMultipoles/last.ckpt'

    def create_lhc(self, n_hod=20, cosmos=None, phase_idx=0, seed_idx=0):
        """
        Create the Latin hypercube samples for the emulator (both input and output features).
        """
        x, x_names = self.create_lhc_x(cosmos=cosmos, n_hod=n_hod)
        sep, y = self.create_lhc_y(n_hod=n_hod, cosmos=cosmos, phase_idx=phase_idx, seed_idx=seed_idx)
        return sep, x, x_names, y

    def create_lhc_y(self, n_hod=100, cosmos=None, phase_idx=0, seed_idx=0):
        """
        Create the output features for the emulator (the galaxy correlation function multipoles).
        """
        import numpy as np
        from pycorr import TwoPointCorrelationFunction
        base_dir = '/pscratch/sd/e/epaillas/emc/training_sets/tpcf/cosmo+hod_bugfix/z0.5/yuan23_prior/'
        if cosmos is None:
            cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
        y = []
        for cosmo_idx in cosmos:
            print(cosmo_idx)
            data_dir = base_dir + f'c{cosmo_idx:03}_ph{phase_idx:03}/seed0/'
            for hod_idx in range(n_hod):
                data_fn = f"{data_dir}/tpcf_hod{hod_idx:03}.npy"
                data = TwoPointCorrelationFunction.load(data_fn)[::4]
                s, multipoles = data(ells=(0, 2, 4), return_sep=True)
                y.append(np.concatenate(multipoles))
        return s, np.array(y)

    def create_lhc_x(self, cosmos=None, n_hod=100):
        """
        Create the input features for the emulator (the cosmological and HOD parameters).
        """
        import pandas
        import numpy as np
        if cosmos is None:
            cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
        lhc_x = []
        for cosmo_idx in cosmos:
            data_dir = '/pscratch/sd/e/epaillas/emc/cosmo+hod_params/'
            data_fn = data_dir + f'AbacusSummit_c{cosmo_idx:03}.csv'
            lhc_x_i = pandas.read_csv(data_fn)
            lhc_x_names = list(lhc_x_i.columns)
            lhc_x_names = [name.replace(' ', '').replace('#', '') for name in lhc_x_names]
            lhc_x.append(lhc_x_i.values[:n_hod, :])
        lhc_x = np.concatenate(lhc_x)
        return lhc_x, lhc_x_names

    def create_small_box_y(self):
        """
        Create the output features for the emulator (the galaxy correlation function multipoles)
        from the small AbacusSummit box.
        """
        from pathlib import Path
        from pycorr import TwoPointCorrelationFunction
        import numpy as np
        data_dir = Path('/pscratch/sd/e/epaillas/emc/covariance_sets/tpcf/z0.5/yuan23_prior')
        data_fns = list(data_dir.glob('tpcf_ph*_hod466.npy'))
        y = []
        for data_fn in data_fns:
            data = TwoPointCorrelationFunction.load(data_fn)[::4]
            s, multipoles = data(ells=(0, 2, 4), return_sep=True)
            y.append(np.concatenate(multipoles))
        return s, np.array(y)

    def compute_phase_correction(self):
        """
        Correction factor to bring the fixed phase precictions (p000) to the ensemble average.
        """
        from pathlib import Path
        import numpy as np
        from pycorr import TwoPointCorrelationFunction, setup_logging
        setup_logging(level='WARNING')
        multipoles_mean = []
        for phase in range(25):
            data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/tpcf/cosmo+hod_bugfix/z0.5/yuan23_prior/c000_ph{phase:03}/seed0'
            multipoles_hods = []
            for hod in range(50):
                data_fn = Path(data_dir) / f'tpcf_hod{hod:03}.npy'
                data = TwoPointCorrelationFunction.load(data_fn)[::4]
                s, multipoles = data(ells=(0, 2, 4), return_sep=True)
                multipoles_hods.append(multipoles)
            multipoles_hods = np.array(multipoles_hods).mean(axis=0)
            multipoles_mean.append(multipoles_hods)
        multipoles_mean = np.array(multipoles_mean).mean(axis=0)

        data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/tpcf/cosmo+hod_bugfix/z0.5/yuan23_prior/c000_ph000/seed0'
        multipoles_ph0 = []
        for hod in range(50):
            data_fn = Path(data_dir) / f'tpcf_hod{hod:03}.npy'
            data = TwoPointCorrelationFunction.load(data_fn)[::4]
            s, multipoles = data(ells=(0, 2, 4), return_sep=True)
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