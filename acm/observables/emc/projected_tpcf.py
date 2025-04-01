from .base import BaseObservable


class GalaxyProjectedCorrelationFunction(BaseObservable):
    """
    Class for the Emulator's Mock Challenge projected galaxy correlation function.
    """
    def __init__(self, select_filters: dict = None, slice_filters: dict = None):
        self.stat_name = 'corrected_wp'
        self.sep_name = 'rp'
        self.select_filters = select_filters
        self.slice_filters = slice_filters
        super().__init__()

    @property
    def coords_lhc_x(self):
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(350)),
            'param_idx': list(range(20))
        }

    @property
    def coords_lhc_y(self):
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(350)),
            self.sep_name: self.separation,
        }

    @property
    def coords_small_box(self):
        return {
            'phase_idx': list(range(1786)),
            self.sep_name: self.separation,
        }

    @property
    def coords_model(self):
        return {
            self.sep_name: self.separation,
        }

    @property
    def model_fn(self):
        # return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/CorrectedGalaxyProjectedCorrelationFunction/cosmo+hod/optuna/log/last-v35.ckpt'
        # return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/CorrectedGalaxyProjectedCorrelationFunction/cosmo+hod/asinh/last-v3.ckpt'
        return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/GalaxyProjectedCorrelationFunction/cosmo+hod/optuna/apr1/log/last-v66.ckpt'
        # return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/CorrectedGalaxyProjectedCorrelationFunction/cosmo+hod/log/last-v4.ckpt'
        # return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/CorrectedGalaxyProjectedCorrelationFunction/cosmo+hod/optuna/mar29/log/last-v91.ckpt'

    def create_lhc(self, n_hod=20, cosmos=None, phase_idx=0, seed_idx=0):
        x, x_names = self.create_lhc_x(cosmos=cosmos, n_hod=n_hod)
        sep, y = self.create_lhc_y(n_hod=n_hod, cosmos=cosmos, phase_idx=phase_idx, seed_idx=seed_idx)
        return sep, x, x_names, y

    def create_lhc_y(self, n_hod=100, cosmos=None, phase_idx=0, seed_idx=0):
        import numpy as np
        from pycorr import TwoPointCorrelationFunction
        base_dir = '/pscratch/sd/e/epaillas/emc/training_sets/xi_rppi/cosmo+hod/z0.5/yuan23_prior/'
        if cosmos is None:
            cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
        y = []
        for cosmo_idx in cosmos:
            print(cosmo_idx)
            data_dir = base_dir + f'c{cosmo_idx:03}_ph{phase_idx:03}/seed0/'
            for hod_idx in range(n_hod):
                data_fn = f"{data_dir}/xi_rppi_hod{hod_idx:03}.npy"
                data = TwoPointCorrelationFunction.load(data_fn)
                rp, wp = data(pimax=None, return_sep=True)
                y.append(wp)
        return rp, np.array(y)

    def create_lhc_x(self, cosmos=None, n_hod=100):
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
        from pycorr import TwoPointCorrelationFunction
        from pathlib import Path
        import numpy as np
        data_dir = Path('/pscratch/sd/e/epaillas/emc/covariance_sets/xi_rppi/z0.5/yuan23_prior/')
        data_fns = list(data_dir.glob('xi_rppi_ph*_hod466.npy'))
        y = []
        for data_fn in data_fns:
            data = TwoPointCorrelationFunction.load(data_fn)
            rp, wp = data(pimax=None, return_sep=True)
            y.append(wp)
        return rp, np.array(y)

    def get_emulator_error(self, select_filters=None, slice_filters=None):
        """
        Calculate the emulator error from a subset of the Latin hypercube,
        which we treat as the test set.
        
        We make a new instance of the class with the test set filters and
        compare the emulator prediction to the true values.
        """
        import numpy as np
        if select_filters is None:
            select_filters = self.select_filters
            select_filters['cosmo_idx'] = list(range(0, 5)) + list(range(13, 14))
            select_filters['hod_idx'] = list(range(350))
        if slice_filters is None:
            slice_filters = self.slice_filters
        # instantiate class with test set filters
        observable = self.__class__(select_filters=select_filters, slice_filters=slice_filters)
        test_x = observable.lhc_x
        test_y = observable.lhc_y
        # reshape to (n_samples, n_features)
        n_samples = len(select_filters['cosmo_idx']) * len(select_filters['hod_idx'])
        test_x = test_x.reshape(n_samples, -1)
        test_y = test_y.reshape(n_samples, -1)
        pred_y = observable.get_model_prediction(test_x, batch=True)
        return np.median(np.abs(test_y - pred_y), axis=0)