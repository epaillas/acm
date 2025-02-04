from .base import BaseObservable


class GalaxyBispectrumMultipoles(BaseObservable):
    """
    Class for the Emulator's Mock Challenge bispectrum.
    """
    def __init__(self, select_filters: dict = None, slice_filters: dict = None):
        self.stat_name = 'bk'
        self.sep_name = 'k123'
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
            'multipoles': [0, 2],
            'bin_index': list(range(len(self.separation.prod(axis=0)))),
        }

    @property
    def coords_small_box(self):
        return {
            'phase_idx': list(range(1786)),
            'multipoles': [0, 2],
            'bin_index': list(range(len(self.separation.prod(axis=0)))),
        }

    @property
    def coords_model(self):
        return {
            'multipoles': [0, 2],
            'bin_index': list(range(len(self.separation.prod(axis=0)))),
        }

    @property
    def model_fn(self):
        # return '/pscratch/sd/e/epaillas/emc/v1.1/trained_models/bk/cosmo+hod/optuna/last-v55.ckpt'
        return '/pscratch/sd/e/epaillas/emc/v1.1/trained_models/GalaxyBispectrumMultipoles/cosmo+hod/optuna/last.ckpt'

    def create_lhc(self, n_hod=100, cosmos=None, ells=[0, 2], phase_idx=0, seed_idx=0):
        x, x_names = self.create_lhc_x(cosmos=cosmos, n_hod=n_hod)
        sep, y = self.create_lhc_y(n_hod=n_hod, cosmos=cosmos, ells=ells, phase_idx=phase_idx, seed_idx=seed_idx)
        return sep, x, x_names, y

    def create_lhc_y(self, n_hod=100, cosmos=None, ells=[0, 2], phase_idx=0, seed_idx=0):
        import numpy as np
        base_dir = '/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/raw/'
        if cosmos is None:
            cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
        y = []
        for cosmo_idx in cosmos:
            print(cosmo_idx)
            data_dir = base_dir + f'bispectrum/kmin0.013_kmax0.253_dk0.020/c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed_idx:01}/'
            for hod in range(n_hod):
                data_fn = f"{data_dir}/bispectrum_hod{hod:03d}.npy"
                data = np.load(data_fn, allow_pickle=True).item()
                k123 = data['k123']
                bk = data['bk']
                weight = k123.prod(axis=0) / 1e5
                multipoles = np.concatenate([weight * bk[f'b{i}'] for i in ells])
                bin_index = len(multipoles)
                y.append(multipoles)
        return k123, np.array(y)

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
        from pathlib import Path
        import numpy as np
        data_dir = Path('/pscratch/sd/e/epaillas/emc/v1.1/abacus/covariance_sets/small_box/raw/bispectrum/kmax0.25_dk0.02/')
        data_fns = list(data_dir.glob('bispectrum_ph*_hod466.npy'))
        y = []
        for data_fn in data_fns:
            data = np.load(data_fn, allow_pickle=True).item()
            k123 = data['k123']
            bk = data['bk']
            weight = k123.prod(axis=0) / 1e5
            multipoles = np.concatenate([weight * bk[f'b{i}'] for i in [0, 2]])
            y.append(multipoles)
        return k123, np.array(y)

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
        