from abc import ABC, abstractmethod
from sunbird.data.data_utils import convert_to_summary
from .paths import emc_paths
from pathlib import Path
import numpy as np
import torch


class BaseObservable(ABC):
    """
    Base class for the Emulator's Mock Challenge observables.
    """
    def __init__(self):
        pass

    def lhc_fname(self):
        """
        File containing Latin hypercube samples.
        """
        lhc_dir = Path(emc_paths['lhc_dir'])
        print(self.stat_name)
        print(lhc_dir / f'{self.stat_name}.npy')
        return lhc_dir / f'{self.stat_name}.npy'

    def emulator_error_fname(self):
        """
        File containing the emulator error.
        """
        emulator_error_dir = Path(emc_paths['emulator_error_dir'])
        return emulator_error_dir / f'{self.stat_name}.npy'

    def small_box_fname(self):
        """
        File containing the output features from the small AbacusSummit box.
        """
        covariance_dir = Path(emc_paths['covariance_dir'])
        print(covariance_dir / f'{self.stat_name}.npy')
        return covariance_dir / f'{self.stat_name}.npy'

    def diffsky_fname(self, phase_idx, sampling):
        """
        File containing the measurements from Diffsky simulations.
        """
        base_dir = Path(emc_paths['diffsky_dir'])
        diffsky_dir = base_dir / f'galsampled_67120_fixedAmp_{phase_idx:03}_{sampling}_v0.3'
        return diffsky / f'{self.stat_name}.npy'

    @property
    @abstractmethod
    def model_fn(self):
        pass

    @property
    def lhc_x(self):
        """
        Latin hypercube of input features (cosmological and/or HOD parameters)
        """
        fn = self.lhc_fname()
        lhc_x = np.load(fn, allow_pickle=True).item()['lhc_x']
        coords = self.coords_lhc_x
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(self.coords_lhc_x.keys())
        lhc_x = lhc_x.reshape(*coords_shape)
        return convert_to_summary(
            data=lhc_x, dimensions=dimensions, coords=coords,
            select_filters=self.select_filters, slice_filters=self.slice_filters
        ).values.reshape(-1)

    @property
    def lhc_x_names(self):
        """
        Names of the input features (cosmological and/or HOD parameters)
        """
        fn = self.lhc_fname()
        return np.load(fn, allow_pickle=True).item()['lhc_x_names']
    
    @property
    def lhc_y(self):
        """
        Latin hypercube of output features (tpcf, power spectrum, etc).
        """
        fn = self.lhc_fname()
        lhc_y = np.load(fn, allow_pickle=True).item()['lhc_y']
        coords = self.coords_lhc_y
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(self.coords_lhc_y.keys())
        lhc_y = lhc_y.reshape(*coords_shape)
        return convert_to_summary(
            data=lhc_y, dimensions=dimensions, coords=coords,
            select_filters=self.select_filters, slice_filters=self.slice_filters
        ).values.reshape(-1)

    @property
    def small_box_y(self):
        """
        Output features from the small AbacusSummit box for covariance
        estimation.
        """
        fn = self.small_box_fname()
        small_box_y = np.load(fn, allow_pickle=True).item()['cov_y']
        coords = self.coords_small_box
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(self.coords_small_box.keys())
        small_box_y = small_box_y.reshape(*coords_shape)
        return convert_to_summary(
            data=small_box_y, dimensions=dimensions, coords=coords,
            select_filters=self.select_filters, slice_filters=self.slice_filters
        ).values.reshape(len(small_box_y), -1)

    @property
    def diffsky_y(self):
        """
        Measurements from Diffsky simulations.
        """
        fn = self.diffsky_fname()
        diffsky_y = np.load(fn, allow_pickle=True).item()['diffsky_y']
        coords = self.coords_model
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(coords.keys())
        diffsky_y = diffsky_y.reshape(*coords_shape)
        return convert_to_summary(
            data=diffsky_y, dimensions=dimensions, coords=coords,
            select_filters=self.select_filters, slice_filters=self.slice_filters
        ).values.reshape(-1)

    @property
    def model(self):
        """
        Load trained theory model from checkpoint file.
        """
        from sunbird.emulators import FCN
        print('importing model')
        model = FCN.load_from_checkpoint(self.model_fn, strict=True)
        model = model.eval().to('cpu')
        if self.stat_name == 'minkowski':
            from sunbird.data.transforms_array import WeiLiuInputTransform, WeiLiuOutputTransForm
            model.transform_output = WeiLiuOutputTransForm()
            model.transform_input = WeiLiuInputTransform()
        return model

    @property
    def emulator_error(self):
        """
        Emulator error.
        """
        fn = self.emulator_error_fname()
        error = np.load(fn, allow_pickle=True).item()['emulator_error']
        coords = self.coords_model
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(self.coords_model.keys())
        error = error.reshape(*coords_shape)
        return convert_to_summary(
            data=error, dimensions=dimensions, coords=coords,
            select_filters=self.select_filters, slice_filters=self.slice_filters
        ).values.reshape(-1)

    @property
    def separation(self):
        """
        Separation values (s for the correlation function, k for power spectrum, etc.)
        """
        fn = self.lhc_fname()
        return np.load(fn, allow_pickle=True).item()[self.sep_name]

    @property
    @abstractmethod
    def coords_lhc_x(self):
        pass

    @property
    @abstractmethod
    def coords_lhc_y(self):
        pass

    @property
    @abstractmethod
    def coords_small_box(self):
        pass

    @property
    @abstractmethod
    def coords_model(self):
        pass

    def get_model_prediction(self, x, batch=False):
        """
        Get model prediction for a given x.

        Args:
            x (np.ndarray): Input features.

        Returns:
            np.ndarray: Model prediction.
        """
        with torch.no_grad():
            prediction = self.model.get_prediction(torch.Tensor(x))
            prediction = prediction.numpy()
        coords = self.coords_model
        coords_shape = tuple(len(v) for k, v in coords.items())
        if len(prediction.shape) > 1: # batch query
            dimensions = ["batch"] + list(coords.keys())
            coords["batch"] = range(len(prediction))
            prediction = prediction.reshape((len(prediction), *coords_shape))
            return convert_to_summary(
                data=prediction, dimensions=dimensions, coords=coords,
                select_filters=self.select_filters, slice_filters=self.slice_filters
            ).values.reshape(len(prediction), -1)
        else:
            coords_shape = tuple(len(v) for k, v in coords.items())
            prediction = prediction.reshape(coords_shape)
            dimensions = list(coords.keys())
            return convert_to_summary(
                data=prediction, dimensions=dimensions, coords=coords,
                select_filters=self.select_filters, slice_filters=self.slice_filters
            ).values.reshape(-1)

    def get_covariance_matrix(self, divide_factor=64):
        """
        Covariance matrix of the combination of observables.

        Args:
            divide_factor (int): Divide the covariance matrix by this value
            to account for the volume difference between the small boxes and the target
            simulation.
        """
        return np.cov(self.small_box_y.T) / divide_factor

    def get_covariance_correction(self, n_s, n_d, n_theta=None, method='percival'):
        """
        Correction factor to debias de inverse covariance matrix.

        Args:
            n_s (int): Number of simulations.
            n_d (int): Number of bins of the data vector.
            n_theta (int): Number of free parameters.
            method (str): Method to compute the correction factor.

        Returns:
            float: Correction factor
        """
        if method == 'percival':
            B = (n_s - n_d - 2) / ((n_s - n_d - 1)*(n_s - n_d - 4))
            return (n_s - 1)*(1 + B*(n_d - n_theta))/(n_s - n_d + n_theta - 1)
        elif _method == 'hartlap':
            return (n_s - 1)/(n_s - n_d - 2)