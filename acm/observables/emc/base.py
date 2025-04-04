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
    def __init__(
        self,
        select_coordinates: dict = {},
        select_mocks: dict = {},
        select_indices: dict = {},
        slice_coordinates: dict = {}
    ):
        if bool((select_coordinates or slice_coordinates) and select_indices):
            raise ValueError(
                "You can only select either coordinates or indices, not both."
            )
        self.select_coordinates = select_coordinates
        self.select_mocks = select_mocks
        self.slice_filters = slice_coordinates
        self.select_indices = select_indices
        self.select_filters = {**select_mocks, **select_coordinates, **select_indices}

        self.model = self.load_model()
        self.separation = self.load_separation()
        
    @property
    def nparams(self):
        """
        Number of parameters in the Latin hypercube samples
        (equal to the number of input parameters for the emulators.
        """
        return len(self.lhc_x_names)

    def lhc_fname(self):
        """
        File containing Latin hypercube samples.
        """
        lhc_dir = Path(emc_paths['lhc_dir'])
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
        return covariance_dir / f'{self.stat_name}.npy'

    def diffsky_fname(self, phase_idx, sampling):
        """
        File containing the measurements from Diffsky simulations.
        """
        base_dir = Path(emc_paths['diffsky_dir'])
        diffsky_dir = base_dir / f'galsampled_67120_fixedAmp_{phase_idx:03}_{sampling}_v0.3'
        return diffsky_dir / f'{self.stat_name}.npy'

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
        coords = self.lhc_indices
        coords.update({'param_idx': list(range(self.nparams))})
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(coords.keys())
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
        coords = self.lhc_indices
        if self.select_indices:
            coords.update(self.coordinates_indices)
        else:
            coords.update(self.coordinates)
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(coords.keys())
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
        coords = self.small_box_indices
        if self.select_indices:
            coords.update(self.coordinates_indices)
        else:
            coords.update(self.coordinates)
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(coords.keys())
        small_box_y = small_box_y.reshape(*coords_shape)
        return convert_to_summary(
            data=small_box_y, dimensions=dimensions, coords=coords,
            select_filters=self.select_filters, slice_filters=self.slice_filters
        ).values.reshape(len(small_box_y), -1)

    def diffsky_y(self, phase_idx=1, sampling='mass_conc'):
        """
        Measurements from Diffsky simulations.
        """
        fn = self.diffsky_fname(phase_idx=phase_idx, sampling=sampling)
        diffsky_y = np.load(fn, allow_pickle=True).item()['diffsky_y']
        coords = self.coordinates_indices if self.select_indices else self.coordinates
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(coords.keys())
        diffsky_y = diffsky_y.reshape(*coords_shape)
        return convert_to_summary(
            data=diffsky_y, dimensions=dimensions, coords=coords,
            select_filters=self.select_filters, slice_filters=self.slice_filters
        ).values.reshape(-1)

    def load_model(self):
        """
        Load trained theory model from checkpoint file.
        """
        from sunbird.emulators import FCN
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
        coords = self.coordinates_indices if self.select_indices else self.coordinates
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(self.coords.keys())
        error = error.reshape(*coords_shape)
        return convert_to_summary(
            data=error, dimensions=dimensions, coords=coords,
            select_filters=self.select_filters, slice_filters=self.slice_filters
        ).values.reshape(-1)

    def load_separation(self):
        """
        Separation values (s for the correlation function, k for power spectrum, etc.)
        """
        fn = self.lhc_fname()
        return np.load(fn, allow_pickle=True).item()[self.sep_name]

    def get_model_prediction(self, x, batch=True):
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
        if hasattr(self, 'phase_correction'):
            prediction = self.apply_phase_correction(prediction)
        coords = self.coordinates_indices if self.select_indices else self.coordinates
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