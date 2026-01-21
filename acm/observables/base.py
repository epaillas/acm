import torch
import xarray
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sunbird.emulators import FCN
from sunbird.data.data_utils import transform_filters_to_slices
from scipy.stats import median_abs_deviation, norm
from acm.utils.xarray import dataset_from_dict
from acm.utils.covariance import orthogonal_gk_mad_covariance, check_covariance_matrix
from acm.utils.plotting import set_plot_style
from acm.utils.decorators import temporary_class_state

# Register safe globals for transform classes to allow loading checkpoints
# with PyTorch 2.6+ (which changed weights_only default to True)
from sunbird.data.transforms_array import LogTransform, ArcsinhTransform
SAFE_CLASSES = [LogTransform, ArcsinhTransform]

class Observable():
    """
    Class to load a compressed Observable file or model and apply filters to their outputs.
    """

    def __init__(
        self,
        stat_name: str,
        dataset: xarray.Dataset = None,
        model: FCN = None,
        select_filters: dict = None,
        slice_filters: dict = None,
        select_indices: list = None,
        select_indices_on: list = ['y', 'covariance_y', 'emulator_error', 'emulator_covariance_y'],
        flat_output_dims: int = None,
        squeeze_output: bool = False,
        numpy_output: bool = False,
        paths: dict = None,
        checkpoint_fn: Path | str = None,
    ):
        """
        Parameters
        ----------
        stat_name: str
            Name or identifier of the statistic to load. It is also the name of the loaded files if applicable.
        dataset : xarray.Dataset, optional
            The xarray Dataset containing the data variables and coordinates.
        model : FCN, optional
            Trained theory model. If None, the model attribute of the class remains undefined. Defaults to None.
        select_filters : dict, optional
            Filters to select values in coordinates. Defaults to None.
        slice_filters : dict, optional
            Filters to slice values in coordinates. Defaults to None.
        select_indices : list, optional
            Indices to select in the flattened data vector. Cannot be used with `select_filters` or `slice_filters`. Defaults to None.
        select_indices_on : list, optional
            List of data variables to apply the indices selection on. Defaults to ['y', 'covariance_y', 'emulator_error', 'emulator_covariance_y'].
        flat_output_dims : int, optional
            If 2, the output will be flattened on two dimensions (sample and features).
            If 1, the output will be flattened on a single dimension (dims) - Not recommended.
            If None, the output will not be flattened. Defaults to None.
        squeeze_output : bool, optional
            If True, the output will be squeezed to remove single-dimensional entries. Defaults to False.
        numpy_output : bool, optional
            If True, the output will be converted to a numpy array. Defaults to False.
        paths : dict, optional
            Paths to the compressed Observable directories and model checkpoint.
            If dataset or model is None, they will be loaded from the provided paths. Defaults to None.
        checkpoint_fn : Path | str, optional
            Legacy parameter for the model checkpoint file path. Use `paths['model_dir']` instead. Defaults to None.
            
        Raises
        ------
        ValueError
            If dataset is not provided and paths is None.

        Example
        -------
        ::
        
            slice_filters = {'sep': (0, 0.5),} 
            select_filters = {'ells': [0, 2],}


        will return the summary statistics for `0 < sep < 0.5` and multipoles 0 and 2
        
        Paths
        -----
        The data is expected to be in `paths[key]/stat_name.npy`, in which an xarray DataSet is stored.
        The possible keys are:
            - 'data_dir': directory containing the data (x, y)
            - 'covariance_dir': directory containing the covariance of the data (covariance_y)
            - 'error_dir': directory containing the emulator error of the data (emulator_error, emulator_covariance_y)
            - 'model_dir': directory containing the trained model checkpoint (`stat_name`.ckpt)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.stat_name = stat_name
        self.numpy_output = numpy_output
        self.squeeze_output = squeeze_output
        self.flat_output_dims = flat_output_dims
        
        if dataset is None:
            if paths is None:
                raise ValueError("If dataset is not provided, paths must be provided to load the dataset.")
            dataset = self.load_dataset_from_files(stat_name, paths)
        
        if model is None:
            try:
                if checkpoint_fn is not None:
                    self.logger.warning("The 'checkpoint_fn' parameter is deprecated. Please use 'paths['model_dir']/stat_name.ckpt' instead.")
                    checkpoint_fn = Path(checkpoint_fn)
                else:
                    checkpoint_fn = Path(paths['model_dir']) / f'{stat_name}.ckpt'
                model = self.load_model(checkpoint_fn)
            except Exception as e:
                self.logger.warning(f"Could not load model from checkpoint: {e}")
        
        self.model = model
        self._dataset = dataset
        self.logger.info("Datasets loaded with the following coordinates: {}".format(list(self._dataset.data_vars.keys())))
        
        # Set the filters
        self.select_filters = select_filters
        self.slice_filters = slice_filters
        self.select_indices = select_indices
        self.select_indices_on = select_indices_on if select_indices_on else [] # Ensure list behavior
        
        # Store paths for reference
        self.paths = paths if paths else {} # Ensure dict behavior
        
    @classmethod
    def load_dataset_from_files(cls, stat_name: str, paths: dict) -> xarray.Dataset:
        """
        Load the dataset from the provided paths.
        
        Parameters
        ----------
        stat_name: str
            Name or identifier of the statistic to load.
        paths: dict
            Paths to the compressed Observable directories.
            Keys can include 'data_dir', 'covariance_dir', 'error_dir'.
            Extra keys are ignored. Files are expected to be in `paths[key]/stat_name.npy`.
            
        Returns
        -------
        xarray.Dataset
            The loaded xarray DataSet.
            
        Raises
        ------
        FileNotFoundError
            If no datasets are found for the given statistic name in the provided paths.
        """
        logger = logging.getLogger(cls.__name__)
        
        # Try to read the paths with data inside
        datasets = []
        for key in ['data_dir', 'covariance_dir', 'error_dir']:
            if key not in paths:
                continue
            path = Path(paths[key]) / f"{stat_name}.npy"
        
            if path.exists():
                datasets.append(dataset_from_dict(np.load(path, allow_pickle=True).item()))
                logger.info(f"Loaded {key} from {path}")
        
        if len(datasets) == 0:
            raise FileNotFoundError(f"No datasets found for statistic '{stat_name}' in provided paths.")
        
        _dataset = xarray.merge(datasets)
        return _dataset  # pyright: ignore[reportReturnType] (xarray.merge return type is not well defined)
    
    @classmethod
    def load_model(cls, checkpoint_fn: Path | str = None) -> FCN:
        """
        Trained theory model loaded from checkpoint.
        
        Parameters
        ----------
        checkpoint_fn : Path | str
            Path to the model checkpoint file. See `sunbird.emulators.FCN.load_from_checkpoint`.
        
        Returns
        -------
        FCN
            The loaded FCN model.
        """
        logger = logging.getLogger(cls.__name__)
        
        # Register the classes as safe globals if torch.serialization.add_safe_globals exists
        if SAFE_CLASSES:
            try:
                torch.serialization.add_safe_globals(SAFE_CLASSES)
            except AttributeError:
                # torch.serialization.add_safe_globals doesn't exist in older PyTorch versions
                logger.debug("torch.serialization.add_safe_globals not available, skipping safe globals registration")
        
        # Load the model
        logger.info(f"Loading model from {checkpoint_fn}")
        model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
        model.eval().to('cpu')
        return model
    
    @classmethod
    def load(cls, filename: str):
        """
        Load an Observable object from a file.
        """
        # TODO
        raise NotImplementedError("Loading from file is not implemented yet.")
    
    def save(self, filename: str):
        """
        Saves the Observable object to a file.
        """
        # TODO
        raise NotImplementedError("Saving to file is not implemented yet.")
    
    def __repr__(self):
        """
        Returns a string representation of the Observable object.
        """
        r = f"<{type(self).__name__}>"
        for key, value in self.__dict__.items():
            if key == "logger":
                continue
            if key == "_dataset":
                key = "dataset"
            r += f"\n  {key}: {repr(value)},"
        if r.endswith(','):
            r = r[:-1]
        return r

    def __getattr__(self, name):
        """
        Returns the attribute of the class xarray _dataset, 
        with the filter applied. Also reshapes the output by stacking coordinates 
        (on one or two dims) if flat_output_dims is set.
        """
        # First, apply the filters
        dataset = self._dataset

        dataset = self.apply_filters(dataset)

        data = getattr(dataset, name)

        # Apply reshaping if name is a data_var
        if name in self._dataset.data_vars:
                
            if name in self.select_indices_on:
                data = self.apply_indices_selection(data)
            
            # Drop NaN dimensions if marked in attributes
            data = self.drop_nan_dimensions(data)
            
            data = self.flatten_output(data, self.flat_output_dims)
        
            if self.squeeze_output:
                data = data.squeeze()
                
            if self.numpy_output:
                data = data.values
            
        return data
    
    def drop_nan_dimensions(self, dataarray: xarray.DataArray) -> xarray.DataArray:
        """
        Drops dimensions that contain only NaN values in a DataArray.
        Does nothing if no 'nan_dims' attribute is found.

        Parameters
        ----------
        dataarray : xarray.DataArray
            The DataArray to drop NaN dimensions from. Must contain a 'nan_dims' attribute listing dimensions to check.

        Returns
        -------
        xarray.DataArray
            The DataArray with NaN dimensions dropped.
        """
        if 'nan_dims' not in dataarray.attrs:
            return dataarray
        
        for dim in dataarray.attrs['nan_dims']:
            # Ignore if dimension not present (e.g., already squeezed or filtered out)
            if dim not in dataarray.dims: 
                continue
            dataarray = dataarray.dropna(dim=dim, how='all')
        if dataarray.size == 0:
            self.logger.warning(f"All values dropped for {dataarray.name} due to NaN filters.")
        return dataarray

    @staticmethod
    def stack_on_attribute(attribute: str|dict, dataarray: xarray.DataArray, **kwargs) -> xarray.DataArray:
        """
        Stacks a DataArray on the dimensions given.

        Parameters
        ----------
        attribute: str | Mapping
            The dimension(s) to stack on. 
            If a string, will be read from the DataArray attributes.
            Will be used as the dim to stack on (see xarray.DataArray.stack)
        dataarray : xarray.DataArray
            The DataArray to stack the dimensions on.
        **kwargs
            Additional keyword arguments to pass to the stack method.

        Returns
        -------
        xarray.DataArray
            The stacked DataArray
        """
        if isinstance(attribute, str):
            if not attribute in dataarray.attrs or attribute in dataarray.dims:
                return dataarray
            attribute_list = [i for i in dataarray.attrs[attribute] if i in dataarray.dims]
            dim = {attribute: attribute_list}
        else: 
            dim = attribute

        dim_name = list(dim.keys())[0]
        
        if len(dim[dim_name]) != 0:
            da = dataarray.stack(**dim, **kwargs)
        else:
            da = dataarray.expand_dims(dim_name)

        return da
    
    def apply_filters(self, dataarray: xarray.DataArray) -> xarray.DataArray:
        """
        Apply the class filters on a given DataArray or Dataset.

        Parameters
        ----------
        dataarray : xarray.DataArray
            The DataArray to apply the filters on.

        Returns
        -------
        xarray.DataArray
            The filtered DataArray.
        """
        dimensions = dataarray.dims
        
        select_filters = {k: v for k, v in self.select_filters.items() if k in dimensions} if self.select_filters else None
        slice_filters = {k: v for k, v in self.slice_filters.items() if k in dimensions} if self.slice_filters else None

        if select_filters:
            dataarray = dataarray.sel(**select_filters)
        if slice_filters:
            slice_filters = transform_filters_to_slices(slice_filters)
            dataarray = dataarray.sel(**slice_filters)
        return dataarray

    @classmethod
    def flatten_output(cls, dataarray: xarray.DataArray, flat_output_dims: int) -> xarray.DataArray:
        """
        Flatten the output of a given DataArray by stacking all dimensions over attributes 'sample' and 'features',
        containing the list of dimensions to stack on.
        
        If flat_output_dims is 2, stacks on both 'sample' and 'features' attributes.
        If flat_output_dims is 1, stacks all dimensions into a single dimension 'dims'.
        Otherwise, returns the DataArray as is.

        Parameters
        ----------
        dataarray : xarray.DataArray
            The DataArray to flatten.
        flat_output_dims : int
            Number of dimensions to flatten the output on (1 or 2).

        Returns
        -------
        xarray.DataArray
            The flattened DataArray.
        """
        dataarray = dataarray.unstack()
        if flat_output_dims == 2:
            dataarray = cls.stack_on_attribute('sample', dataarray)
            dataarray = cls.stack_on_attribute('features', dataarray)
            dataarray = dataarray.transpose('sample', 'features')
        elif flat_output_dims == 1:
            dataarray = dataarray.stack(dims=[...])
        
        return dataarray

    def apply_indices_selection(self, dataarray: xarray.DataArray) -> xarray.DataArray:
        """
        Apply the indices selection on a given DataArray.
        Should be called after filters are applied and before flattening.
        Does nothing if select_indices is None.

        Parameters
        ----------
        dataarray : xarray.DataArray
            The DataArray to apply the indices selection on.

        Returns
        -------
        xarray.DataArray
            The DataArray with the selected indices.
        """
        if self.select_indices is None:
            return dataarray
        
        dataarray = self.stack_on_attribute('features', dataarray)
        # Warn if filters are applied to features dimensions
        if self.select_filters:
            features_filters = [k for k in dataarray.attrs['features'] if k in self.select_filters.keys()]
            if any(features_filters):
                self.logger.warning("Filters applied to features dimensions: {}".format(features_filters))
        if self.slice_filters:
            features_filters = [k for k in dataarray.attrs['features'] if k in self.slice_filters.keys()]
            if any(features_filters):
                self.logger.warning("Filters applied to features dimensions: {}".format(features_filters))

        return dataarray.isel(features=self.select_indices)

    def get_coordinate_list(self, name: str) -> list:
        """
        Returns the list of values of a coordinate of the dataset

        Parameters
        ----------
        name : str
            The name of the coordinate to retrieve.

        Returns
        -------
        list
            The list of values of the specified coordinate.
        """
        coordinate_list = self.coords[name].values.tolist() # pyright: ignore[reportAttributeAccessIssue] (DataArray.coords type is a DataArray object)
        
        if not isinstance(coordinate_list, list):
            coordinate_list = [coordinate_list]
        return coordinate_list

    @property
    def x_names(self) -> list:
        """
        Returns the list of the parameters coordinate of the x dataset.

        Returns
        -------
        list
            The list of the parameters of the x dataset.
        """
        return self.get_coordinate_list('parameters')

    @property
    def emulator_error(self):
        """
        Returns the emulator error of the statistic, with filters applied.
        Reads the emulator error from the error_dir if it is provided, otherwise uses the get_emulator_error method if implemented.
        """
        if hasattr(self._dataset, 'emulator_error'):
            data = self._dataset.emulator_error
            data = self.apply_filters(data)
            if 'emulator_error' in self.select_indices_on:
                data = self.apply_indices_selection(data)
            data = self.flatten_output(data, self.flat_output_dims)
            if self.squeeze_output:
                data = data.squeeze()
            if self.numpy_output:
                data = data.values
            return data
        elif hasattr(self, 'get_emulator_error'):
            return getattr(self, 'get_emulator_error')()
        else:
            raise NotImplementedError("No emulator error found. Please provide an error_dir or implement the get_emulator_error method.")
    
    @property
    def emulator_covariance_y(self):
        """
        Returns the covariance of the emulator error of the statistic, with filters applied.
        Reads the emulator covariance from the error_dir if it is provided, otherwise uses the get_emulator_covariance_y method if implemented.
        """
        if hasattr(self._dataset, 'emulator_covariance_y'):
            data = self._dataset.emulator_covariance_y
            data = self.apply_filters(data)
            if 'emulator_covariance_y' in self.select_indices_on:
                data = self.apply_indices_selection(data)
            data = self.flatten_output(data, self.flat_output_dims)
            if self.squeeze_output:
                data = data.squeeze()
            if self.numpy_output:
                data = data.values
            return data
        elif hasattr(self, 'get_emulator_covariance_y'):
            return getattr(self, 'get_emulator_covariance_y')()
        else:
            raise NotImplementedError("No emulator covariance found. Please provide an error_dir or implement the get_emulator_covariance_y method.")
    
    def get_model_prediction(self, x, model=None, coords: dict = None, attrs: dict = None, nofilters: bool = False):
        """
        Get the prediction from the model.
        
        Parameters
        ----------
        x : array_like, dict
            Input features for the model. 
            If an array, it should have shape (n_samples, n_params). 
            If a dict, it should have keys matching the model input names and values as lists/1d-arrays of shape (n_samples,).
        model : FCN
            Trained theory model. If None, the model attribute of the class is used. Defaults to None.
        coords : dict, optional
            Coordinates for the output DataArray. If None, the coordinates of _dataset.y are used. Defaults to None.
        attrs : dict, optional
            Attributes for the output DataArray. If None, the attributes of _dataset.y are used. Defaults to None.
        nofilters : bool, optional
            If True, no filters are applied to the output and the full DataArray is returned. Defaults to False.
        
        Returns
        -------
        array_like
            Model prediction.
        """
        if isinstance(x, dict):
            missing = set(self.x_names) - set(x.keys())
            extra = set(x.keys()) - set(self.x_names)
            if missing:
                raise ValueError(
                    "Input x dictionary keys do not match the model input names. "
                    f"Missing keys: {missing}"
                )
            if extra:
                self.logger.warning(
                    "Input x dictionary contains unexpected keys not used by the model. "
                    f"Unexpected keys: {extra}"
                )
            x = [x[name] for name in self.x_names]
            x = np.asarray(x).T  # Need to transpose to (n_samples, n_params)
        else:
            x = np.asarray(x)  # Ensure x is an array to make torch.Tensor faster
        
        if model is None:
            model = self.model

        with torch.no_grad():
            pred = model.get_prediction(torch.Tensor(x))
            pred = pred.numpy()
        
        if coords is None:
            coords = {
                **{k: self._dataset.y.coords[k] for k in self._dataset.y.dims if k in self._dataset.y.attrs['features']}
            }
        if attrs is None:
            attrs = {
                'sample': ['n_pred'],
                'features': self._dataset.y.attrs['features'],
            }

        n_pred = pred.shape[0] if len(pred.shape) > 1 else 1 # Edge case if only one prediction
        coords = {**{'n_pred': np.arange(n_pred)}, **coords} # Add extra coordinate for the number of predictions
        pred = pred.reshape([len(c) for c in coords.values()]) # reshape to the right shape
        pred = xarray.DataArray(
            pred, 
            coords = coords,
            attrs = attrs,
        )
        
        if nofilters:
            return pred
        
        pred = self.apply_filters(pred)
        pred = self.apply_indices_selection(pred)
        pred = self.flatten_output(pred, self.flat_output_dims)
        
        if self.squeeze_output:
            pred = pred.squeeze()
        if self.numpy_output:
            pred = pred.values
        return pred
    
    def get_covariance_matrix(self, volume_factor: float = 64, prefactor: float = 1, **kwargs) -> np.ndarray:
        """
        Covariance matrix for the statistic. 
        The prefactor is here for corrections if needed, and the volume factor is the volume correction of the boxes.
        
        Parameters
        ----------
        volume_factor : float
            Volume correction factor for the boxes. Default is 64.
        prefactor : float
            Prefactor to apply to the covariance matrix (e.g. Hartlap or Percival).
        **kwargs : dict
            Additional arguments for the covariance matrix checker.
            
        Returns
        -------
        np.ndarray
            The combined data covariance matrix.
        """   
        cov_y = self.covariance_y
        
        # Ensure 2D shape of the covariance array
        if isinstance(cov_y, xarray.DataArray):
            cov_y = self.flatten_output(cov_y, flat_output_dims=2) # Force 2D flattening for covariance
        elif len(cov_y.shape) > 2:
            self.logger.warning("Covariance array has more than 2 dimensions, reshaping to 2D assuming first dimension is the sample dimension.")
            cov_y = cov_y.reshape(cov_y.shape[0], -1) # Expect first dimension to be the sample dimension
        elif len(cov_y.shape) < 2:
            self.logger.error("Covariance array has less than 2 dimensions, covariance matrix computation might return some unexpected results.")
        
        prefactor = prefactor / volume_factor
        
        cov = prefactor * np.cov(cov_y, rowvar=False) # rowvar=False : each column is a variable and each row is an observation
        
        # Perform sanity checks on the covariance matrix
        check_covariance_matrix(cov, name=f"{self.stat_name} data covariance", **kwargs)
        
        return cov
    
    def get_emulator_covariance_matrix(self, prefactor: float = 1, method: str = 'median', diag: bool = False, **kwargs) -> np.ndarray:
        """
        Covariance matrix of the emulator residuals.

        Parameters
        ----------
        prefactor : float
            Prefactor to apply to the covariance matrix (e.g. Hartlap or Percival). Defaults to 1.
        method : str
            Method to compute the covariance matrix from the emulator residuals.
            Options include the mean absolute deviation ('mean'), median absolute deviation ('median'),
            or standard deviation ('stdev'). Defaults to 'median'.
        diag : bool
            If True, only the diagonal of the covariance matrix is computed. Defaults to False.
        **kwargs : dict
            Additional arguments for the covariance matrix checker.

        Returns
        -------
        np.ndarray
            The emulator covariance matrix.
        """
        cov_y = self.emulator_covariance_y
        
        # Ensure 2D shape of the covariance array
        if isinstance(cov_y, xarray.DataArray):
            cov_y = self.flatten_output(cov_y, flat_output_dims=2).values  # Force 2D flattening for covariance
        elif len(cov_y.shape) > 2:
            self.logger.warning("Covariance array has more than 2 dimensions, reshaping to 2D assuming first dimension is the sample dimension.")
            cov_y = cov_y.reshape(cov_y.shape[0], -1) # Expect first dimension to be the sample dimension
        elif len(cov_y.shape) < 2:
            self.logger.error("Covariance array has less than 2 dimensions, covariance matrix computation might return some unexpected results.")

        if method == 'median':
            if diag:
                mad = median_abs_deviation(cov_y, axis=0)
                mad *= 1/norm.ppf(3/4)   #  make summary consistent with stdev for a normal distribution
                cov = np.diag(mad ** 2)
            else:
                cov = orthogonal_gk_mad_covariance(cov_y)
        elif method == 'mean':
            if diag:
                mad = np.mean(np.abs(cov_y - np.mean(cov_y, axis=0)), axis=0)
                mad *= np.sqrt(np.pi/2)  # make summary consistent with stdev for a normal distribution
                cov = np.diag(mad ** 2)
            else:
                raise NotImplementedError("Mean absolute deviation covariance is not implemented for full matrix (diag=False).")
        elif method == 'stdev':
            if diag:
                std = np.std(cov_y, axis=0)
                cov = np.diag(std ** 2)
            else:
                cov = np.cov(cov_y, rowvar=False)
        else:
            raise ValueError(f"Unknown method '{method}' for emulator covariance matrix computation.")
        self.logger.info(f"Emulator covariance matrix computed using method '{method}' with diag={diag}.")
        
        cov *= prefactor 
        
        # Perform sanity checks on the covariance matrix
        check_covariance_matrix(cov, name=f"{self.stat_name} emulator covariance", **kwargs)
        
        return cov

    def get_save_handle(self, save_dir: str|Path = None) -> str|Path:
        """
        Creates a handle that includes the statistics and filters used.
        This can be used to save anything related to this observable.

        Parameters
        ----------
        save_dir : str
            Directory where the results will be saved.
            If provided, the directory is created if it does not exist.
            If None, the handle is returned as a string.
            Default is None.

        Returns
        -------
        str|Path
            The handle for saving the results, to be completed with the file extension.
            Returned as a Path instance if save_dir is provided as a Path.
        """
        slice_filters = self.slice_filters
        
        statistic_handle = self.stat_name
        if slice_filters:
            for key, value in slice_filters.items():
                statistic_handle += f'_{key}_{value[0]:.2f}-{value[1]:.2f}'
            # TODO : add select filters to the handle ?
        
        if save_dir is None:
            return statistic_handle
        
        # If save_path is provided, make sure it exists
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        cout = Path(save_dir) / f'{statistic_handle}'
        
        if isinstance(save_dir, str):
            return cout.as_posix() # Return as string if save_dir is a string
        return cout

    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_observable(self, model_params: dict, save_fn: str = None, **kwargs) -> tuple:
        """
        Plot the observable with error bars and the model prediction, along with the residuals.

        Parameters
        ----------
        model_params : dict
            Dictionary of model parameters for the prediction.
        save_fn : str, optional
            Filename to save the plot. If None, the plot is not saved.
        **kwargs : dict
            Additional arguments for the plot, such as height_ratios and show_legend.
            The parameters volume_factor and prefactor are passed to get_covariance_matrix() 
            to scale the covariance estimates.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, numpy.ndarray
            Figure and axes of the plot.
        """
        height_ratios = kwargs.pop('height_ratios', [3, 1])
        show_legend = kwargs.pop('show_legend', False)
        figsize = (6, 1.5 * sum(height_ratios))
        fig, ax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0.1)
        
        ax[-1].set_xlabel(r'$\textrm{bin index}$', fontsize=15)
        ax[0].set_ylabel(r'${\rm X}$', fontsize=15)
        
        data = self.y
        bin_idx = np.arange(len(data))
        model = self.get_model_prediction(model_params)
        
        if len(data.shape) > 1:
            self.logger.warning("Multiple samples found in the data. This might lead to unexpected plotting behavior.")
        
        volume_factor = kwargs.pop('volume_factor', 64)
        prefactor = kwargs.pop('prefactor', 1)
        cov = self.get_covariance_matrix(volume_factor=volume_factor, prefactor=prefactor)
        error = np.sqrt(np.diag(cov))
        
        ax[0].errorbar(bin_idx, data, error, marker='o', ms=4, ls='', color=f'C0', elinewidth=1.0, capsize=None)
        ax[0].plot(bin_idx, model, ls='-', color=f'C0')
        ax[1].plot(bin_idx, (data - model) / error, ls='-', color=f'C0')
        
        for offset in [-2, 2]: 
            ax[1].axhline(offset, color='k', ls='--')
            
        ax[1].set_ylabel(r'$\Delta{\rm X} / \sigma_{\rm data}$', fontsize=15)
        ax[1].set_ylim(-4, 4)
        
        for a in ax:
            a.grid(True)
            a.tick_params(axis='both', labelsize=14)
            
        if show_legend: 
            ax[0].legend(fontsize=15)
        
        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving plot to {save_fn}')
        return fig, ax
    
    @set_plot_style
    @temporary_class_state(flat_output_dims=2, numpy_output=False)
    def plot_emulator_residuals(self, save_fn: str = None, **kwargs) -> tuple:
        """
        Plot the emulator residuals.
        
        Parameters
        ----------
        save_fn : str
            Filename to save the plot. If None, the plot is not saved.
        **kwargs : dict
            Additional arguments for the plot, such as figsize, and volume_factor and prefactor for covariance calculation.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, numpy.ndarray
            Figure and axes of the plot.
        """

        volume_factor = kwargs.pop('volume_factor', 64)
        prefactor = kwargs.pop('prefactor', 1)
        data_cov = self.get_covariance_matrix(volume_factor=volume_factor, prefactor=prefactor)
        data_err = np.sqrt(np.diag(data_cov))
        residuals = self.emulator_covariance_y
        
        figsize = kwargs.pop('figsize', (4, 4))
        fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

        for res in residuals:
            ax[0].plot(res / data_err, color='gray', alpha=0.3, lw=0.5)

        # summary statistics of the emulator residuals
        for method in ['mean', 'median', 'stdev']:
            emu_cov = self.get_emulator_covariance_matrix(method=method, diag=True)
            emu_err = np.sqrt(np.diag(emu_cov))

            ax[1].plot(emu_err / data_err, lw=1.0, label=method)

        ax[1].axhline(1.0, color='k', ls=':', lw=0.7)
        ax[1].set_xlabel('bin index', fontsize=13)
        ax[0].set_ylabel(r'$\Delta X / \sigma_{\rm data}$', fontsize=13)
        ax[1].set_ylabel(r'$\sigma_{\rm emulator} / \sigma_{\rm data}$', fontsize=13)
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        if save_fn is not None:
            plt.savefig(save_fn, dpi=300, bbox_inches='tight')
            self.logger.info(f'Saving plot to {save_fn}')
        return fig, ax