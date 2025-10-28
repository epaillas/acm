import torch
import xarray
import numpy as np
from pathlib import Path 
from sunbird.emulators import FCN
from sunbird.data.data_utils import transform_filters_to_slices
from acm.utils.xarray_data import dataset_from_dict
import logging

class Observable():
    """
    Class to load a compressed Observable file or model and apply filters to their outputs.
    """

    def __init__(
        self,
        stat_name: str,
        paths: dict = None,
        select_filters: dict = None,
        slice_filters: dict = None,
        select_indices: list = None,
        select_indices_on: list = ['y', 'covariance_y', 'emulator_error', 'emulator_covariance_y'],
        flat_output_dims: int = None,
        squeeze_output: bool = False,
        numpy_output: bool = False,
    ):
        """
        Parameters
        ----------
        stat_name: str
            Name of the statistic to load. Also the name of the file containing the data.
        paths: dict, optional
            Paths to the compressed Observable files or models.
            If None, the internal dataset will be None. Defaults to None.
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
        
        Paths
        -----
        The data is expected to be in paths[key]/stat_name.npy, in which an xarray DataSet is stored.
        The possible keys are:
            - 'data_dir': directory containing the data (x, y)
            - 'covariance_dir': directory containing the covariance of the data (covariance_y)
            - 'error_dir': directory containing the emulator error of the data (emulator_error, emulator_covariance_y)
            - 'model_dir': directory containing the trained model (model.pth)
            - 'checkpoint_name': name of the checkpoint file (default: 'model.pth')

        Example
        -------
        ::
        
            slice_filters = {'sep': (0, 0.5),} 
            select_filters = {'multipoles': [0, 2],}


        will return the summary statistics for `0 < sep < 0.5` and multipoles 0 and 2
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.stat_name = stat_name
        self.paths = paths if paths else {} # Ensure dict behavior
        self.flat_output_dims = flat_output_dims
        self.numpy_output = numpy_output
        self.squeeze_output = squeeze_output

        # Try to read the paths with data inside
        datasets = []
        for key in ['data_dir', 'covariance_dir', 'error_dir']:
            if key not in self.paths:
                continue
            path = Path(self.paths[key]) / f"{self.stat_name}.npy"
        
            if path.exists():
                datasets.append(dataset_from_dict(np.load(path, allow_pickle=True).item()))
                self.logger.info(f"Loaded {key} from {path}")
        
        if len(datasets) == 0:
            self.logger.warning("No datasets found within provided paths.")
            self._dataset = None
        else:
            self._dataset = xarray.merge(datasets)
            self.logger.info("Datasets loaded with the following coordinates: {}".format(list(self._dataset.data_vars.keys())))
        
        # Load the model
        try:
            self.model = self.load_model() 
        except Exception as e: # handle the case where the model checkpoint is not found
            self.logger.warning(f"{e}, model will be undefined. If you are training a new model, this is expected.")
        
        # Set the filters
        self.select_filters = select_filters
        self.slice_filters = slice_filters
        self.select_indices = select_indices
        self.select_indices_on = select_indices_on if select_indices_on else [] # Ensure list behavior
        
        if self.select_indices and (self.select_filters or self.slice_filters):
            self.logger.warning("Indice selection and filters used at the same time. Check what you filter, you might not get the result you expect!")

    def __str__(self):
        """
        Returns a string representation of the object (statistic names and slice filters).
        """
        # TODO : improve this and __repr__ later ?
        return self.get_save_handle()

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
            
            data = self.flatten_output(data)
        
            if self.squeeze_output:
                data = data.squeeze()
                
            if self.numpy_output:
                data = data.values
            
        return data

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

    def flatten_output(self, dataarray: xarray.DataArray) -> xarray.DataArray:
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

        Returns
        -------
        xarray.DataArray
            The flattened DataArray.
        """
        dataarray = dataarray.unstack()
        if self.flat_output_dims == 2:
            dataarray = self.stack_on_attribute('sample', dataarray)
            dataarray = self.stack_on_attribute('features', dataarray)
            dataarray = dataarray.transpose('sample', 'features')
        elif self.flat_output_dims == 1:
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
        coordinate_list = self.coords[name].values.tolist()
        
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
            data = self.flatten_output(data)
            if self.squeeze_output:
                data = data.squeeze()
            if self.numpy_output:
                data = data.values
            return data
        elif hasattr(self, 'get_emulator_error'):
            return self.get_emulator_error()
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
            data = self.flatten_output(data)
            if self.squeeze_output:
                data = data.squeeze()
            if self.numpy_output:
                data = data.values
            return data
        elif hasattr(self, 'get_emulator_covariance_y'):
            return self.get_emulator_covariance_y()
        else:
            raise NotImplementedError("No emulator covariance found. Please provide an error_dir or implement the get_emulator_covariance_y method.")

    @property
    def checkpoint_fn(self) -> str:
        """
        Path to the checkpoint file of the model, constructed from the paths and the statistic name.
        """
        return self.paths['model_dir'] + f'{self.stat_name}/' + self.paths['checkpoint_name'] # FIXME : Update this format later
   
    def load_model(self, checkpoint_fn: str = None) -> FCN:
        """
        Trained theory model.
        """
        if checkpoint_fn is None:
            checkpoint_fn = self.checkpoint_fn
        
        # Load the model
        model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
        model.eval().to('cpu')
        if self.stat_name.startswith('minkowski'):
            from sunbird.data.transforms_array import WeiLiuInputTransform, WeiLiuOutputTransForm
            model.transform_output = WeiLiuOutputTransForm()
            model.transform_input = WeiLiuInputTransform()
        return model
    
    def get_model_prediction(self, x, model=None, coords=None, attrs=None, nofilters: bool = False) -> xarray.DataArray:
        """
        Get the prediction from the model.
        
        Parameters
        ----------
        x : array_like, dict
            Input features.
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
                logger.warning(
                    "Input x dictionary contains unexpected keys not used by the model. "
                    f"Unexpected keys: {extra}"
                )
            x = [x[name] for name in self.x_names]
            x = np.asarray(x).T  # Need to transpose to (n_samples, n_features)
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
        pred = self.flatten_output(pred)
        
        if self.squeeze_output:
            pred = pred.squeeze()
        if self.numpy_output:
            pred = pred.values
        return pred
    
    def get_covariance_matrix(self, volume_factor: float = 64, prefactor: float = 1) -> np.ndarray:
        """
        Covariance matrix for the statistic. 
        The prefactor is here for corrections if needed, and the volume factor is the volume correction of the boxes.
        """   
        cov_y = self.covariance_y
        
        # Ensure 2D shape of the covariance array
        if isinstance(cov_y, xarray.DataArray):
            cov_y = cov_y.unstack()
            cov_y = self.stack_on_attribute('sample', cov_y)
            cov_y = self.stack_on_attribute('features', cov_y)
            cov_y = cov_y.transpose('sample', 'features')
        elif len(cov_y.shape) > 2:
            self.logger.warning("Covariance array has more than 2 dimensions, reshaping to 2D assuming first dimension is the sample dimension.")
            cov_y = cov_y.reshape(cov_y.shape[0], -1) # Expect first dimension to be the sample dimension
        elif len(cov_y.shape) < 2:
            self.logger.error("Covariance array has less than 2 dimensions, covariance matrix computation might return some unexpected results.")
        
        prefactor = prefactor / volume_factor
        
        cov = prefactor * np.cov(cov_y, rowvar=False) # rowvar=False : each column is a variable and each row is an observation
        return cov
    
    def get_emulator_covariance_matrix(self, prefactor: float = 1) -> np.ndarray:
        """
        Emulator covariance matrix for the statistic. The prefactor is here for corrections if needed.
        """
        cov_y = self.emulator_covariance_y
        prefactor = prefactor
        
        # Ensure 2D shape of the covariance array
        if isinstance(cov_y, xarray.DataArray):
            cov_y = cov_y.unstack()
            cov_y = self.stack_on_attribute('sample', cov_y)
            cov_y = self.stack_on_attribute('features', cov_y)
            cov_y = cov_y.transpose('sample', 'features')
        elif len(cov_y.shape) > 2:
            self.logger.warning("Covariance array has more than 2 dimensions, reshaping to 2D assuming first dimension is the sample dimension.")
            cov_y = cov_y.reshape(cov_y.shape[0], -1) # Expect first dimension to be the sample dimension
        elif len(cov_y.shape) < 2:
            self.logger.error("Covariance array has less than 2 dimensions, covariance matrix computation might return some unexpected results.")
        
        cov = prefactor * np.cov(cov_y, rowvar=False)
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
