from pathlib import Path
import numpy as np
import yaml
from getdist import MCSamples

from sunbird.emulators import FCN
from sunbird.data.data_utils import convert_to_summary

from acm.data.default import summary_coords_dict


def summary_coords(
    statistic: str, 
    coord_type: str, 
    bin_values = None,
    summary_coords_dict: dict = summary_coords_dict 
    ) -> dict:
    """
    Finds the summary coordinates for the given statistic and coordinate type.
    Returns a dictionary containing the summary coordinates, in a format that can be used to reshape the data. (see `filter_data`)

    Parameters
    ----------
    statistic : str
        Statistic name
    coord_type : str
        Type of coordinates for which to find the coordinates.
        can be set to : 
        - `'lhc_y'` will return the summary coordinates for the LHC data (cosmo_idx, hod_idx, statistics and bin_values).
        - `'lhc_x'` will return the summary coordinates for the LHC data (cosmo_idx, hod_idx, param_idx).
        - `'smallbox'` will return the summary coordinates for the small box data (phase_idx, statistics and bin_values). 
        - `'emulator_error'` will return the summary coordinates for the emulator error data (statistics and bin_values).
        - `'statistic'` will return the summary coordinates for the statistic (statistics and bin_values).
        
    bin_values : _type_, optional
        Values of the bins on which the summary statistics are computed. 
        If set to None, the bin_values are not included in the summary coordinates. Defaults to None.
    summary_coords_dict : dict, optional
        Dictionary containing the summary coordinates for each statistic. 
        It also contains the comology indexes, the number of HODs, parameters and phases.
        Defaults to summary_coords_dict from `acm.data.default`.
        
    Returns
    -------
    dict
        Dictionary containing the summary coordinates for the given statistic and coordinate type.
    """
    
    cosmo_idx = summary_coords_dict['cosmo_idx']
    hod_number = summary_coords_dict['hod_number']
    param_number = summary_coords_dict['param_number']
    phase_number = summary_coords_dict['phase_number']
    summary_coords_stat = summary_coords_dict['statistics']
    
    input_dict = {
        'cosmo_idx': cosmo_idx,
        'hod_idx': list(range(hod_number)),
    }
    
    stat_dict = {
        **summary_coords_stat[statistic],
        'bin_values': bin_values,
    }
    
    # Sometimes, bin_values is not needed !!
    if bin_values is None:
        stat_dict.pop('bin_values')
    
    param_dict = {
        'param_idx': list(range(param_number)),
    }
    
    phase_dict = {
        'phase_idx': list(range(phase_number)),
    }
    
    if coord_type == 'lhc_y':
        return {**input_dict, **stat_dict}
    elif coord_type == 'lhc_x':
        return {**input_dict, **param_dict}
    elif coord_type == 'smallbox':
        return {**phase_dict, **stat_dict}
    elif coord_type == 'emulator_error' or coord_type == 'statistic':
        return {**stat_dict}
    else:
        raise ValueError(f'Unknown coord_type: {coord_type}')


def lhc_fnames(statistic: str, 
               data_dir: str) -> Path:
    """
    Finds the file name of the LHC data for the emulator. The file name is constructed from the statistic and the data directory.

    Parameters
    ----------
    statistic : str
        Statistic to read.
    data_dir : str
        Directory where the data is stored.

    Returns
    -------
    Path
        Path to the LHC data file.
    """
    
    return Path(data_dir) / f'{statistic}_lhc.npy' 


def emulator_error_fnames(statistic: str, 
                          error_dir: str,
                          add_statistic: bool = False) -> Path:
    """
    Finds the file name of the emulator error data for the emulator. The file name can be constructed from the statistic and the data directory.

    Parameters
    ----------
    statistic : str
        Statistic to read.
    error_dir : str
        Directory where the error is stored.
    add_statistic : bool, optional
        Weather to add the statistic to the file name (for retrocompatibility). Defaults to False.

    Returns
    -------
    Path
        Path to the emulator error data file.
    """
    
    if add_statistic:
        error_dir = Path(error_dir) / f'{statistic}/'
    return Path(error_dir) / f'{statistic}_emulator_error.npy' 


def get_bin_values(data: dict) -> np.ndarray:
    """
    Get the bin values from the data dictionary. The bin values are stored in the `bin_values` key.
    This fuction is here to avoid errors later if there is not bin_values key in the data dictionary.

    Parameters
    ----------
    data : dict
        Data dictionary containing the bin values.

    Returns
    -------
    np.ndarray
        Array of the bin values, or None if the key is not present.
    """
    if 'bin_values' not in data:
        return None
    return data['bin_values']


def read_lhc(statistics: list, 
             data_dir: str,
             select_filters: dict = None, 
             slice_filters: dict = None, 
             return_sep: bool = False,
             summary_coords_dict: dict = summary_coords_dict
             ) -> tuple:
    """
    Read the LHC data for the emulator. It has to have been saved as a dictionary with keys `bin_values`, `lhc_x` and `lhc_y`.
    The `bin_values` key is the bins on which the statistic is computed, `lhc_x` is the input features and `lhc_y` is the output features.
    The `lhc_x` and `lhc_y` arrays are sliced to the first `n_hod` elements, and thus must have the same length.
    The files read are `lhc_info_ccf.npy`, `lhc_info_acf.npy` and `lhc_info_tpcf.npy` for the CCF, ACF and TPCF respectively.

    Parameters
    ----------
    statistic : list
        Statistics to read. Will be concatenated in the output features in the given order.
    data_dir : str
        Directory where the data is stored.
    select_filters : dict, optional
        Filters to select values in coordinates. Defaults to None.
    slice_filters : dict, optional
       Filters to slice values in coordinates. Defaults to None.
    return_sep : bool, optional
        Wether to return the bin_values array. Defaults to False.
    summary_coords_dict : dict, optional
        Dictionary containing the summary coordinates for each statistic. 
        It also contains the comology indexes, the number of HODs, parameters and phases.
        Defaults to `summary_coords_dict` from `acm.data.default`.

    Returns
    -------
    tuple
        Tuple of arrays with the input features, output features and bin_values array if `return_sep` is True : 
        `(bin_values), lhc_x, lhc_y, lhc_x_names` 
    
    Example
    -------
    ::
    
        slice_filters = {'bin_values': (0, 0.5),} 
        select_filters = {'multipoles': [0, 2],}
    
    
    will return the summary statistics for `0 < bin_values < 0.5` and multipoles 0 and 2
    """
   
    lhc_y_all = [] # List of the output features for all statistics
    
    for statistic in statistics:
        data_fn = lhc_fnames(statistic, data_dir)
        data = np.load(data_fn, allow_pickle=True).item()
        
        bin_values = get_bin_values(data)
        lhc_x_names = data['lhc_x_names']
        lhc_x = data['lhc_x']
        lhc_y = data['lhc_y']
        
        # If filters are provided, filter the data
        if select_filters or slice_filters: 
            # Get the summary coordinates for the given statistic
            coords_x = summary_coords(statistic, coord_type='lhc_x', bin_values=bin_values, summary_coords_dict=summary_coords_dict)
            coords_y = summary_coords(statistic, coord_type='lhc_y', bin_values=bin_values, summary_coords_dict=summary_coords_dict)
            # lhc_x can also be filtered ! (for example, to select only some cosmologies)
            lhc_x = filter(lhc_x, coords_x, select_filters, slice_filters)
            lhc_y = filter(lhc_y, coords_y, select_filters, slice_filters)
            # Filter the bin_values too
            if bin_values is not None:
                coords_bin = {'bin_values': bin_values}
                bin_values = filter(bin_values, coords_bin, select_filters, slice_filters)
            
        lhc_y_all.append(lhc_y)
    
    # Concatenate the output features for all statistics
    lhc_y_all = np.concatenate(lhc_y_all, axis=-1) 
    
    toret = (lhc_x, lhc_y_all, lhc_x_names)
    
    if return_sep:
        toret = (bin_values, *toret)
    return toret


def read_covariance_y(statistic: str,
                      data_dir: str,
                      select_filters: dict = None,
                      slice_filters: dict = None,
                      summary_coords_dict: dict = summary_coords_dict
                      ) -> np.ndarray:
    """
    Reads the data covariance array from the lhc file, stored under the `cov_y` key.

    Parameters
    ----------
    statistic : str
        Statistic to read.
    data_dir : str
        Directory where the data is stored.
    select_filters : dict, optional
        Filters to select values in coordinates. Defaults to None.
    slice_filters : dict, optional
       Filters to slice values in coordinates. Defaults to None.
    summary_coords_dict : dict, optional
        Dictionary containing the summary coordinates for each statistic. 
        It also contains the comology indexes, the number of HODs, parameters and phases.
        Defaults to `summary_coords_dict` from `acm.data.default`.

    Returns
    -------
    np.ndarray
        Array of covariance array for the given statistic.
    
    Example
    -------
    ::
    
        slice_filters = {'bin_values': (0, 0.5),} 
        select_filters = {'multipoles': [0, 2],}
    
    
    will return the summary statistics for `0 < bin_values < 0.5` and multipoles 0 and 2
    """
    data_fn = lhc_fnames(statistic, data_dir)
    data = np.load(data_fn, allow_pickle=True).item()
    bin_values = get_bin_values(data)
    y = data['cov_y']
    
    # If filters are provided, filter the data
    if select_filters or slice_filters: 
        # Get the summary coordinates for the given statistic
        coords = summary_coords(statistic, coord_type='smallbox', bin_values=bin_values, summary_coords_dict=summary_coords_dict)
        y = filter(y, coords, select_filters, slice_filters)
    
    return y

def read_covariance(statistics: list,
                    data_dir: str, 
                    volume_factor: float = 64, 
                    select_filters: dict = None, 
                    slice_filters: dict = None,
                    summary_coords_dict: dict = summary_coords_dict
                    ) -> tuple:
    """
    Read the covariance matrix from the data. The covariance matrix is computed from the output features of the LHC data.
    A volume factor is applied to account for the fact that the covariance matrix is computed on a smaller volume than the data.

    Parameters
    ----------
    statistics : list
        Statistics to read. Will be concatenated in the output features in the given order.
    data_dir : str
        Directory where the data is stored.
    volume_factor : int, optional
        Volume factor to account for the fact that the covariance matrix is computed on a smaller volume than the data. Defaults to 64
    select_filters : dict, optional
        Filters to select values in coordinates. Defaults to None.
    slice_filters : dict, optional
       Filters to slice values in coordinates. Defaults to None.
    summary_coords_dict : dict, optional
        Dictionary containing the summary coordinates for each statistic. 
        It also contains the comology indexes, the number of HODs, parameters and phases.
        Defaults to `summary_coords_dict` from `acm.data.default`.

    Returns
    -------
    tuple
        Tuple of the covariance matrix and the number of data points in the output features.
    
    Example
    -------
    ::
    
        slice_filters = {'bin_values': (0, 0.5),} 
        select_filters = {'multipoles': [0, 2],}
    
    
    will return the summary statistics for `0 < bin_values < 0.5` and multipoles 0 and 2
    
    Volume factor
    -------------
    The volume factor is used to account for the fact that the covariance matrix is computed on a smaller volume than the data.
    The covariance matrix is computed on a volume of 500^3 Mpc/h, while the data is computed on a volume of 2000^3 Mpc/h.
    The covariance matrix is then scaled by the factor (2000/500)^3 = 64 to account for the difference in volume.
    
    """
    
    y_all = [] # List of the output features for all statistics
    for statistic in statistics:
        y = read_covariance_y(statistic, data_dir, select_filters, slice_filters, summary_coords_dict)
        
        y_all.append(y)
        
    # Concatenate the output features for all statistics
    y_all = np.concatenate(y_all, axis=-1)
    
    prefactor = 1 / volume_factor
    cov = prefactor * np.cov(y_all, rowvar=False) # each row is a simulation, so rowvar=False
    
    return cov, len(y)


def read_model(statistics: list,
               model_fn: dict | str,
               ) -> list:
    """
    Load the model from the checkpoint file. The checkpoint file is constructed from the model directory, the statistic, the model subdirectory and the checkpoint name.

    Parameters
    ----------
    statistics : list
        Statistics to load the model for. The models will be loaded in the given order.
    model_path : dict | str
        Path to the model directory. If a dictionary is given, the model subdirectory for each statistic is taken from the dictionary.
        If a string is given, the model subdirectory is the same for all statistics. (can be useful if only one statistic is used)
    
    Returns
    -------
    list
        List of the models for all statistics.
    
    Example
    -------
    ::
    
        >>> model_path = {'tpcf': '/ACM_pipeline/sunbird_training/models/cosmo+hod/last.ckpt',
        ...               'ccf': '/ACM_pipeline/sunbird_training/models/cosmo/last.ckpt'}
        >>> statistics = ['tpcf', 'acf']
    
    
    The model for the TPCF statistic will be loaded from the checkpoint file `/ACM_pipeline/sunbird_training/models/tpcf/cosmo+hod/last.ckpt`\n
    The model for the ACF statistic will be loaded from the checkpoint file `/ACM_pipeline/sunbird_training/models/acf/cosmo/last.ckpt`
    """
    
    # Handle the case where the model path is a string
    if isinstance(model_fn, str):
        model_fn = {statistic: model_fn for statistic in statistics}
    
    model_all = []
    for statistic in statistics:        
        # Get the checkpoint file name
        checkpoint_fn = Path(model_fn[statistic])

        # Load the model
        model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
        model.eval().to('cpu')
        if statistic == 'minkowski':
            from sunbird.data.transforms_array import WeiLiuInputTransform, WeiLiuOutputTransForm
            model.transform_output = WeiLiuOutputTransForm()
            model.transform_input = WeiLiuInputTransform()
        model_all.append(model)
    return model_all


def read_emulator_error(statistics: list,
                        error_dir: str,
                        select_filters: dict = None, 
                        slice_filters: dict = None,
                        summary_coords_dict: dict = summary_coords_dict
                        ) -> np.ndarray:
    """
    Read the emulator error data for the emulator. The emulator error data is saved as a dictionary with the key `emulator_error`.

    Parameters
    ----------
    statistics : list
        Statistics to read. Will be concatenated in the output features in the given order.
    error_dir : str
        Directory where the error is stored. 
    select_filters : dict, optional
        Filters to select values in coordinates. Defaults to None.
    slice_filters : dict, optional
       Filters to slice values in coordinates. Defaults to None.
    summary_coords_dict : dict, optional
        Dictionary containing the summary coordinates for each statistic. 
        It also contains the comology indexes, the number of HODs, parameters and phases.
        Defaults to `summary_coords_dict` from `acm.data.default`.

    Returns
    -------
    np.ndarray
        Array of the output features for all statistics.
    
    Example
    -------
    ::
    
        slice_filters = {'bin_values': (0, 0.5),} 
        select_filters = {'multipoles': [0, 2],}
    
    
    will return the summary statistics for `0 < bin_values < 0.5` and multipoles 0 and 2
    """
    
    y_all = [] # List of the output features for all statistics
    
    for statistic in statistics:
        data_fn = emulator_error_fnames(statistic, error_dir)
        data = np.load(data_fn, allow_pickle=True).item()
        bin_values = get_bin_values(data)
        y = data['emulator_error']
        
        # If filters are provided, filter the data
        if slice_filters or slice_filters: 
            # Get the summary coordinates for the given statistic
            coords = summary_coords(statistic, coord_type='emulator_error', bin_values=bin_values, summary_coords_dict=summary_coords_dict)
            y = filter(y, coords, select_filters, slice_filters)
            
        y_all.append(y)
        
    # Concatenate the output features for all statistics
    y_all = np.concatenate(y_all, axis=-1)
    
    return y_all

def read_emulator_covariance_y(statistic: str,
                               error_dir: str,
                               select_filters: dict = None,
                               slice_filters: dict = None,
                               summary_coords_dict: dict = summary_coords_dict
                               ) -> np.ndarray:
    data_fn = emulator_error_fnames(statistic, error_dir)
    data = np.load(data_fn, allow_pickle=True).item()
    bin_values = get_bin_values(data)
    y = data['emulator_cov_y']
    
    # If filters are provided, filter the data
    if slice_filters or slice_filters: 
        # Get the summary coordinates for the given statistic
        coords = summary_coords(statistic, coord_type='emulator_error', bin_values=bin_values, summary_coords_dict=summary_coords_dict)
        coords = {'n_extra': list(range(len(y))), **coords} # Add the n_test dimension to the coords (to allow reshaping in filter)
        y = filter(y, coords, select_filters, slice_filters)
        # NOTE : move n_test to summary_coords_dict ? ideally we should not have to modify it anyways
    return y

def read_emulator_covariance(statistics: list,
                             error_dir: str,
                             select_filters: dict = None, 
                             slice_filters: dict = None,
                             summary_coords_dict: dict = summary_coords_dict
                             ) -> tuple:
    """
    Read the covariance matrix from the data. The covariance matrix is computed from the output features of the emulator error data.
    No volume factor is applied to the covariance matrix, because the emulator error data is already computed on the same volume as the data.

    Parameters
    ----------
    statistics : list
        Statistics to read. Will be concatenated in the output features in the given order.
    error_dir : str
        Directory where the error is stored. 
    select_filters : dict, optional
        Filters to select values in coordinates. Defaults to None.
    slice_filters : dict, optional
       Filters to slice values in coordinates. Defaults to None.
    summary_coords_dict : dict, optional
        Dictionary containing the summary coordinates for each statistic. 
        It also contains the comology indexes, the number of HODs, parameters and phases.
        Defaults to `summary_coords_dict` from `acm.data.default`.

    Returns
    -------
    tuple
        Tuple of the covariance matrix and the number of data points in the output features.
    
    Example
    -------
    ::
    
        slice_filters = {'bin_values': (0, 0.5),} 
        select_filters = {'multipoles': [0, 2],}
    
    
    will return the summary statistics for `0 < bin_values < 0.5` and multipoles 0 and 2
    """
    
    y_all = [] # List of the output features for all statistics
    
    for statistic in statistics:
        y = read_emulator_covariance_y(statistic, error_dir, select_filters, slice_filters, summary_coords_dict)
            
        y_all.append(y)
    
    # Concatenate the output features for all statistics
    y_all = np.concatenate(y_all, axis=-1)

    cov = np.cov(y_all, rowvar=False) # each line is a simulation, so rowvar=False
    
    return cov, len(y) 


def filter(y,
           coords: dict, # Order of the keys is VERY important here !!
           select_filters: dict = None, 
           slice_filters: dict = None,
           n_sim: int = None,
           ) -> tuple:
    """
    Filter the data based on the given filters. The filters are applied to the data based on the summary coordinates.

    Parameters
    ----------
    y : Array-like
        Data to filter.
    coords : dict
        Dictionary containing the summary coordinates for the data. Obtained from `summary_coords`.
    select_filters : dict, optional
        Filters to select values in coordinates. Defaults to None.
    slice_filters : dict, optional
       Filters to slice values in coordinates. Defaults to None.
    n_sim : int, optional
        Final number of simulations (in case the keys of the coords are not the default ones). 
        If not provided, it is computed from the data coords keys. 
        If it cannot be computed, it is set to 1. Defaults to None.

    Returns
    -------
    np.ndarray
        Filtered data, reshaped in the shape (n_sim, n_statistics), where n_statistics is the concatenated length of the statistics.
        This format is the expected format for the emulator.
        
    Example
    -------
    ::
    
        slice_filters = {'bin_values': (0, 0.5),} 
        select_filters = {'multipoles': [0, 2],}
    
    
    will return the summary statistics for `0 < bin_values < 0.5` and multipoles 0 and 2
    """
    
    # Convert the data to a summary object
    dimensions = list(coords.keys())
    y = y.reshape([len(coords[d]) for d in dimensions])
    y = convert_to_summary(
        data=y,
        dimensions=dimensions, 
        coords=coords, 
        select_filters=select_filters,
        slice_filters=slice_filters) # Filter the data
    
    # Figure out the number of simulations (first dimension of the data)
    if n_sim is not None and n_sim > 1: # > 1 because 1 is the default value
        n_sim = n_sim # NOTE : this is just in case we ever need to filter by hand (which should not happen ideally)
    elif 'cosmo_idx' and 'hod_idx' in y.sizes: # LHC
        n_sim = y.sizes['cosmo_idx'] * y.sizes['hod_idx']
    elif 'phase_idx' in y.sizes: # Smallbox
        n_sim = y.sizes['phase_idx']
    elif 'n_test' in y.sizes: # Emulator error
        n_sim = y.sizes['n_test']
    else: # assume 1 otherwise
        n_sim = 1
        
    # Reshape the data to the expected format 
    if n_sim >1 :
        y = y.values.reshape(n_sim, -1) # Concatenate the data on the last axis 
    else: # Edge case where there is only one simulation
        y = y.values.reshape(-1)
        
    return y 


def get_covariance_correction(n_s, n_d, n_theta=None, method='percival'):
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
    elif method == 'hartlap':
        return (n_s - 1)/(n_s - n_d - 2)


def correlation_from_covariance(covariance):
    """
    Compute the correlation matrix from the covariance matrix.

    Parameters
    ----------
    covariance : array_like
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Correlation matrix.
    """
    
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def read_chain(chain_fn: str|Path, 
               return_labels: bool = False):
    """
    Read the chain from the given file name. 
    The chain is saved as a dictionary with the keys `samples`, `weights`, `names`, `ranges` and `labels`.

    Parameters
    ----------
    chain_fn : str|Path
        File name of the chain to read.
    return_labels : bool, optional
        Weather to return the labels of the chain. Defaults to False.

    Returns
    -------
    MCSamples
        Chain read from the file.
    dict
        Dictionary containing the labels of the chain if `return_labels` is True.
    """
    
    data = np.load(chain_fn, allow_pickle=True).item()
    chain = MCSamples(
                samples=data['samples'],
                weights=data['weights'],
                names=data['names'],
                ranges=data['ranges'],
                labels=data['labels'],
            )
    if return_labels:
        return chain, data['labels']
    return chain

