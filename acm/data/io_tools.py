from pathlib import Path
import numpy as np
import yaml

from sunbird.emulators import FCN
from sunbird.data.data_utils import convert_to_summary

from acm.data.default import cosmo_list, summary_coords_dict


fourier_stats = ['pk', 'dsc_pk']
conf_stats = ['tpcf', 'dsc_conf']

labels_stats = {
    'dsc_conf': 'Density-split',
    'dsc_pk': 'Density-split 'r'$P_\ell$',
    'dsc_conf_cross': 'Density-split (CCF)',
    'tpcf': 'Galaxy 2PCF',
    'tpcf+dsc_conf': 'DSC + Galaxy 2PCF',
    'number_density+tpcf': 'nbar + Galaxy 2PCF',
    'number_density+pk': 'nbar + P(k)',
    'pk': 'P(k)',
}

def summary_coords(
    statistic: str, 
    coord_type: str, 
    bin_values = None, # TODO : detect this in edge case later
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
        
    bin_values : _type_, optional
        Values of the bins on which the summary statistics are computed. 
        If set to None, the bin_values are not included in the summary coordinates. Defaults to None.
    summary_coords_dict : dict, optional
        Dictionary containing the summary coordinates for each statistic. It also contains the number of HODs, parameters and phases.
        Defaults to summary_coords_dict from `acm.data.default`.
        
    Returns
    -------
    dict
        Dictionary containing the summary coordinates for the given statistic and coordinate type.
    """
    hod_number = summary_coords_dict['hod_number']
    param_number = summary_coords_dict['param_number']
    phase_number = summary_coords_dict['phase_number']
    summary_coords_stat = summary_coords_dict['statistics']
    
    input_dict = {
        'cosmo_idx': cosmo_list,
        'hod_idx': list(range(hod_number)),
    }
    
    stat_dict = {
        **summary_coords_stat[statistic],
        'bin_values': bin_values,
    }
    
    # NOTE : Sometimes, bin_values is not needed !!
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
    elif coord_type == 'emulator_error':
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
                          add_statistic: bool = True) -> Path:
    """
    Finds the file name of the emulator error data for the emulator. The file name can be constructed from the statistic and the data directory.

    Parameters
    ----------
    statistic : str
        Statistic to read.
    error_dir : str
        Directory where the error is stored.
    add_statistic : bool, optional
        Weather to add the statistic to the file name. Defaults to True.

    Returns
    -------
    Path
        Path to the emulator error data file.
    """
    if add_statistic:
        error_dir = Path(error_dir) / f'{statistic}/'
    return Path(error_dir) / f'{statistic}_emulator_error.npy' 

def read_lhc(statistics: list, 
             data_dir: str,
             select_filters: dict = None, 
             slice_filters: dict = None, 
             return_mask: bool = False, 
             return_sep: bool = False,
             summary_coords_dict: dict = summary_coords_dict
             ) -> tuple:
    """
    Read the LHC data for the emulator. It has to have been saved as a dictionary with keys `s`, `lhc_x` and `lhc_y`.
    The `s` key is the separation array, `lhc_x` is the input features and `lhc_y` is the output features.
    The `lhc_x` and `lhc_y` arrays are sliced to the first `n_hod` elements, and thus must have the same length.
    The files read are `lhc_info_ccf.npy`, `lhc_info_acf.npy` and `lhc_info_tpcf.npy` for the CCF, ACF and TPCF respectively.

    Parameters
    ----------
    statistic : list
        Statistics to read. Will be concatenated in the output features in the given order.
    data_dir : str
        Directory where the data is stored.
    select_filters : dict, optional
        TODO
    slice_filters : dict, optional
        TODO
    return_mask : bool, optional
        Weather to return the mask array of the filtering process. Defaults to False.
    return_sep : bool, optional
        Weather to return the bin_values array. Defaults to False.
    summary_coords_dict : dict, optional
        Dictionary containing the summary coordinates for each statistic. It also contains the number of HODs, parameters and phases.
        Defaults to summary_coords_dict from `acm.data.default`.

    Returns
    -------
    Tuple
        Tuple of arrays with the input features, output features and separation array if `return_sep` or `return_mask` is True : 
        `(bin_values), lhc_x, lhc_y, lhc_x_names, (mask)` 
    """
    
    lhc_y_all = [] # List of the output features for all statistics
    mask_all = []
    
    for statistic in statistics:
        data_fn = lhc_fnames(statistic, data_dir)
        data = np.load(data_fn, allow_pickle=True).item()
        
        bin_values = data['bin_values']
        lhc_x = data['lhc_x']
        lhc_x_names = data['lhc_x_names']
        lhc_y = data['lhc_y']
        
        # Get the summary coordinates for the given statistic
        coords_y = summary_coords(statistic, coord_type='lhc_y', bin_values=bin_values, summary_coords_dict=summary_coords_dict)
        coords_x = summary_coords(statistic, coord_type='lhc_x', bin_values=bin_values, summary_coords_dict=summary_coords_dict)
        # If filters are provided, filter the data
        if coords_y and (select_filters or slice_filters):
            lhc_y, mask = filter_lhc(lhc_y, coords_y, select_filters, slice_filters)
            lhc_x, _ = filter_lhc(lhc_x, coords_x, select_filters, slice_filters)
            mask_all.append(mask)
        else:
            mask_all.append(np.full(lhc_y.shape, False))
        
        lhc_y_all.append(lhc_y)
    
    # Concatenate the output features for all statistics
    lhc_y_all = np.concatenate(lhc_y_all, axis=0) # TODO : check axis=0 or axis=1 ?? (or axis=-1)
    
    toret = (lhc_x, lhc_y_all, lhc_x_names)
    
    if return_mask:
        toret = (*toret, mask_all)
    if return_sep:
        toret = (bin_values, *toret)
    return toret


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
        TODO
    slice_filters : dict, optional
        TODO
    summary_coords_dict : dict, optional
        Dictionary containing the summary coordinates for each statistic. It also contains the number of HODs, parameters and phases.
        Defaults to summary_coords_dict from `acm.data.default`.

    Returns
    -------
    tuple
        Tuple of the covariance matrix and the number of data points in the output features.
        
    Volume factor
    -------------
    The volume factor is used to account for the fact that the covariance matrix is computed on a smaller volume than the data.
    The covariance matrix is computed on a volume of 500^3 Mpc/h, while the data is computed on a volume of 2000^3 Mpc/h.
    The covariance matrix is then scaled by the factor (2000/500)^3 = 64 to account for the difference in volume.
    
    """
    
    y_all = [] # List of the output features for all statistics
    for statistic in statistics:
        data_fn = lhc_fnames(statistic, data_dir)
        data = np.load(data_fn, allow_pickle=True).item()
        bin_values = data['bin_values']
        y = data['cov_y']
        
        # Get the summary coordinates for the given statistic
        coords = summary_coords(statistic, coord_type='smallbox', bin_values=bin_values, summary_coords_dict=summary_coords_dict)
        # If filters are provided, filter the data
        if coords and (select_filters or slice_filters):
            y, mask = filter_smallbox(y, coords, select_filters, slice_filters)
        
        y_all.append(y)
        
    # Concatenate the output features for all statistics
    y_all = np.concatenate(y_all, axis=1)
    
    prefactor = 1 / volume_factor
    cov = prefactor * np.cov(y_all, rowvar=False) # each line is a simulation, so rowvar=False
    
    return cov, len(y)


def read_model(statistics: list,
               model_path: dict | str,
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
    ```python
    >>> model_path = {'tpcf': '/ACM_pipeline/sunbird_training/models/cosmo+hod/last.ckpt',
    ...               'ccf': '/ACM_pipeline/sunbird_training/models/cosmo/last.ckpt'}
    >>> statistics = ['tpcf', 'acf']
    ```
    The model for the TPCF statistic will be loaded from the checkpoint file `/ACM_pipeline/sunbird_training/models/tpcf/cosmo+hod/last.ckpt`\n
    The model for the ACF statistic will be loaded from the checkpoint file `/ACM_pipeline/sunbird_training/models/acf/cosmo/last.ckpt`
    """
    
    # Handle the case where the model path is a string
    if isinstance(model_path, str):
        model_path = {statistic: model_path for statistic in statistics}
    
    model_all = []
    for statistic in statistics:        
        # Get the checkpoint file name
        checkpoint_fn = Path(model_path[statistic])

        # Load the model
        model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
        model.eval()
        # NOTE : There was a condition on the minkowski statistic here. Keep it ?
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
        TODO
    slice_filters : dict, optional
        TODO
    summary_coords_dict : dict, optional
        Dictionary containing the summary coordinates for each statistic. It also contains the number of HODs, parameters and phases.
        Defaults to summary_coords_dict from `acm.data.default`.

    Returns
    -------
    np.ndarray
        Array of the output features for all statistics.
    """
    y_all = [] # List of the output features for all statistics
    
    for statistic in statistics:
        data_fn = emulator_error_fnames(statistic, error_dir)
        data = np.load(data_fn, allow_pickle=True).item()
        bin_values = data['bin_values']
        y = data['emulator_error']
        
        # Get the summary coordinates for the given statistic
        coords = summary_coords(statistic, coord_type='emulator_error', bin_values=bin_values, summary_coords_dict=summary_coords_dict)
        # If filters are provided, filter the data
        if coords and slice_filters:
            y, mask = filter_emulator_error(y, coords, select_filters, slice_filters)
            
        y_all.append(y)
        
    # Concatenate the output features for all statistics
    y_all = np.concatenate(y_all, axis=0)
    
    return y_all


def filter_lhc(lhc_y, coords, select_filters, slice_filters):
    select_filters = {key: value for key, value in select_filters.items() if key in coords}
    slice_filters = {key: value for key, value in slice_filters.items() if key in coords}
    dimensions = list(coords.keys())
    lhc_y = lhc_y.reshape([len(coords[d]) for d in dimensions])
    lhc_y = convert_to_summary(data=lhc_y, dimensions=dimensions, coords=coords)
    if select_filters:
        select_filters = [getattr(getattr(lhc_y, key), 'isin')(value) for key, value in select_filters.items()]
        for i, cond in enumerate(select_filters):
            select_mask = select_mask & cond if i > 0 else select_filters[0]
        select_mask = lhc_y.where(select_mask).to_masked_array().mask
    else:
        select_mask = np.full(lhc_y.shape, False)
    if slice_filters:
        slice_filters = [(getattr(lhc_y, key) >= value[0]) & (getattr(lhc_y, key) <= value[1]) for key, value in slice_filters.items()]
        for i, cond in enumerate(slice_filters):
            slice_mask = slice_mask & cond if i > 0 else slice_filters[0]
        slice_mask = lhc_y.where(slice_mask).to_masked_array().mask
    else:
        slice_mask = np.full(lhc_y.shape, False)
    mask = select_mask | slice_mask
    return lhc_y.values[~mask], mask[np.where(~mask)[0][0], np.where(~mask)[1][0]].reshape(-1)

def filter_smallbox(lhc_y, coords, select_filters, slice_filters):
    select_filters = {key: value for key, value in select_filters.items() if key in coords}
    slice_filters = {key: value for key, value in slice_filters.items() if key in coords}
    dimensions = list(coords.keys())
    lhc_y = lhc_y.reshape([len(coords[d]) for d in dimensions])
    lhc_y = convert_to_summary(data=lhc_y, dimensions=dimensions, coords=coords)
    if select_filters:
        select_filters = [getattr(getattr(lhc_y, key), 'isin')(value) for key, value in select_filters.items()]
        for i, cond in enumerate(select_filters):
            select_mask = select_mask & cond if i > 0 else select_filters[0]
        select_mask = lhc_y.where(select_mask).to_masked_array().mask
    else:
        select_mask = np.full(lhc_y.shape, False)
    if slice_filters:
        slice_filters = [(getattr(lhc_y, key) >= value[0]) & (getattr(lhc_y, key) <= value[1]) for key, value in slice_filters.items()]
        for i, cond in enumerate(slice_filters):
            slice_mask = slice_mask & cond if i > 0 else slice_filters[0]
        slice_mask = lhc_y.where(slice_mask).to_masked_array().mask
    else:
        slice_mask = np.full(lhc_y.shape, False)
    mask = select_mask | slice_mask
    return lhc_y.values[~mask].reshape(lhc_y.shape[0], -1), mask[0]


def filter_emulator_error(y, coords, select_filters, slice_filters):
    if coords:
        select_filters = {key: value for key, value in select_filters.items() if key in coords}
        slice_filters = {key: value for key, value in slice_filters.items() if key in coords}
    dimensions = list(coords.keys())
    y = y.reshape([len(coords[d]) for d in dimensions])
    y = convert_to_summary(data=y, dimensions=dimensions, coords=coords)
    if select_filters:
        select_filters = [getattr(getattr(y, key), 'isin')(value) for key, value in select_filters.items()]
        for i, cond in enumerate(select_filters):
            select_mask = select_mask & cond if i > 0 else select_filters[0]
        select_mask = y.where(select_mask).to_masked_array().mask
    else:
        select_mask = np.full(y.shape, False)
    if slice_filters:
        slice_filters = [(getattr(y, key) >= value[0]) & (getattr(y, key) <= value[1]) for key, value in slice_filters.items()]
        for i, cond in enumerate(slice_filters):
            slice_mask = slice_mask & cond if i > 0 else slice_filters[0]
        slice_mask = y.where(slice_mask).to_masked_array().mask
    else:
        slice_mask = np.full(y.shape, False)
    mask = select_mask | slice_mask
    return y.values[~mask], mask


def read_chain(chain_fn, return_labels=False):
    from getdist import MCSamples
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

