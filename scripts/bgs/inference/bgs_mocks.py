import yaml
import logging
from pathlib import Path

import xarray
import numpy as np
from pycorr import TwoPointEstimator

from acm.observables import Observable, CombinedObservable
from acm.utils.covariance import check_covariance_matrix

logger = logging.getLogger(__file__.split('/')[-1])

# TODO: move this to acm.paths eventually ?
SECONDGEN_DIR = '/pscratch/sd/s/sbouchar/SecondGen/CubicBox/BGS/' 
UCHUU_DIR = '/pscratch/sd/s/sbouchar/UCHUU/CubicBox/BGS/'

def eval_str(s: str):
    if isinstance(s, str):
        return eval(s)
    return s

def compress_mock_x(
    mock_type: str,
    z: float = 0.2,
    cosmo: int = 0,
    Mr_cut: float = -20,
    sigma_to_log10: bool = True,
) -> xarray.DataArray:
    """
    Compress mock cosmology and HOD parameters into a DataArray.
    
    Parameters
    ----------
    mock_type : str
        Type of mock to load ('SecondGen' or 'Uchuu'). Will determine the data directory.
    z : float, optional
        Redshift of the measurements. Defaults to 0.2
    cosmo : int, optional
        Cosmology index. Defaults to 0
    Mr_cut : float, optional
        Magnitude cut for the sample. Defaults to -20
    sigma_to_log10 : bool, optional
        Whether to convert the 'sigma' parameter to its logarithm base 10. Defaults to True.
        
    Returns
    -------
    xarray.DataArray
        DataArray containing the cosmology and HOD parameters.
    """
    if mock_type == 'SecondGen':
        data_dir = Path(SECONDGEN_DIR)
        mock_name = f'AbacusSummit_base_c{cosmo:03d}_ph000'
    elif mock_type == 'Uchuu':
        data_dir = Path(UCHUU_DIR)
        mock_name = '' # TODO: TBD
    else:
        raise ValueError(f'Unknown mock type: {mock_type}')
    
    fn = data_dir / f'z{z:.3f}' / mock_name / 'parameters.yaml'
    with open(fn, 'rb') as f:
        params = yaml.safe_load(f)
    
    cosmo_params = params['cosmo_params']
    hod_params = params['hod_params'][f'Mr{Mr_cut}']
    
    all_params = {**cosmo_params, **hod_params}
    parameters = list(all_params.keys()) # Ordered to follow the BGS order
    values = np.array([eval_str(all_params[name]) for name in parameters])
    
    if sigma_to_log10:
        sigma_idx = parameters.index('sigma')
        values[sigma_idx] = np.log10(values[sigma_idx])
        logger.info(f'Converted sigma to log10(sigma): {values[sigma_idx]}')
    
    x = xarray.DataArray(
        data = values,
        coords = {
            'parameters': parameters,
        },
        attrs = {
            'sample': [],
            'features': ['parameters'],
        },
        name = 'x',
    )
    return x

def compress_mock_tpcf(
    mock_type: str,
    z: float = 0.2,
    cosmo: int = 0,
    phase: int = 0,
    Mr_cut: float = -20,
    los: list[str] = ['x', 'y', 'z'],
    rebin: int = 3,
    ells: list[int] = [0, 2],
    phase_range: tuple[int, int] = (0, 25),
    sigma_to_log10: bool = True,
) -> xarray.Dataset:
    """
    Compress mock two-point correlation function (tpcf) measurements.

    Parameters
    ----------
    mock_type : str
        Type of mock to load ('SecondGen' or 'Uchuu'). Will determine the data directory.
    z : float, optional
        Redshift of the measurements. Defaults to 0.2
    cosmo : int, optional
        Cosmology index. Defaults to 0
    phase : int, optional
        Phase index. Defaults to 0
    Mr_cut : float, optional
        Magnitude cut for the sample. Defaults to -20
    los : list[str], optional
        Lines of sight to consider. Defaults to ['x', 'y', 'z']
    rebin : int, optional
        Rebinning factor. Defaults to 3
    ells : list[int], optional
        Multipole moments to consider. Defaults to [0, 2]
    phase_range : tuple[int, int], optional
        Range of phases to use for covariance estimation. Defaults to (0, 25)
    sigma_to_log10 : bool, optional
        Whether to convert the 'sigma' parameter to its logarithm base 10. Defaults to True.
    """
    if mock_type == 'SecondGen':
        data_dir = Path(SECONDGEN_DIR)
        mock_name = f'AbacusSummit_base_c{cosmo:03d}_ph{phase:03d}'
        
    elif mock_type == 'Uchuu':
        data_dir = Path(UCHUU_DIR)
        mock_name = '' # TODO: TBD
    else:
        raise ValueError(f'Unknown mock type: {mock_type}')
    
    fns = [data_dir / f'z{z:.3f}' / mock_name / 'measurements' / f'Mr{Mr_cut}' / f'tpcf_los_{l}.npy' for l in los]
    data = sum([TwoPointEstimator.load(fn).normalize() for fn in fns if fn.exists()])
    if data == 0:
        raise FileNotFoundError(f'No measurement files found in {data_dir}, cannot compute covariance.')
    s, multipoles = data[::rebin](ells=ells, return_sep=True)
    
    y = xarray.DataArray(
        data = multipoles,
        coords = {
            'ells': ells,
            's': s,
        },
        attrs = {
            'sample': [],
            'features': ['ells', 's'],
        },
        name = 'y',
    )
    
    covariance_y = []  
    phase_idx = list(range(phase_range[0], phase_range[1]))  
    for phase_cov in phase_idx:
        if mock_type == 'SecondGen':
            mock_name_cov = f'AbacusSummit_base_c{cosmo:03d}_ph{phase_cov:03d}'
        elif mock_type == 'Uchuu':
            mock_name_cov = '' # TODO: TBD
            
        fns_cov = [data_dir / f'z{z:.3f}' / mock_name_cov / 'measurements' / f'Mr{Mr_cut}' / f'tpcf_los_{l}.npy' for l in los]
        data_cov = sum([TwoPointEstimator.load(fn).normalize() for fn in fns_cov if fn.exists()])
        if data_cov == 0:
            raise FileNotFoundError(f'No measurement files found in {data_dir}, cannot compute covariance.')
        multipoles_cov = data_cov[::rebin](ells=ells, return_sep=False) # Same s for all measurements
        covariance_y.append(multipoles_cov)
    covariance_y = np.array(covariance_y)
    
    covariance_y = xarray.DataArray(
        data = covariance_y,
        coords = {
            'phase_idx': phase_idx,
            'ells': ells,
            's': s,
        },
        attrs = {
            'sample': ['phase_idx'],
            'features': ['ells', 's'],
        },
        name = 'covariance_y',
    )
    
    x = compress_mock_x(
        mock_type = mock_type,
        z = z,
        cosmo = cosmo,
        Mr_cut = Mr_cut,
        sigma_to_log10 = sigma_to_log10,
    )
    
    logger.info(f'Loaded {mock_type} data with shape: {x.shape}, {y.shape}')
    logger.info(f'Loaded {mock_type} covariance array with shape: {covariance_y.shape}')
    
    cout = xarray.Dataset(
        data_vars = {
            'x': x,
            'y': y,
            'covariance_y': covariance_y,
        },
    )
    return cout


def compress_mock_ds(
    measurement_root: str,
    mock_type: str,
    z: float = 0.2,
    cosmo: int = 0,
    phase: int = 0,
    Mr_cut: float = -20,
    los: list[str] = ['x', 'y', 'z'],
    rebin: int = 1,
    ells: list[int] = [0, 2],
    quantiles: list[int] = [0, 1, 3, 4],
    add_covariance: bool = True,
    phase_range: tuple[int, int] = (0, 25),
    sigma_to_log10: bool = True,
) -> xarray.Dataset:
    if mock_type == 'SecondGen':
        data_dir = Path(SECONDGEN_DIR)
        mock_name = f'AbacusSummit_base_c{cosmo:03d}_ph{phase:03d}'
    elif mock_type == 'Uchuu':
        data_dir = Path(UCHUU_DIR)
        mock_name = '' # TODO: TBD
    else:
        raise ValueError(f'Unknown mock type: {mock_type}')
    
    fns = [data_dir / f'z{z:.3f}' / mock_name / 'measurements' / f'Mr{Mr_cut}' / f'{measurement_root}_los_{l}.npy' for l in los]
    y_quantiles = []
    for q in quantiles:
        data = sum([np.load(fn, allow_pickle=True)[q].normalize() for fn in fns if fn.exists()])
        if data == 0:
            raise FileNotFoundError(f'No measurement files found in {data_dir} for quantile {q}, cannot load data.')
        s, multipoles = data[::rebin](ells=ells, return_sep=True)
        y_quantiles.append(multipoles)
    y = np.array(y_quantiles)
    
    y = xarray.DataArray(
        data = y,
        coords = {
            'quantiles': quantiles,
            'ells': ells,
            's': s,
        },
        attrs = {
            'sample': [],
            'features': ['quantiles', 'ells', 's'],
        },
        name = 'y',
    )
    
    if add_covariance:
        covariance_y = []
        phase_idx = list(range(phase_range[0], phase_range[1]))
        for phase_cov in phase_idx:
            if mock_type == 'SecondGen':
                mock_name_cov = f'AbacusSummit_base_c{cosmo:03d}_ph{phase_cov:03d}'
            elif mock_type == 'Uchuu':
                mock_name_cov = '' # TODO: TBD
            
            fns_cov = [data_dir / f'z{z:.3f}' / mock_name_cov / 'measurements' / f'Mr{Mr_cut}' / f'{measurement_root}_los_{l}.npy' for l in los]
            y_quantiles = []
            for q in quantiles:
                data_cov = sum([np.load(fn, allow_pickle=True)[q].normalize() for fn in fns_cov if fn.exists()])
                if data_cov == 0:
                    raise FileNotFoundError(f'No measurement files found in {data_dir} for quantile {q}, cannot compute covariance.')
                multipoles_cov = data_cov[::rebin](ells=ells, return_sep=False) # Same s for all measurements
                y_quantiles.append(multipoles_cov)
            covariance_y.append(y_quantiles)
        covariance_y = np.array(covariance_y)
        
        covariance_y = xarray.DataArray(
            data = covariance_y,
            coords = {
                'phase_idx': phase_idx,
                'quantiles': quantiles,
                'ells': ells,
                's': s,
            },
            attrs = {
                'sample': ['phase_idx'],
                'features': ['quantiles', 'ells', 's'],
            },
            name = 'covariance_y',
        )
    
    x = compress_mock_x(
        mock_type = mock_type,
        z = z,
        cosmo = cosmo,
        Mr_cut = Mr_cut,
        sigma_to_log10 = sigma_to_log10,
    )
    
    logger.info(f'Loaded {mock_type} data with shape: {x.shape}, {y.shape}')
    logger.info(f'Loaded {mock_type} covariance array with shape: {covariance_y.shape}')
    
    cout = xarray.Dataset(
        data_vars = {
            'x': x,
            'y': y,
        },
    )
    if add_covariance:
        cout = xarray.merge([cout, covariance_y])
    return cout

def compress_mock(stat_name: str, **kwargs):
    """
    Compress mock statistics based on the statistic name.
    
    Parameters
    ----------
    stat_name : str
        Name of the statistic to compress. Options are 'tpcf', 'ds_xiqg', 'ds_xiqq'.
    **kwargs
        Additional keyword arguments to pass to the compression functions.
    
    Returns
    -------
    dataset : xarray.Dataset
        The compressed dataset for the specified statistic, containing the true values, compressed measurements, and covariance.
    """
    if stat_name == 'tpcf':
        dataset = compress_mock_tpcf(**kwargs)
    elif stat_name == 'ds_xiqg':
        dataset = compress_mock_ds(measurement_root='quantile_data_correlation', **kwargs)
    elif stat_name == 'ds_xiqq':
        dataset = compress_mock_ds(measurement_root='quantile_correlation', **kwargs)
    else:
        raise ValueError(f"Unknown statistic name: {stat_name}")
    return dataset

def check_datasets_compatibility(*observables) -> None:
    """
    Check if multiple observables are compatible for combination.
    Compatibility is defined as having the same features shapes for y, the same number of parameters for x, and the same covariance shape.
    
    Parameters
    ----------
    *observables : Observable
        List of observable objects to check for compatibility.
        
    Raises
    ------
    ValueError
        If the observables are not compatible, raises a ValueError with details.
    """
    ref = observables[0]
    for obs in observables[1:]:
        # Check the parameter order is the same
        # This also checks that the number of parameters is the same
        if ref.x_names != obs.x_names:
            raise ValueError(f'Incompatible x_names: {ref.x_names} vs {obs.x_names}')
        # Check the covariance shape is the same (volume factor is irrelevant here)
        cov_ref = ref.get_covariance_matrix(raise_warnings=False)
        cov_obs = obs.get_covariance_matrix(raise_warnings=False)
        if cov_ref.shape != cov_obs.shape:
            raise ValueError(f'Incompatible covariance shapes: {cov_ref.shape} vs {cov_obs.shape}')
        # Check the y features dimensions are the same HOW ?

def get_mock_data(observable, return_obs: bool = False, **kwargs) -> tuple: 
    """
    Load mock data (x, y) and covariance from the given directory.
    It will try to create an Observable instance matching the statistic names
    and shapes.

    Parameters
    ----------
    observable : CombinedObservable
        The observable object containing data. It will be used to determine the statistics to load.
    return_obs : bool, optional
        If True, returns only the constructed mock observable (for debugging). Defaults to False.
    **kwargs
        Additional keyword arguments for data loading.
        
    Returns
    -------
    x : np.ndarray
        The input data (e.g., cosmo and hod parameters).
    y : np.ndarray
        The observed data (e.g., clustering measurements).
    cov : np.ndarray
        The covariance matrix.
    """
    mock_type = kwargs.get('mock_type', 'SecondGen')
    
    stat_names = observable.stat_name
    if isinstance(stat_names, str):
        stat_names = [stat_names]
        observable = CombinedObservable([observable]) # Wrap in list for consistency
        
    observables = []
    for stat_name in stat_names:
        target_obs = observable[stat_name] # Get the specific observable
        tmp_obs = Observable(
            stat_name = stat_name,
            select_filters = target_obs.select_filters,
            slice_filters = target_obs.slice_filters,
            select_indices = target_obs.select_indices,
            select_indices_on = target_obs.select_indices_on,
            flat_output_dims = target_obs.flat_output_dims,
            squeeze_output = target_obs.squeeze_output,
            numpy_output = target_obs.numpy_output,
        ) # match the configuration of the target observable
        
        dataset = compress_mock(stat_name, **kwargs)
        tmp_obs._dataset = dataset # Assign the dataset directly
        
        check_datasets_compatibility(target_obs, tmp_obs)
        logger.info(f'Loaded compatible {mock_type} data for statistic: {stat_name}')
        observables.append(tmp_obs)
    
    mock_observable = CombinedObservable(observables)
    
    # Second compatibility check just to be sure
    check_datasets_compatibility(observable, mock_observable)
    logger.info(f'{mock_type} data loaded and compatible with the provided observable.')
    
    if return_obs:
        return mock_observable
    
    x = mock_observable.x
    x_names = mock_observable.x_names
    y = mock_observable.y
    cov = mock_observable.get_covariance_matrix(volume_factor=1) # No volume scaling needed for mock data
    
    # Diagonalize only if the combined covariance matrix fails checks,
    # as the combination of statistics might solve some individual issues.
    if not check_covariance_matrix(cov, name=f'{mock_type} covariance matrix', raise_warnings=False):
        logger.warning(f'The loaded {mock_type} covariance matrix failed some checks. Passing only the diagonal to mitigate issues.')
        cov = np.diag(np.diag(cov))
    
    logger.info(f'Loaded {mock_type} x with shape {x.shape} and names {x_names}')
    logger.info(f'Loaded {mock_type} y with shape {y.shape}')
    logger.info(f'Loaded {mock_type} covariance with shape {cov.shape}')
    
    return x, y, cov