from pathlib import Path
import numpy as np
from sunbird.data.data_utils import convert_to_summary
import yaml

from acm.data.default import cosmo_list, summary_coords_stat


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
    hod_number: int = 100, 
    param_number: int = 20,
    phase_number: int = 1786,
    summary_coords_stat: dict = summary_coords_stat
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
    param_number : int, optional
        Number of parameters used to generate the simulations. Useful for `coord_type='lhc_x'`. Defaults to 20.
    phase_number : int, optional
        Number of phases in the small box simulations. Useful for `coord_type='smallbox'`. Defaults to 1786.
    summary_coords_stat : dict, optional
        Dictionary containing the summary coordinates for each statistic. Defaults to summary_coords_stat from `acm.data.default`.

    Returns
    -------
    dict
        Dictionary containing the summary coordinates for the given statistic and coordinate type.
    """
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


def covariance_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/v1.1/abacus/covariance_sets/small_box'
    return Path(data_dir) / f'{statistic}.npy'

def read_separation(statistic, data):
    if statistic == 'number_density':
        return None
    elif statistic in ['pk', 'dsc_pk']:
        return data['k']
    elif statistic == 'knn':
        return None
    elif statistic == 'wst':
        return data['coeff_idx']
    elif statistic == 'cgf_r10':
        return data['lambda']
    elif 'pdf' in statistic:
        return data['delta']
    elif statistic == 'mst':
        return data['coeff_idx']
    elif statistic == 'wp':
        return data['rp']
    elif statistic == 'minkowski':
        return data['delta']
    elif statistic in ['tpcf', 'dsc_conf', 'voxel_voids']:
        return data['s']
    else:
        raise ValueError(f'Unknown statistic: {statistic}')

def read_lhc(statistics, select_filters={}, slice_filters={}, return_mask=False, return_sep=False):
    lhc_y_all = []
    mask_all = []
    for statistic in statistics:
        data_fn = lhc_fnames(statistic)
        data = np.load(data_fn, allow_pickle=True).item()
        sep = read_separation(statistic, data)
        coords_y = summary_coords_lhc_y(statistic, sep)
        coords_x = summary_coords_lhc_x(statistic, sep)
        lhc_x = data['lhc_x']
        lhc_x_names = data['lhc_x_names']
        lhc_y = data['lhc_y']
        if coords_y and (select_filters or slice_filters):
            lhc_y, mask = filter_lhc(lhc_y, coords_y, select_filters, slice_filters)
            lhc_x, _ = filter_lhc(lhc_x, coords_x, select_filters, slice_filters)
            mask_all.append(mask)
        else:
            mask_all.append(np.full(lhc_y.shape, False))
        lhc_y_all.append(lhc_y)
    lhc_y_all = np.concatenate(lhc_y_all, axis=0)
    toret = (lhc_x, lhc_y_all, lhc_x_names)
    if return_mask:
        toret = (*toret, mask_all)
    if return_sep:
        toret = (sep, *toret)
    return toret


def read_covariance(statistics, volume_factor=64, select_filters={}, slice_filters={}):
    y_all = []
    for statistic in statistics:
        data_fn = covariance_fnames(statistic)
        data = np.load(data_fn, allow_pickle=True).item()
        sep = read_separation(statistic, data)
        coords = summary_coords_smallbox(statistic, sep)
        y = data['cov_y']
        if coords and (select_filters or slice_filters):
            y, mask = filter_smallbox(y, coords, select_filters, slice_filters)
        y_all.append(y)
    y_all = np.concatenate(y_all, axis=1)
    prefactor = 1 / volume_factor
    cov = prefactor * np.cov(y_all, rowvar=False)
    return cov, len(y)

def read_model(statistics):
    from sunbird.emulators import FCN
    model_all = []
    for statistic in statistics:
        if statistic == 'number_density':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/number_density/cosmo+hod/aug10/last.ckpt'
        if statistic == 'wp':
            # checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/wp/cosmo+hod/jul10_trans/last-v30.ckpt'
            # checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/wp/cosmo+hod/optuna_arcsinh/last-v73.ckpt'
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/wp/cosmo+hod/optuna_log/last-v44.ckpt'
        if statistic == 'pk':
            # checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/pk/cosmo+hod/aug8/last.ckpt'
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/pk/cosmo+hod/optuna/last-v31.ckpt'
        elif statistic == 'tpcf':
            # checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/tpcf/cosmo+hod/aug9_asinh/last.ckpt'
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/tpcf/cosmo+hod/optuna_log/last-v54.ckpt' #change log to asinh
        elif statistic == 'dsc_conf':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_conf/cosmo+hod/aug9/last-v1.ckpt'
        elif statistic == 'dsc_pk':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_fourier/cosmo+hod/optuna/last-v25.ckpt'
        elif statistic == 'knn':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/knn/cosmo+hod/optuna/last-v13.ckpt'
        elif statistic == 'wst':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/wst/cosmo+hod/optuna/last-v80.ckpt'
        elif statistic == 'voxel_voids':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/voxel_voids/cosmo+hod/sep16/last.ckpt'
        elif statistic == 'minkowski':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/minkowski/cosmo+hod/best-model-epoch=132-val_loss=0.0319.ckpt'
        elif statistic == 'mst':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/mst/cosmo+hod/optuna/last-v8.ckpt'
        elif statistic == 'cgf_r10':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/cgf_r10/cosmo+hod/nov20/last-v3.ckpt'
        elif statistic == 'pdf_r10':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/{statistic}/cosmo+hod/optuna/last-v13.ckpt'
        elif statistic == 'pdf_r20':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/{statistic}/cosmo+hod/optuna/last-v33.ckpt'
        model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
        print(checkpoint_fn)
        model.eval()
        if statistic == 'minkowski':
            from sunbird.data.transforms_array import WeiLiuInputTransform, WeiLiuOutputTransForm
            model.transform_output = WeiLiuOutputTransForm()
            model.transform_input = WeiLiuInputTransform()
        model_all.append(model)
    return model_all

def read_emulator_error(statistics, select_filters={}, slice_filters={}):
    y_all = []
    for statistic in statistics:
        data_fn = emulator_error_fnames(statistic)
        data = np.load(data_fn, allow_pickle=True).item()
        sep = read_separation(statistic, data)
        coords = summary_coords_emulator_error(statistic, sep)
        y = data['emulator_error']
        if coords and slice_filters:
            y, mask = filter_emulator_error(y, coords, select_filters, slice_filters)
        y_all.append(y)
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

