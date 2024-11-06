from pathlib import Path
import numpy as np
from sunbird.data.data_utils import convert_to_summary

DEFAULT_MODEL_DIR_DICT = {
    'number_density': f'/pscratch/sd/e/epaillas/emc/trained_models/number_density/cosmo+hod/aug10/last.ckpt',
    'wp': f'/pscratch/sd/e/epaillas/emc/trained_models/wp/cosmo+hod/jul10_trans/last-v30.ckpt',
    'pk': f'/pscratch/sd/e/epaillas/emc/trained_models/pk/cosmo+hod/optuna/last-v31.ckpt',
    'tpcf': f'/pscratch/sd/e/epaillas/emc/trained_models/tpcf/cosmo+hod/aug9_asinh/last.ckpt',
    'dsc_conf': f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_conf/cosmo+hod/aug9/last-v1.ckpt',
    'dsc_fourier': f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_fourier/cosmo+hod/sep4/last.ckpt',
    'knn': f'/pscratch/sd/e/epaillas/emc/trained_models/knn/cosmo+hod/optuna/last-v13.ckpt',
    'wst': f'/pscratch/sd/e/epaillas/emc/trained_models/wst/cosmo+hod/optuna/last-v80.ckpt',
    'voxel_voids': f'/pscratch/sd/e/epaillas/emc/trained_models/voxel_voids/cosmo+hod/sep16/last.ckpt',
    'minkowski': f'/pscratch/sd/e/epaillas/emc/trained_models/minkowski/cosmo+hod/sep17/best-model-epoch=217-val_loss=0.0217.ckpt',
}

fourier_stats = ['pk', 'dsc_fourier']
conf_stats = ['tpcf', 'dsc_conf']

labels_stats = {
    'dsc_conf': 'Density-split',
    'dsc_fourier': 'Density-split 'r'$P_\ell$',
    'dsc_conf_cross': 'Density-split (CCF)',
    'tpcf': 'Galaxy 2PCF',
    'tpcf+dsc_conf': 'DSC + Galaxy 2PCF',
    'number_density+tpcf': 'nbar + Galaxy 2PCF',
    'number_density+pk': 'nbar + P(k)',
    'pk': 'P(k)',
}

def summary_coords_diffsky(statistic, sep):
    if statistic == 'tpcf':
        return {
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'dsc_fourier':
        return {
            'statistics': ['quantile_data_power', 'quantile_power'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            'k': sep,
        }

def summary_coords_diffsky(statistic, sep):
    if statistic == 'tpcf':
        return {
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'dsc_fourier':
        return {
            'statistics': ['quantile_data_power', 'quantile_power'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            'k': sep,
        }

def summary_coords_lhc_y(statistic, sep):
    if statistic == 'number_density':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(100)),
        }
    if statistic == 'dsc_conf':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(100)),
            'statistics': ['quantile_data_correlation', 'quantile_correlation'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'dsc_fourier':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(250)),
            'statistics': ['quantile_data_power', 'quantile_power'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            'k': sep,
        }
    if statistic == 'tpcf':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(100)),
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'voxel_voids':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(100)),
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'pk':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(250)),
            'multipoles': [0, 2],
            'k': sep,
        }
    if statistic == 'wp':
        return {
            'r_p': sep,
        }
    if statistic == 'knn':
        return {
        }
    if statistic == 'wst':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(350)),
            'coeff_idx': sep,
        }
    if statistic == 'minkowski':
        return {
            'delta': sep,
        }

def summary_coords_lhc_x(statistic, sep):
    if statistic == 'number_density':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(100)),
            'param_idx': list(range(20)),
        }
    if statistic == 'dsc_conf':
        return {
            'statistics': ['quantile_data_correlation', 'quantile_correlation'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'dsc_fourier':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(250)),
            'param_idx': list(range(20))
        }
    if statistic == 'tpcf':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(100)),
            'param_idx': list(range(20))
        }
    if statistic =='voxel_voids':
        return {
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'pk':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(250)),
            'param_idx': list(range(20))
        }
    if statistic == 'wp':
        return {
            'r_p': sep,
        }
    if statistic == 'knn':
        return {
        }
    if statistic == 'wst':
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(350)),
            'param_idx': list(range(20))
        }
    if statistic == 'minkowski':
        return {
            'delta': sep,
        }

def summary_coords_emulator_error(statistic, sep):
    if statistic == 'number_density':
        return {
        }
    if statistic == 'dsc_conf':
        return {
            'statistics': ['quantile_data_correlation', 'quantile_correlation'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'dsc_fourier':
        return {
            'statistics': ['quantile_data_power', 'quantile_power'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            'k': sep,
        }
    if statistic == 'voxel_voids':
        return {
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'pk':
        return {
            'multipoles': [0, 2],
            'k': sep,
        }
    if statistic == 'tpcf':
        return {
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'wp':
        return {
            'r_p': sep,
        }
    if statistic == 'knn':
        return {
        }
    if statistic == 'wst':
        return {
            # 'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            # 'hod_idx': list(range(350)),
            'coeff_idx': sep,
        }
    if statistic == 'minkowski':
        return {
            'delta': sep,
        }

def summary_coords_smallbox(statistic, sep):
    if statistic == 'number_density':
        return {
            'phase_idx': list(range(1786)),
        }
    if statistic == 'dsc_conf':
        return {
            'statistics': ['quantile_data_correlation', 'quantile_correlation'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'dsc_fourier':
        return {
            'phase_idx': list(range(1786)),
            'statistics': ['quantile_data_power', 'quantile_power'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            'k': sep,
        }
    if statistic == 'voxel_voids':
        return {
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'pk':
        return {
            'phase_idx': list(range(1786)),
            'multipoles': [0, 2],
            'k': sep,
        }
    if statistic == 'tpcf':
        return {
            'phase_idx': list(range(1786)),
            'multipoles': [0, 2],
            's': sep,
        }
    if statistic == 'wp':
        return {
            'r_p': sep,
        }
    if statistic == 'knn':
        return {
        }
    if statistic == 'wst':
        return {
            'phase_idx': list(range(1786)),
            'coeff_idx': sep,
        }
    if statistic == 'minkowski':
        return {
            'delta': sep,
        }

def lhc_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/{statistic}/cosmo+hod/z0.5/yuan23_prior/ph000/seed0/'
    return Path(data_dir) / f'{statistic}_lhc.npy'

def emulator_error_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/emulator_error/{statistic}/'
    return Path(data_dir) / f'{statistic}_emulator_error.npy'

def diffsky_fnames(statistic, redshift=0.5, phase_idx=1, galsample='mass_conc', version=0.3):
    adict = {0.5: '67120', 0.8: '54980'}  # redshift to scale factor string for UNIT
    data_dir = f'/pscratch/sd/e/epaillas/emc/data_vectors/diffsky/{statistic}/z{redshift}'
    return Path(data_dir) / f'{statistic}_galsampled_diffsky_mock_{adict[redshift]}_fixedAmp_{phase_idx:03}_{galsample}_v{version}.npy'

def covariance_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/{statistic}/cosmo+hod/z0.5/yuan23_prior/ph000/seed0/'
    return Path(data_dir) / f'{statistic}_lhc.npy'

def read_separation(statistic, data):
    if statistic == 'number_density':
        return None
    elif statistic in ['pk', 'dsc_fourier']:
        return data['k']
    elif statistic == 'knn':
        return None
    elif statistic == 'wst':
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
        if statistic == 'minkowski':
            lhc_x = data['lhc_test_x']
            lhc_x_names = data['lhc_x_names']
            lhc_y = data['lhc_test_y']
        else:
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

def read_diffsky(statistics, select_filters={}, slice_filters={}, return_mask=False, return_sep=False):
    y_all = []
    mask_all = []
    for statistic in statistics:
        data_fn = diffsky_fnames(statistic)
        data = np.load(data_fn, allow_pickle=True).item()
        sep = read_separation(statistic, data)
        coords = summary_coords_diffsky(statistic, sep)
        y = data['diffsky_y']
        if coords and (select_filters or slice_filters):
            y, mask = filter_diffsky(y, coords, select_filters, slice_filters)
            mask_all.append(mask)
        else:
            mask_all.append(np.full(y.shape, False))
        y_all.append(y)
    y_all = np.concatenate(y_all, axis=0)
    toret = (y_all,)
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

def read_model(statistics, model_dir_dict = None,):
    from sunbird.emulators import FCN
    model_all = []
    for statistic in statistics:
        if model_dir_dict is None:
            checkpoint_fn = DEFAULT_MODEL_DIR_DICT[statistic]
        else:
            checkpoint_fn = model_dir_dict[statistic]

        if statistic == 'number_density':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/number_density/cosmo+hod/aug10/last.ckpt'
        if statistic == 'wp':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/wp/cosmo+hod/jul10_trans/last-v30.ckpt'
        if statistic == 'pk':
            # checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/pk/cosmo+hod/aug8/last.ckpt'
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/pk/cosmo+hod/optuna/last-v31.ckpt'
        elif statistic == 'tpcf':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/tpcf/cosmo+hod/aug9_asinh/last.ckpt'
        elif statistic == 'dsc_conf':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_conf/cosmo+hod/aug9/last-v1.ckpt'
        elif statistic == 'dsc_fourier':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_fourier/cosmo+hod/optuna/last-v25.ckpt'
        elif statistic == 'knn':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/knn/cosmo+hod/optuna/last-v13.ckpt'
        elif statistic == 'wst':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/wst/cosmo+hod/optuna/last-v80.ckpt'
        elif statistic == 'voxel_voids':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/voxel_voids/cosmo+hod/sep16/last.ckpt'
        elif statistic == 'minkowski':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/minkowski/cosmo+hod/sep17/best-model-epoch=217-val_loss=0.0217.ckpt'
        model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
        model.eval()
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
    if select_filters is not None:
        select_filters = {key: value for key, value in select_filters.items() if key in coords}
    if slice_filters is not None:
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
    if select_filters is not None:
        select_filters = {key: value for key, value in select_filters.items() if key in coords}
    if slice_filters is not None:
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

def filter_diffsky(y, coords, select_filters, slice_filters):
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
    return y.values[~mask].reshape(-1), mask

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

def get_chain_fn(statistic, mock_idx, kmin, kmax, smin, smax):
    data_dir = f'/pscratch/sd/e/epaillas/emc/posteriors/sep6/{statistic}/'
    scales_str = ''
    if any([stat in fourier_stats for stat in statistic.split('+')]):
        scales_str += f'_kmin{kmin}_kmax{kmax}'
    if any([stat in conf_stats for stat in statistic.split('+')]):
        scales_str += f'_smin{smin}_smax{smax}'
    return Path(data_dir) / f'chain_idx{mock_idx}{scales_str}.npy'

def read_chain(statistic, mock_idx=0, kmin=0, kmax=1, smin=0, smax=150, return_labels=False):
    from getdist import MCSamples
    chain_fn = get_chain_fn(statistic, mock_idx, kmin, kmax, smin, smax)
    return read_chain_from_fn(chain_fn=chain_fn, return_labels=return_labels,)


def read_chain_from_fn(chain_fn, return_labels=False, return_true_params=False):
    from getdist import MCSamples
    data = np.load(chain_fn, allow_pickle=True).item()
    ranges=[data['param_ranges'][name] for name in data['param_names']]
    chain = MCSamples(
                samples=data['samples'],
                weights=data['weights'],
                names=data['param_names'],
                ranges = ranges,
                labels=[data['param_labels'][name] for name in data['param_names']],
            )
    
    # Construct the return tuple based on the flags
    if return_labels and return_true_params:
        return chain, data['param_labels'], [data['true_params'][name] for name in data['param_names']]
    elif return_labels:
        return chain, data['param_labels']
    elif return_true_params:
        return chain, data['true_params']
    return chain
