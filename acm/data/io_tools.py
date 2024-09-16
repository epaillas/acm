from pathlib import Path
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from sunbird.data.data_utils import convert_to_summary
import torch
import glob


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

def summary_coords(statistic, sep):
    if statistic == {'number_density'}:
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
    elif statistic in ['tpcf', 'voxel_voids']:
        return {
            'multipoles': [0, 2],
            's': sep,
        }
    elif statistic == 'pk':
        return {
            'multipoles': [0, 2],
            'k': sep,
        }
    elif statistic == 'wp':
        return {
            'r_p': sep,
        }
    elif statistic == 'knn':
        return {
            'k': list(range(1, 10)),
            'rp': sep[0],
            'pi': sep[1],
        }
    elif statistic == 'wst':
        return {
            'coeff_idx': sep,
        }

def lhc_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/{statistic}/cosmo+hod/z0.5/yuan23_prior/ph000/seed0/'
    return Path(data_dir) / f'{statistic}_lhc.npy'

def emulator_error_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/emulator_error/{statistic}/'
    return Path(data_dir) / f'{statistic}_emulator_error.npy'

def covariance_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/{statistic}/cosmo+hod/z0.5/yuan23_prior/ph000/seed0/'
    return Path(data_dir) / f'{statistic}_lhc.npy'

def read_separation(statistic, data):
    if statistic == 'number_density':
        return None
    elif statistic in ['pk', 'dsc_fourier']:
        return data['k']
    elif statistic == 'knn':
        return data['rp'], data['pi']
    elif statistic == 'wst':
        return data['coeff_idx']
    elif statistic == 'wp':
        return data['rp']
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
        coords = summary_coords(statistic, sep)
        lhc_x = data['lhc_x']
        lhc_x_names = data['lhc_x_names']
        lhc_y = data['lhc_y']
        if coords and (select_filters or slice_filters):
            lhc_y, mask = filter_lhc(lhc_y, coords, select_filters, slice_filters)
            mask_all.append(mask)
        else:
            mask_all.append(np.full(lhc_y.shape[1], False))
        lhc_y_all.append(lhc_y)
    lhc_y_all = np.concatenate(lhc_y_all, axis=1)
    toret = (lhc_x, lhc_y_all, lhc_x_names)
    if return_mask:
        toret = (*toret, mask_all)
    if return_sep:
        toret = (sep, *toret)
    return toret

def read_covariance(statistics, select_filters={}, slice_filters={}):
    y_all = []
    for statistic in statistics:
        data_fn = covariance_fnames(statistic)
        data = np.load(data_fn, allow_pickle=True).item()
        sep = read_separation(statistic, data)
        coords = summary_coords(statistic, sep)
        y = data['cov_y']
        if coords and (select_filters or slice_filters):
            y, mask = filter_lhc(y, coords, select_filters, slice_filters)
        y_all.append(y)
    y_all = np.concatenate(y_all, axis=1)
    prefactor = 1 / 64
    cov = prefactor * np.cov(y_all, rowvar=False)
    return cov, len(y)

def read_model(statistics):
    from sunbird.emulators import FCN
    model_all = []
    for statistic in statistics:
        if statistic == 'number_density':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/number_density/cosmo+hod/aug10/last.ckpt'
        if statistic == 'wp':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/wp/cosmo+hod/jul10_trans/last-v30.ckpt'
        if statistic == 'pk':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/pk/cosmo+hod/aug8/last.ckpt'
        elif statistic == 'tpcf':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/tpcf/cosmo+hod/aug9_asinh/last.ckpt'
        elif statistic == 'dsc_conf':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_conf/cosmo+hod/aug9/last-v1.ckpt'
        elif statistic == 'dsc_fourier':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/dsc_fourier/cosmo+hod/sep4/last.ckpt'
        elif statistic == 'knn':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/knn/cosmo+hod/sep12/last.ckpt'
        elif statistic == 'wst':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/wst/cosmo+hod/sep16/last.ckpt'
        elif statistic == 'minkowski':
            checkpoint_fn = f'/pscratch/sd/e/epaillas/emc/trained_models/minkowski/Minkowski-best-model-epoch=276-val_loss=0.02366.ckpt'
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
        coords = summary_coords(statistic, sep)
        y = data['emulator_error']
        if coords and (select_filters or slice_filters):
            coords = summary_coords(statistic, sep)
            y, mask = filter_emulator_error(y, coords, select_filters, slice_filters)
        y_all.append(y)
    y_all = np.concatenate(y_all, axis=0)
    return y_all

def filter_lhc(lhc_y, coords, select_filters, slice_filters):
    select_filters = {key: value for key, value in select_filters.items() if key in coords}
    slice_filters = {key: value for key, value in slice_filters.items() if key in coords}
    dimensions = list(coords.keys())
    dimensions.insert(0, 'mock_idx')
    coords['mock_idx'] = np.arange(lhc_y.shape[0])
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
    return y.values[~mask].reshape(-1), mask

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

