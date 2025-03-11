from pathlib import Path
import numpy as np
from sunbird.data.data_utils import convert_to_summary
import matplotlib.pyplot as plt


def lhc_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/{statistic}/cosmo+hod/z0.5/yuan23_prior/ph000/seed0/'
    return Path(data_dir) / f'{statistic}_lhc.npy'

def covariance_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/{statistic}/cosmo+hod/z0.5/yuan23_prior/ph000/seed0/'
    return Path(data_dir) / f'{statistic}_lhc.npy'

def diffsky_fnames(statistic):
    data_dir = f'/pscratch/sd/e/epaillas/emc/data_vectors/diffsky/{statistic}/z0.5'
    return Path(data_dir) / f'{statistic}_galsampled_diffsky_mock_67120_fixedAmp_001_mass_conc_v0.3.npy'

def read_lhc(statistic='dsc_conf', filters={}):
    data_fn = lhc_fnames(statistic)
    data = np.load(data_fn, allow_pickle=True).item()
    lhc_x = data['lhc_x']
    lhc_x_names = data['lhc_x_names']
    lhc_y = data['lhc_y']
    if filters:
        coords = summary_coords(statistic, data)
        lhc_y = filter_lhc(lhc_y, coords, filters)
    return lhc_x, lhc_y, lhc_x_names

def read_covariance(statistic, filters={}):
    data_fn = covariance_fnames(statistic)
    data = np.load(data_fn, allow_pickle=True).item()
    coords = summary_coords(statistic, data)
    y = data['cov_y']
    if filters:
        y = filter_lhc(y, coords, filters)
    prefactor = 1 / 8
    cov = prefactor * np.cov(y, rowvar=False)
    return cov, len(y)

def read_diffsky(statistic, filters={}):
    from pycorr import TwoPointCorrelationFunction
    data_fn = diffsky_fnames(statistic)
    data = TwoPointCorrelationFunction.load(data_fn)[::4]
    s, y = data(ells=(0, 2), return_sep=True)
    if filters:
        coords = summary_coords(statistic, {'s': s})
        y = filter_diffsky(y, coords, filters)
    return s, y

def summary_coords(statistic, data):
    if statistic == 'dsc_conf':
        return {
            'statistics': ['quantile_data_correlation', 'quantile_correlation'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
            's': data['s'],
        }
    elif statistic == 'tpcf':
        return {
            'multipoles': [0, 2],
            's': data['s'],
        }

def filter_lhc(lhc_y, coords, filters):
    dimensions = list(coords.keys())
    dimensions.insert(0, 'mock_idx')
    coords['mock_idx'] = np.arange(lhc_y.shape[0])
    lhc_y = lhc_y.reshape([len(coords[d]) for d in dimensions])
    lhc_y = convert_to_summary(data=lhc_y, dimensions=dimensions, coords=coords)
    fil = [getattr(getattr(lhc_y, key), 'isin')(value) for key, value in filters.items()]
    for i, cond in enumerate(fil):
        mask = mask & cond if i > 0 else fil[0]
    mask = lhc_y.where(mask).to_masked_array().mask
    return lhc_y.values[~mask].reshape(lhc_y.shape[0], -1)

def filter_diffsky(diffsky_y, coords, filters):
    dimensions = list(coords.keys())
    diffsky_y = diffsky_y.reshape([len(coords[d]) for d in dimensions])
    diffsky_y = convert_to_summary(data=diffsky_y, dimensions=dimensions, coords=coords)
    fil = [getattr(getattr(diffsky_y, key), 'isin')(value) for key, value in filters.items()]
    for i, cond in enumerate(fil):
        mask = mask & cond if i > 0 else fil[0]
    mask = diffsky_y.where(mask).to_masked_array().mask
    return diffsky_y.values[~mask].reshape(-1)


statistic = 'tpcf'
filters = {'multipoles': [0, 2]}

covariance_matrix, n_sim = read_covariance(statistic=statistic, filters=filters)
print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

# load the data
lhc_x, lhc_y, lhc_x_names = read_lhc(statistic=statistic, filters=filters)
print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

s, diffsky_y = read_diffsky(statistic=statistic, filters=filters)
print(f'Loaded diffsky with shape: {diffsky_y.shape}')

# loop over lhc and find the mock that minimizes the chi2 with the diffsky
precision_matrix = np.linalg.inv(covariance_matrix)
mock_idx = list(range(400, 500))
chi2 = np.zeros_like(mock_idx)
for i, idx in enumerate(mock_idx):
    chi2[i] = (diffsky_y - lhc_y[idx]) @ precision_matrix @ (diffsky_y - lhc_y[idx]).T

i_min = np.argmin(chi2)
idx_min = mock_idx[i_min]
print(f'Index of minimum chi2: {idx_min}')
print(f'chi2 at minimum: {chi2[i_min]}')

fig, ax = plt.subplots()
ax.errorbar(s,
            s**2*diffsky_y[:len(s)],
            s**2*np.sqrt(np.diag(covariance_matrix))[:len(s)],
            marker='o', ls='', ms=2.0, label='diffsky')
ax.plot(s, s**2*lhc_y[idx_min][:len(s)], ls='-', label='best mock')
ax.legend()
plt.show()