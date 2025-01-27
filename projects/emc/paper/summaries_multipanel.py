import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from acm.data.io_tools import *
import acm.observables.emc as emc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_data(statistic, return_model=True):
    stat = getattr(emc, statistic)
    observable = stat(select_filters=select_filters, slice_filters=slice_filters)
    covariance_matrix = observable.get_covariance_matrix(divide_factor=64)
    data_x = observable.lhc_x
    data_y = observable.lhc_y
    error = np.sqrt(np.diag(covariance_matrix))
    sep = observable.separation
    if return_model:
        model = observable.get_model_prediction(data_x)
        return sep, data_y, error, model
    return sep, data_y, error


fig, ax = plt.subplots(3, 4, figsize=(14, 10))

# projected correlation function
statistic = 'GalaxyProjectedCorrelationFunction'
select_filters = {'cosmo_idx': 0, 'hod_idx': 30}
slice_filters = {}
sep, data_y, error, model = get_data(statistic)

ax[0, 0].errorbar(sep, data_y, error, markersize=3.0, elinewidth=1.0,
                  marker='o', ls='', color='dimgrey')
ax[0, 0].plot(sep, model, color='r')
ax[0, 0].set_xscale('log')
ax[0, 0].set_yscale('log')
ax[0][0].set_xlabel(r'$r\,[h^{-1}{\rm Mpc}]$', fontsize=15)
ax[0][0].set_ylabel(r'$w_p(r)$', fontsize=15)
ax[0][0].set_title(r'$\textrm{Projected 2PCF}$', fontsize=15)

# 2PCF multipoles
statistic = 'GalaxyCorrelationFunctionMultipoles'
for ell in [0, 2]:
    select_filters = {'multipoles': [ell], 'cosmo_idx': 0, 'hod_idx': 30}
    slice_filters = {'s': (0, 150)}
    sep, data_y, error, model = get_data(statistic)
    sep = sep[(sep >= 0) & (sep <= 150)]
    ax[0, 1].errorbar(sep, sep**2*data_y, sep**2*error, markersize=3.0, elinewidth=1.0,
                    marker='o', ls='', color='dimgrey')
    ax[0, 1].plot(sep, sep**2*model, color='crimson')
ax[0][1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=15)
ax[0][1].set_ylabel(r'$s^2\xi_\ell(s)\,[h^{-2}{\rm Mpc}^2]$', fontsize=15)
ax[0][1].set_title(r'$\textrm{2PCF multipoles}$', fontsize=15)

# power spectrum
statistic = 'GalaxyPowerSpectrumMultipoles'
for ell in [0, 2]:
    select_filters = {'multipoles': [ell], 'cosmo_idx': 0, 'hod_idx': 30}
    slice_filters = {'k': (0.0, 0.5)}
    sep, data_y, error, model = get_data(statistic)

    ax[0, 2].errorbar(sep, sep*data_y, sep*error, markersize=3.0, elinewidth=1.0,
                    marker='o', ls='', color='dimgrey')
    ax[0, 2].plot(sep, sep*model, color='r')
ax[0][2].set_xlabel(r'$k\,[h/{\rm Mpc}]$', fontsize=15)
ax[0][2].set_ylabel(r'$k\,P(k)\,[h^{-2}{\rm Mpc}^2]$', fontsize=15)
ax[0][2].set_title(r'$\textrm{Power spectrum multipoles}$', fontsize=15)

# density-split statistics
statistic = 'DensitySplitPowerSpectrumMultipoles'
for q in [0, 1, 3, 4]:
    select_filters = {'multipoles': [0], 'cosmo_idx': 0, 'hod_idx': 30,
                      'quantiles': [q], 'statistics': ['quantile_data_power']}
    slice_filters = {'k': (0.0, 0.5)}
    sep, data_y, error, model = get_data(statistic)

    ax[1, 0].errorbar(sep, sep**2*data_y, sep**2*error, markersize=3.0, elinewidth=1.0,
                    marker='o', ls='', color='dimgrey')
    ax[1, 0].plot(sep, sep**2*model, color='r')
ax[1][0].set_xlabel(r'$k\,[h/{\rm Mpc}]$', fontsize=15)
ax[1][0].set_ylabel(r'$k\,P(k)\,[h^{-2}{\rm Mpc}^2]$', fontsize=15)
ax[1][0].set_title(r'$\textrm{Density-split multipoles}$', fontsize=15)

# Minkowski functionals
statistic = 'MinkowskiFunctionals'
select_filters = {'cosmo_idx': 0, 'hod_idx': 30}
slice_filters = {}
sep, data_y, error, model = get_data(statistic)
ax[2, 0].errorbar(sep[::3], data_y[::3], error[::3], markersize=3.0, elinewidth=1.0,
                marker='o', ls='', color='dimgrey')
ax[2, 0].plot(sep[::3], model[::3], color='r')
ax[2][0].set_xlabel(r'$\textrm{Overdensity } \Delta$', fontsize=15)
ax[2][0].set_ylabel(r'$W_i$', fontsize=15)
ax[2][0].set_title(r'$\textrm{Minkowski functionals}$', fontsize=15)

# Wavelet scattering transform
statistic = 'WaveletScatteringTransform'
select_filters = {'cosmo_idx': 0, 'hod_idx': 30}
slice_filters = {}
sep, data_y, error, model = get_data(statistic)
ax[2, 1].errorbar(sep[::2], data_y[::2], error[::2], markersize=3.0, elinewidth=1.0,
                marker='o', ls='', color='dimgrey')
ax[2, 1].plot(sep[::2], model[::2], color='r')
ax[2][1].set_xlabel(r'$\textrm{Coefficient index}$', fontsize=15)
ax[2][1].set_ylabel(r'$\textrm{WST coefficient}$', fontsize=15)
ax[2][1].set_title(r'$\textrm{Wavelet scattering transform}$', fontsize=15)

# Density PDF
statistic = f'GalaxyOverdensityPDF'
select_filters = {'cosmo_idx': 0, 'hod_idx': 30}
slice_filters = {}
sep, data_y, error, model = get_data(statistic)
ax[1, 2].errorbar(sep, data_y, error, markersize=3.0, elinewidth=1.0,
                marker='o', ls='', color='dimgrey')
ax[1, 2].plot(sep, model, color='r')
ax[1][2].set_xlim(-1.2, 2)
ax[1][2].set_xlabel(r'$\textrm{Overdensity } \Delta$', fontsize=15)
ax[1][2].set_ylabel(r'$\textrm{PDF}$', fontsize=15)
ax[1][2].set_title(r'$\textrm{Overdensity PDF}$', fontsize=15)

# Cumulant Generating Function
statistic = f'CumulantGeneratingFunction'
select_filters = {'cosmo_idx': 0, 'hod_idx': 30}
slice_filters = {}
sep, data_y, error = get_data(statistic, return_model=False)
ax[1, 3].errorbar(sep[::2], data_y[::2], error[::2], markersize=3.0, elinewidth=1.0,
                marker='o', ls='', color='dimgrey')
# ax[1, 3].plot(sep, model, color='r')
# ax[1][3].set_xlim(-1.2, 2)
ax[1][3].set_xlabel(r'$\lambda$', fontsize=15)
ax[1][3].set_ylabel(r'$\log\langle e^{\lambda \delta}\rangle$', fontsize=15)
ax[1][3].set_title(r'$\textrm{Cumulant Generating Function}$', fontsize=15)

# Minimum Spanning Tree
statistic = f'MinimumSpanningTree'
select_filters = {'cosmo_idx': 0, 'hod_idx': 30}
slice_filters = {}
sep, data_y, error, model = get_data(statistic)
ax[1, 1].errorbar(sep[::3], data_y[::3], error[::3], markersize=3.0, elinewidth=1.0,
                marker='o', ls='', color='dimgrey')
ax[1, 1].plot(sep, model, color='r')
# ax[1][2].set_xlim(-1.2, 2)
ax[1][1].set_xlabel(r'$\textrm{Coefficient index}$', fontsize=15)
# ax[1][2].set_ylabel(r'$\textrm{PDF}$', fontsize=15)
ax[1][1].set_title(r'$\textrm{MST coefficient}$', fontsize=15)

# Voxel Void-galaxy 2PCF
statistic = 'VoxelVoidGalaxyCorrelationFunctionMultipoles'
for ell in [0, 2]:
    select_filters = {'multipoles': [ell], 'cosmo_idx': 0, 'hod_idx': 30}
    slice_filters = {}
    sep, data_y, error, model = get_data(statistic)
    ax[2, 2].errorbar(sep[::2], data_y[::2], error[::2], markersize=3.0, elinewidth=1.0,
                    marker='o', ls='', color='dimgrey')
    ax[2, 2].plot(sep[::2], model[::2], color='r')
ax[2][2].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=15)
ax[2][2].set_ylabel(r'$\xi_\ell(s)$', fontsize=15)
ax[2][2].set_title(r'$\textrm{Voxel Void-galaxy CF}$', fontsize=15)

# DT Void-galaxy 2PCF
statistic = 'DTVoidGalaxyCorrelationFunctionMultipoles'
for ell in [0, 2]:
    select_filters = {'multipoles': [ell], 'cosmo_idx': 0, 'hod_idx': 30}
    slice_filters = {}
    sep, data_y, error, model = get_data(statistic)
    ax[2, 3].errorbar(sep[::2], data_y[::2], error[::2], markersize=3.0, elinewidth=1.0,
                    marker='o', ls='', color='dimgrey')
    ax[2, 3].plot(sep[::2], model[::2], color='r')
ax[2][3].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=15)
ax[2][3].set_ylabel(r'$\xi_\ell(s)$', fontsize=15)
ax[2][3].set_title(r'$\textrm{DT Void-galaxy CF}$', fontsize=15)

# Density-field cumulants
# statistic = 'cgf_r10'
# select_filters = {'cosmo_idx': 0, 'hod_idx': 30}
# slice_filters = {}
# sep, data_y, error = get_data(statistic, return_model=False)
# ax[2, 2].errorbar(sep, data_y, error, markersize=3.0, elinewidth=1.0,
#                 marker='o', ls='', color='dimgrey')
# ax[2, 2].plot(sep[::2], model[::2], color='r')
# ax[2][2].set_xlabel(r'$\textrm{Overdensity } \Delta$', fontsize=15)
# ax[2][2].set_ylabel(r'$\textrm{PDF}$', fontsize=15)
# ax[2][2].set_title(r'$\textrm{Cumulant generating function}$', fontsize=15)


for ax in fig.axes:
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)

plt.tight_layout()
plt.savefig('fig/summaries_multipanel.pdf', bbox_inches='tight')