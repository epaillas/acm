from sunbird.inference.pocomc import PocoMCSampler
from sunbird.inference.priors import Yuan23, AbacusSummit
from sunbird import setup_logging

import acm.observables.emc as emc

from pathlib import Path
import numpy as np
import argparse

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_priors(cosmo=True, hod=True):
    stats_module = 'scipy.stats'
    priors, ranges, labels = {}, {}, {}
    if cosmo:
        priors.update(AbacusSummit(stats_module).priors)
        ranges.update(AbacusSummit(stats_module).ranges)
        labels.update(AbacusSummit(stats_module).labels)
    if hod:
        priors.update(Yuan23(stats_module).priors)
        ranges.update(Yuan23(stats_module).ranges)
        labels.update(Yuan23(stats_module).labels)
    return priors, ranges, labels


parser = argparse.ArgumentParser()
parser.add_argument("--cosmo_idx", type=int, default=0)
parser.add_argument("--hod_idx", type=int, default=30)

args = parser.parse_args()
setup_logging()

# set up the inference
priors, ranges, labels = get_priors(cosmo=True, hod=True)
# fixed_params = []
fixed_params = ['w0_fld', 'wa_fld', 'nrun', 'N_ur']
# , 'sigma', 'kappa', 'alpha', 's', 'A_cen', 'A_sat', 'B_cen', 'B_sat', 'alpha_s', 'alpha_c']
add_emulator_error = True

# load observables with their custom filters
observable = emc.GalaxyCorrelationFunctionMultipoles(
    select_filters={
        'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx,
        'multipoles': [0, 2]
    },
    slice_filters={
    }
)

statistics = observable.stat_name
print(f'Fitting {statistics} with cosmo_idx={args.cosmo_idx} and hod_idx={args.hod_idx}')

# load the data
data_x = observable.lhc_x
data_x_names = observable.lhc_x_names
data_y = observable.lhc_y
print(f'Loaded LHC x with shape: {data_x.shape}')
print(f'Loaded LHC y with shape {data_y.shape}')

# load the covariance matrix
covariance_matrix = observable.get_covariance_matrix(divide_factor=64)
error = np.sqrt(np.diag(covariance_matrix))
print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

# load the model
models = observable.model
model_coordinates = observable.coords_model

s = observable.separation



for param_name in ['n_s', 'B_cen', 's', 'omega_cdm', 'sigma8_m', 'w0_fld', 'logM_cut']:

    fig, ax = plt.subplots(2, 1, figsize=(4, 5), sharex=True)

    prior_sample = np.linspace(ranges[param_name][0], ranges[param_name][1], 100)
    prior_norm = prior_sample - prior_sample.min()
    prior_norm /= prior_norm.max()

    cmap = matplotlib.cm.get_cmap('RdBu')

    for i, param_value in enumerate(prior_sample):
        param_idx = data_x_names.index(param_name)
        data_x[param_idx] = param_value
        pred_y = observable.get_model_prediction(data_x)

        ax[0].plot(s, s**2 * pred_y[:len(s)], color=cmap(prior_norm[i]))
        ax[1].plot(s, s**2 * pred_y[len(s):], color=cmap(prior_norm[i]))

    divider = make_axes_locatable(fig.axes[0])
    cax = divider.append_axes('top', size="7%", pad=0.15)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=prior_sample.min(), vmax=prior_sample.max()))
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label(labels[param_name], rotation=0, labelpad=10, fontsize=20)

    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    for aa in fig.axes:
        aa.tick_params(axis='both', which='major', labelsize=11)
        aa.tick_params(axis='both', which='minor', labelsize=11)

    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].set_xlim(s.min(), s.max())
    ax[0].set_ylabel(rf'$s^2\xi_{0}(s)\,[h^{{-2}}{{\rm Mpc}}^2]$', fontsize=13)
    ax[1].set_ylabel(rf'$s^2\xi_{2}(s)\,[h^{{-2}}{{\rm Mpc}}^2]$', fontsize=13)
    ax[1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    plt.savefig(f'model_sensitivity_{param_name}.png', dpi=300, bbox_inches='tight')
