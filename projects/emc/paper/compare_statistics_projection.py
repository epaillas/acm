import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

chains = []

legend_labels = []

cosmo_idx = 0
hod_idx = 30
params = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s']
stats = [
    'pk',
    'tpcf',
    'bk',
    # 'wp',
    # 'voxel_voids',
    # 'dt_gv',
    # 'minkowski',
    # 'wst',
    # 'pdf_r10',
    # 'mst'
][::-1]
labels_stats = {
    'pk': 'Power spectrum multipoles',
    'bk': 'Bispectrum multipoles',
    'tpcf': '2PCF multipoles',
    'wp': 'Projected 2PCF',
    'voxel_voids': 'Voxel void-galaxy CCF',
    'dt_gv': 'DT void-galaxy CCF',
    'minkowski': 'Minkowski functionals',
    'wst': 'Scattering Transform',
    'pdf_r10': '$\delta$ PDF ($r=10\,h^{-1}$Mpc)',
    'mst': 'Minimum Spanning Tree',
}
labels_params = {
    'omega_b': r'$\omega_{\rm b}$',
    'omega_cdm': r'$\omega_{\rm cdm}$',
    'sigma8_m': r'$\sigma_8$',
    'n_s': r'$n_s$',
}

fig, ax = plt.subplots(1, len(params), figsize=(10, 3), sharey=True)

for istat, stat in enumerate(stats):
    data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/feb11/c000_hod030/lcdm/projection_effects/'
    data_fn = Path(data_dir) / f"chain_{stat}.npy"
    data = np.load(data_fn, allow_pickle=True).item()
    samples = MCSamples(
                samples=data['samples'],
                weights=data['weights'],
                names=data['names'],
                ranges=data['ranges'],
                labels=[data['labels'][n] for n in data['names']],
            )
    markers = data['markers']
    # maxlike = data['samples'][data['log_likelihood'].argmax()]

    profiles_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/feb11/c000_hod030/lcdm/projection_effects/'
    profiles_fn = Path(data_dir) / f"profiles_{stat}.npy"
    profiles = np.load(profiles_fn, allow_pickle=True).item()
    maxl = profiles['bestfit']
    

    for iparam, param in enumerate(params):
        mean = samples.mean(param)
        std = samples.std(param)
        if istat == 0:
            ax[iparam].axvline(markers[param], color='k', linestyle='--', lw=1.0)

        ax[iparam].errorbar(x=mean, y=istat, xerr=std, ms=4.0, elinewidth=1.0,
                      marker='o', color='k', capsize=3)

        ax[iparam].plot(maxl[param], istat, marker='*', color='hotpink')
#         # plot maximum likelihood
#         ml = maxlike[data['names'].index(param)]
#         # ax[i].plot(kmax, ml, 'o', color='k', ms=4.0)
#         ax[i].axhline(markers[param], color='k', linestyle='--', lw=1.0)

#         ax[i].set_ylabel(data['labels'][param], fontsize=15)

#         # tick size
#         ax[i].tick_params(axis='both', which='major', labelsize=13)

        ax[iparam].set_xlabel(labels_params[param], fontsize=15)

ylabels = [labels_stats[stat] for stat in stats]
yvals = np.arange(len(stats))
ax[0].set_yticks(yvals)
ax[0].set_yticklabels(ylabels)

# ax[-1].set_xlabel(r'$k_{\rm max}\,[h/{\rm Mpc}]$', fontsize=15)
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.0)
# plt.savefig('compare_statistics.pdf', bbox_inches='tight')
plt.savefig('compare_statistics_projection.png', bbox_inches='tight', dpi=300)