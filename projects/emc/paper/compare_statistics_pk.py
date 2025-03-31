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
    'pk+bk',
    # 'pk+tpcf',
    # 'pk+wp',
    # 'pk+voxel_voids',
    'pk+dt_gv',
    # 'pk+minkowski',
    'pk+wst',
    # 'pk+pdf_r10',
    # 'pk+mst'
][::-1]
labels_stats = {
    'pk': 'Power spectrum multipoles',
    'pk+bk': 'Bispectrum multipoles',
    'pk+tpcf': '2PCF multipoles',
    'pk+wp': 'Projected 2PCF',
    'pk+voxel_voids': 'Voxel void-galaxy CCF',
    'pk+dt_gv': 'DT void-galaxy CCF',
    'pk+minkowski': 'Minkowski functionals',
    'pk+wst': 'Scattering Transform',
    'pk+pdf_r10': '$\delta$ PDF ($r=10\,h^{-1}$Mpc)',
    'pk+mst': 'Minimum Spanning Tree',
}
labels_params = {
    'omega_b': r'$\omega_{\rm b}$',
    'omega_cdm': r'$\omega_{\rm cdm}$',
    'sigma8_m': r'$\sigma_8$',
    'n_s': r'$n_s$',
}

fig, ax = plt.subplots(1, len(params), figsize=(10, 5), sharey=True)

for istat, stat in enumerate(stats):
    data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/feb11/c000_hod030/lcdm/'
    data_fn = Path(data_dir) / f"{stat}_chain.npy"
    data = np.load(data_fn, allow_pickle=True).item()
    samples = MCSamples(
                samples=data['samples'],
                weights=data['weights'],
                names=data['names'],
                ranges=data['ranges'],
                labels=[data['labels'][n] for n in data['names']],
            )
    markers = data['markers']
    maxlike = data['samples'][data['log_likelihood'].argmax()]

    for iparam, param in enumerate(params):
        mean = samples.mean(param)
        std = samples.std(param)
        ax[iparam].errorbar(x=mean, y=istat, xerr=std, ms=4.0, elinewidth=1.0,
                      marker='o', color='k', capsize=3)

        # ax[iparam].plot(maxlike[iparam], istat, marker='o', color='hotpink')
        ax[iparam].axvline(markers[param], color='k', linestyle='--', lw=1.0)
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
plt.savefig('compare_statistics_pk.png', bbox_inches='tight', dpi=300)