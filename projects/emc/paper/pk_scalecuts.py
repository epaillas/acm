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
params = ['omega_cdm', 'sigma8_m', 'n_s']

fig, ax = plt.subplots(len(params), 1, figsize=(5, 5), sharex=True)

kmin = 0.0
for kmax in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:

    data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/c000_hod030/base/n_total10k'
    data_fn = Path(data_dir) / f"pk+number_density_k{kmin:.2f}-{kmax:.2f}_chain.npy"
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

    for i, param in enumerate(params):

        ax[i].errorbar(kmax, samples.mean(param), samples.std(param), ms=4.0, elinewidth=1.0,
                      marker='o', color='k', capsize=3)
        # plot maximum likelihood
        ml = maxlike[data['names'].index(param)]
        # ax[i].plot(kmax, ml, 'o', color='k', ms=4.0)
        ax[i].axhline(markers[param], color='k', linestyle='--', lw=1.0)

        ax[i].set_ylabel(data['labels'][param], fontsize=15)

        # tick size
        ax[i].tick_params(axis='both', which='major', labelsize=13)

ax[-1].set_xlabel(r'$k_{\rm max}\,[h/{\rm Mpc}]$', fontsize=15)
plt.tight_layout()
plt.subplots_adjust(hspace=0.0)
plt.savefig('fig/pk_scalecuts.pdf', bbox_inches='tight')