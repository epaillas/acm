import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

labels_stats = {
    'wp': r'$\textrm{Galaxy } w_p$',
    'dsc_conf': 'Density-split',
    'dsc_conf_cross': 'Density-split (CCF)',
    'tpcf': 'Galaxy 2PCF',
    'number_density+tpcf': 'nbar + Galaxy 2PCF',
    'tpcf+dsc_conf': 'Galaxy 2PCF + DSC',
}


chains = []

legend_labels = []

cosmo_idx = 0
hod_idx = 30
params = ['omega_cdm', 'sigma8_m', 'n_s']

fig, ax = plt.subplots(len(params), 1, figsize=(5, 5), sharex=True)

smax = 200.0
for smin in [0, 5, 10, 20, 40, 60, 80, 100]:

    data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/c000_hod030/base'
    data_fn = Path(data_dir) / f"tpcf+number_density_s{smin:.2f}-{smax:.2f}_chain.npy"
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

        ax[i].errorbar(smin, samples.mean(param), samples.std(param), ms=4.0, elinewidth=1.0,
                      marker=None, color='k', capsize=3)
        ml = maxlike[data['names'].index(param)]
        ax[i].plot(smin, ml, 'o', color='k', ms=4.0)
        ax[i].axhline(markers[param], color='k', linestyle='--', lw=1.0)

        ax[i].set_ylabel(data['labels'][param], fontsize=15)

        # tick size
        ax[i].tick_params(axis='both', which='major', labelsize=13)

ax[-1].set_xlabel(r'$s_{\rm min}$ [Mpc/h]', fontsize=15)

plt.savefig('fig/tpcf_scalcuts.pdf', bbox_inches='tight')