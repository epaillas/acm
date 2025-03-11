import numpy as np
from pathlib import Path
from acm.observables.emc import GalaxyPowerSpectrumMultipoles
import matplotlib.pyplot as plt

# latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# reference hod
ref = GalaxyPowerSpectrumMultipoles(
    select_filters={
        'cosmo_idx': 0, 'hod_idx': 30,
    },
)

k = ref.separation
pk_ref = ref.lhc_y
cov_ref = ref.get_covariance_matrix(divide_factor=64)
error_ref = np.sqrt(np.diag(cov_ref))

fig, ax = plt.subplots(figsize=(4, 3))
ax.errorbar(k, k * pk_ref[:len(k)], k * error_ref[:len(k)], marker='o',
            ls='', ms=2.0, elinewidth=1.0, label=r'$\textrm{reference}$', color='k')
ax.errorbar(k, k * pk_ref[len(k):], k * error_ref[len(k):], marker='o',
            ls='', ms=2.0, elinewidth=1.0, color='k')

for cosmo_idx in range(1, 5):
    chi2_cosmo = []
    pk_cosmo = []
    for hod_idx in range(100):

        lhc = GalaxyPowerSpectrumMultipoles(
            select_filters={
                'cosmo_idx': cosmo_idx, 'hod_idx': hod_idx,
            },
        )
        pk_lhc = lhc.lhc_y

        # calculate chi2 with respect to the reference
        diff = pk_lhc - pk_ref
        chi2 = diff @ np.linalg.inv(cov_ref) @ diff.T

        chi2_cosmo.append(chi2)
        pk_cosmo.append(pk_lhc)

    # hod that minimizes the chi2
    idx_min = np.argmin(chi2_cosmo)
    print(f'cosmo={cosmo_idx}, hod={idx_min}, chi2={chi2_cosmo[idx_min]}')

    ax.plot(k, k * pk_cosmo[idx_min][:len(k)], label=fr'$\textrm{{c{cosmo_idx:03}, hod{idx_min:03}}}$')
    ax.plot(k, k * pk_cosmo[idx_min][len(k):])


ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$', fontsize=15)
ax.set_ylabel(r'$k\,P_\ell(k)\,[h^{-1}{\rm Mpc}]$', fontsize=15)
ax.legend(fontsize=9, handlelength=1.0, ncol=1)

plt.savefig('pk_best_hods.png', dpi=300, bbox_inches='tight')
plt.close()