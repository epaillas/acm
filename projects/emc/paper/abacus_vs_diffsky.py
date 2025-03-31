import acm.observables.emc as emc
import numpy as np
import matplotlib.pyplot as plt 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(figsize=(4, 3))

cosmo_idx = 0
hod_idx = list(range(50))

observable = emc.GalaxyCorrelationFunctionMultipoles(
        select_filters={
            'cosmo_idx': cosmo_idx, 'hod_idx': hod_idx,
            'multipoles': [0]
        },
        slice_filters={
        }
    )

abacus_y = observable.lhc_y.reshape(len(hod_idx), -1)
diffsky_y = observable.diffsky_y
s = observable.separation
cov = observable.get_covariance_matrix(divide_factor=8)
error = np.sqrt(np.diag(cov))

for hod in hod_idx:
    label = r'$\texttt{AbacusSummit}$' if hod == hod_idx[0] else None
    ax.plot(s, s**2 * abacus_y[hod], color='silver', lw=0.3, label=label)

ax.errorbar(s, s**2 * diffsky_y, s**2 * error, marker='o', ls='', ms=1.5, 
            color='dodgerblue', label=r'$\texttt{Diffsky}$', elinewidth=0.7)

ax.set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=13)
ax.set_ylabel(r'$s^2\xi_0(s)\,[h^{-2}{\rm Mpc}^2]$', fontsize=13)

ax.set_ylim(-80, 180)
ax.legend(fontsize=12, handlelength=0.5, handletextpad=0.5, loc='best', frameon=False)
plt.savefig('abacus_vs_diffsky.png', dpi=300, bbox_inches='tight')