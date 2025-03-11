import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from acm.observables.emc import GalaxyCorrelationFunctionMultipoles

# latex
plt.rc('text', usetex=True)
plt.rc('font', family='Helvetica')

observable = GalaxyCorrelationFunctionMultipoles(select_filters={'multipoles':[0]})
s = observable.separation
lhc_x = observable.lhc_x
lhc_y = observable.lhc_y
cov = observable.get_covariance_matrix(divide_factor=64)
error = np.sqrt(np.diag(cov))

# reshape the features to have the format (n_samples, n_features)
n_samples = len(observable.coords_lhc_x['cosmo_idx']) * len(observable.coords_lhc_x['hod_idx'])
lhc_x = lhc_x.reshape(n_samples, -1)
lhc_y = lhc_y.reshape(n_samples, -1)
print(f'Reshaped LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

pred_y = observable.get_model_prediction(lhc_x, batch=True)

fig, ax = plt.subplots(figsize=(4, 3))

for test_idx in [30]:
    # ax.plot(s, s**2*pred_y[test_idx], label=r'$\textrm{emulator}$', ls='-', color='hotpink', lw=1.0)
    ax.errorbar(s, s**2*lhc_y[test_idx], s**2*error, marker='o', ls='', label=r'$\textrm{data}$',
                ms=3.0, color='dimgrey', mew=0.5, mfc='none', elinewidth=0.5)
ax.set_xlabel(r'$s\,[{h^{-1}{\rm Mpc}}]$', fontsize=13)
ax.set_ylabel(r'$s^2\xi_0(s)\,[h^{-2}{\rm Mpc}^2]$', fontsize=13)
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.legend(fontsize=10, handlelength=1.0)
plt.tight_layout()
plt.savefig(f'xi_emulator.png', dpi=300, bbox_inches='tight')
plt.close()
