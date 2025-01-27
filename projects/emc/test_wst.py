import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from acm.observables.emc import WaveletScatteringTransform

# latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

observable = WaveletScatteringTransform()
coeff_idx = observable.separation
lhc_x = observable.lhc_x
lhc_y = observable.lhc_y
cov = observable.get_covariance_matrix(divide_factor=8)
error = np.sqrt(np.diag(cov))

# reshape the features to have the format (n_samples, n_features)
n_samples = len(observable.coords_lhc_x['cosmo_idx']) * len(observable.coords_lhc_x['hod_idx'])
lhc_x = lhc_x.reshape(n_samples, -1)
lhc_y = lhc_y.reshape(n_samples, -1)
print(f'Reshaped LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

pred_y = observable.get_model_prediction(lhc_x, batch=True)

fig, ax = plt.subplots(figsize=(10, 3))

for test_idx in range(5):
    color = plt.cm.viridis(test_idx / 5)
    if test_idx == 0:
        ax.plot(lhc_y[test_idx], marker='o', ls='', label='simulation', ms=2.0, color=color, mew=1.0, mfc='none')
        ax.plot(pred_y[test_idx], label='emulator', ls='--', color=color, lw=0.5)
    else:
        ax.plot(lhc_y[test_idx], marker='o', ls='', ms=2.0, color=color, mew=1.0, mfc='none')
        ax.plot(pred_y[test_idx], ls='--', color=color, lw=0.5)
ax.legend()
ax.set_xlabel(r'$\textrm{bin index}$')
ax.set_ylabel(r'$\textrm{WST coefficient}$')
plt.savefig(f'wst_emulator.png', dpi=300, bbox_inches='tight')
plt.close()



fig, ax = plt.subplots(figsize=(10, 3))
emulator_error = observable.get_emulator_error()
ax.plot(emulator_error / error, marker='', ls='-', ms=3.0)
ax.set_xlabel(r'$\textrm{bin index}$')
ax.set_ylabel(r'$(X_{\rm model} - X_{\rm test})/\sigma_{\rm diffsky}$', fontsize=15)
ax.grid()
plt.tight_layout()
plt.savefig(f'wst_emulator_error.png', dpi=300, bbox_inches='tight')
plt.show()