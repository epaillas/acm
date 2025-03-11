import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from acm.observables.emc import GalaxyBispectrumMultipoles

# latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ells = [0, 2]

for ell in ells:

    observable = GalaxyBispectrumMultipoles(select_filters={'multipoles': [ell]})
    k123 = observable.separation
    weight = k123.prod(axis=0)
    bin_index = list(range(len(weight)))
    lhc_x = observable.lhc_x
    lhc_y = observable.lhc_y
    cov = observable.get_covariance_matrix(divide_factor=64)
    error = np.sqrt(np.diag(cov))
    print(np.shape(weight))

    # reshape the features to have the format (n_samples, n_features)
    n_samples = len(observable.coords_lhc_x['cosmo_idx']) * len(observable.coords_lhc_x['hod_idx'])
    lhc_x = lhc_x.reshape(n_samples, -1)
    lhc_y = lhc_y.reshape(n_samples, -1)
    print(f'Reshaped LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

    pred_y = observable.get_model_prediction(lhc_x, batch=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    weight = 1
    test_idxs = [30]
    for i, test_idx in enumerate(test_idxs):
        # color = plt.cm.viridis(test_idx / 5)
        color = 'k'
        if i == 0:
            ax.errorbar(bin_index, lhc_y[test_idx], error, marker='o', ls='', label='simulation',
                ms=2.0, color=color, mew=1.0, mfc='none', elinewidth=1.0)
            ax.plot(pred_y[test_idx], label='emulator', ls='--', color=color, lw=0.5)
        else:
            ax.errorbar(bin_index, lhc_y[test_idx], error, marker='o', ls='', ms=2.0,
                        color=color, mew=1.0, mfc='none')
            ax.plot(pred_y[test_idx], ls='--', color=color, lw=0.5)
    ax.legend()
    ax.set_xlabel(r'$\textrm{bin index}$')
    ax.set_ylabel(rf'$k_1k_2k_3B_{ell}(k_1, k_2, k_3)\,[h^{-3}{{\rm Mpc}}^3]$')
    # if ell == 0:
    #     ax.set_yscale('log')
    plt.savefig(f'bispectrum_b{ell}_emulator.png', dpi=300, bbox_inches='tight')
    plt.close()



    fig, ax = plt.subplots(figsize=(10, 3))
    emulator_error = observable.get_emulator_error()
    ax.plot(emulator_error / error, marker='', ls='-', ms=3.0)
    ax.set_xlabel(r'$\textrm{bin index}$')
    ax.set_ylabel(r'$(X_{\rm model} - X_{\rm test})/\sigma_{\rm abacus}$', fontsize=15)
    ax.grid()
    plt.tight_layout()
    plt.savefig(f'bispectrum_b{ell}_emulator_error.png', dpi=300, bbox_inches='tight')
    plt.show()