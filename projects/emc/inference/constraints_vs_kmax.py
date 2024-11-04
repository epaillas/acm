import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from io_tools import *
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def kmax_plot(statistics, mock_idx=30):
    fig, ax = plt.subplots(len(params), 1)
    chains = []
    for istat, statistic in enumerate(statistics):
        constraints = {}
        for param in params:
            constraints[param] = []
        for kmax in kmaxs:
            chain, labels = read_chain(statistic, mock_idx=mock_idx,
                                       kmin=kmin, kmax=kmax, return_labels=True)
            chains.append(chain)
            for param in params:
                constraints[param].append([chain.mean(param), chain.std(param)])
        for param in params:
            constraints[param] = np.asarray(constraints[param])
        for i, param in enumerate(params):
            ax[i].fill_between(kmaxs, constraints[param][:, 0] - constraints[param][:, 1],
                               constraints[param][:, 0] + constraints[param][:, 1], alpha=0.1)
            ax[i].errorbar(kmaxs + 0.02*istat, constraints[param][:, 0], constraints[param][:, 1],
                           marker='o', ls='', capsize=3, ms=4.0, label=labels_stats[statistic])
            ax[i].plot(kmaxs, [truth[param]]*len(kmaxs), ls='--', color='grey')

            ax[i].set_ylabel(labels[param], fontsize=13)
    for aa in ax[:-1]:
        aa.axes.get_xaxis().set_visible(False)
    ax[-1].set_xlabel(r'$k_{\rm max}\,[h{\rm Mpc}^{-1}]$', fontsize=13)
    ax[0].legend(bbox_to_anchor=(0.2, 1.05), ncols=2)
    plt.subplots_adjust(hspace=0.1)
    plt.savefig('test.pdf')
    plt.show()


if __name__ == '__main__':
    stats = ['pk', 'dsc_fourier']

    covariance_matrix, correlation_matrix = read_covariance(statistics=stats)
    s, lhc_x, lhc_y, lhc_x_names = read_lhc(statistics=stats, return_sep=True)
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
    print(f'Loaded covariance with shape: {covariance_matrix.shape}')


    mock_idx = 30
    truth = dict(zip(lhc_x_names, lhc_x[mock_idx]))

    smax = 152
    smin = 0

    kmin = 0.0
    kmaxs = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    params = ['omega_cdm', 'sigma8_m', 'n_s']

    kmax_plot(stats, mock_idx)

