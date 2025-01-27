from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from sunbird.data.transforms_array import WeiLiuOutputTransForm, WeiLiuInputTransform
from acm.data.io_tools import *
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_emulator_error(lhc_test_y, pred_test_y):
    return np.median(np.abs(lhc_test_y - pred_test_y), axis=0)

def get_emulator_covariance(lhc_test, pred_test):
    diff = lhc_test - pred_test
    return np.cov(diff.T)

def get_chi2(diff, covariance_matrix):
    inv_cov = np.linalg.inv(covariance_matrix)
    # print(np.diag(inv_cov))
    chi2 = []
    for i in range(len(diff)):
        chi2.append(diff[i] @ inv_cov @ diff[i])
    return chi2

def get_loglikelihood(diff, covariance_matrix):
    inv_cov = np.linalg.inv(covariance_matrix)
    print(np.linalg.det(covariance_matrix))
    print(np.linalg.inv(covariance_matrix))
    norm = len(diff) * np.log(2 * np.pi) + np.log(np.linalg.det(covariance_matrix))
    loglikelihood = []
    for i in range(len(diff)):
        like = -0.5 * (norm + diff[i] @ inv_cov @ diff[i])
        loglikelihood.append(like)
    return loglikelihood

def plot_model(mock_idx=30):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(lhc_test_y[mock_idx], marker='o', ls='', ms=3.0, label='data')
    ax.plot(pred_test_y[mock_idx], label='emulator')
    ax.set_xlabel('bin number', fontsize=15)
    ax.set_ylabel(r'$X$', fontsize=15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'figures/{statistic}_model_idx{mock_idx}.pdf')
    plt.close()

def plot_emulator_error():
    fig, ax = plt.subplots(figsize=(4, 3))
    emulator_error = get_emulator_error(pred_test_y, lhc_test_y)
    if statistic == 'number_density':
        ax.plot(emulator_error/np.sqrt(covariance_matrix), marker='o', ms=3.0)
    else:
        ax.plot(emulator_error/np.sqrt(np.diag(covariance_matrix)))
    ax.set_xlabel('bin number', fontsize=15)
    ax.set_ylabel(r'$(X_{\rm model} - X_{\rm test})/\sigma_{\rm abacus}$', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'figures/{statistic}_emulator_error_dataunits.pdf')
    plt.close()

    # fig, ax = plt.subplots(figsize=(4, 3))
    # emulator_error = get_emulator_error(pred_test_y, lhc_test_y)
    # ax.plot(emulator_error, label='emulator error')
    # ax.plot(np.sqrt(np.diag(covariance_matrix)), label='abacus error')
    # ax.set_xlabel('bin number', fontsize=15)
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(f'figures/{statistic}_emulator_data_error.pdf')
    # plt.close()

def plot_chi2_histogram():
    fig, ax = plt.subplots(figsize=(4, 3))
    chi2 = get_chi2(pred_test_y - lhc_test_y, covariance_matrix)
    ax.hist(chi2, bins=30)
    ax.set_xlabel(r'$\chi^2$', fontsize=15)
    ax.set_ylabel('count', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'figures/{statistic}_chi2_histogram.pdf')
    plt.show()

def plot_chi2_scatter():
    fig, ax = plt.subplots(figsize=(4, 3))
    chi2_pred = get_chi2(pred_test_y, covariance_matrix)
    chi2_test = get_chi2(lhc_test_y, covariance_matrix)
    ax.scatter(chi2_pred, chi2_test)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.plot([xmin, xmax], [ymin, ymax], color='black', ls='--')
    ax.set_xlabel(r'$\chi^2_{\rm pred}$', fontsize=15)
    ax.set_ylabel(r'$\chi^2_{\rm test}$', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'figures/{statistic}_chi2_scatter.pdf')
    plt.close()

def plot_loglike_scatter():
    fig, ax = plt.subplots(figsize=(4, 3))
    loglike_pred = get_loglikelihood(pred_test_y, covariance_matrix)
    loglike_test = get_loglikelihood(lhc_test_y, covariance_matrix)
    ax.scatter(loglike_pred, loglike_test)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.plot([xmin, xmax], [ymin, ymax], color='black', ls='--')
    ax.set_xlabel(r'$-\log \mathcal{L}_{\rm pred}$', fontsize=15)
    ax.set_ylabel(r'$-\log \mathcal{L}_{\rm test}$', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'figures/{statistic}_loglike_scatter.pdf')
    plt.close()


def save_emulator_error():
    emulator_error = get_emulator_error(pred_test_y, lhc_test_y)
    emulator_cov = get_emulator_covariance(lhc_test_y, pred_test_y)
    save_dir = f'/pscratch/sd/e/epaillas/emc/v1.1/emulator_error/'
    save_fn = Path(save_dir) / f'{statistic}.npy'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if statistic == 'number_density':
        np.save(save_fn, {'emulator_error': emulator_error})
    elif statistic in ['pk', 'dsc_fourier']:
        np.save(save_fn, {'k': sep, 'emulator_error': emulator_error})
    elif statistic == 'wp':
        np.save(save_fn, {'rp': sep, 'emulator_error': emulator_error})
    elif statistic == 'wst':
        np.save(save_fn, {'coeff_idx': sep, 'emulator_error': emulator_error})
    elif statistic == 'mst':
        np.save(save_fn, {'coeff_idx': sep, 'emulator_error': emulator_error})
    elif 'pdf' in statistic:
        np.save(save_fn, {'delta': sep, 'emulator_error': emulator_error})
    else:
        np.save(save_fn, {'s': sep, 'emulator_error': emulator_error})


if __name__ == '__main__':
    # statistics = ['wp', 'tpcf', 'voxel_voids', 'knn', 'wst', 'dsc_fourier', 'pk', 'pdf_r10', 'pdf_r20']
    statistics = ['tpcf']
    for statistic in statistics:
        print(f'Loading {statistic}')
        select_filters = {}
        slice_filters = {}

        covariance_matrix, n_sim = read_covariance(statistics=[statistic],
                                                select_filters=select_filters,
                                                slice_filters=slice_filters)
        sep, lhc_x, lhc_y, lhc_x_names = read_lhc(statistics=[statistic],
                                                select_filters=select_filters,
                                                slice_filters=slice_filters,
                                                return_sep=True)
        model = read_model(statistics=[statistic])
        print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
        print(f'Loaded covariance with shape: {covariance_matrix.shape}')

        nhod = int(len(lhc_y) / 85)
        if statistic == 'knn':
            ntest = nhod * 3
        else:
            ntest = nhod * 6

        lhc_train_x = lhc_x[ntest:]
        lhc_train_y = lhc_y[ntest:]
        lhc_test_x = lhc_x[:ntest]
        lhc_test_y = lhc_y[:ntest]

        print(f'Calculating emulator error over {len(lhc_test_y)} test samples.')

        with torch.no_grad():
            pred_test_y = model[0].get_prediction(torch.Tensor(lhc_test_x))
            pred_test_y = pred_test_y.numpy()
        
        plot_model(mock_idx=30)
        plot_emulator_error()
        plot_chi2_scatter()
        save_emulator_error()