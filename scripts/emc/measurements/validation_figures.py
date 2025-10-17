from jaxpower import read
from pycorr import TwoPointCorrelationFunction
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_spectrum_fns(cosmo_idx=0, phase_idx=0, n_hod=1, sim_type='base'):
    if sim_type == 'base':
        base_dir = Path(f'/pscratch/sd/e/epaillas/emc/v1.2/abacus/base/spectrum/c{cosmo_idx:03}_ph000/seed0/')
        handle = f'mesh2_spectrum_poles_c{cosmo_idx:03}_hod???.h5'
        fns = sorted(base_dir.glob(handle))
        return fns[:n_hod]
    elif sim_type == 'small':
        base_dir = Path(f'/pscratch/sd/e/epaillas/emc/v1.2/abacus/small/spectrum/')
        handle = f'mesh2_spectrum_poles_ph*.h5'
        fns = sorted(base_dir.glob(handle))
        return fns

def get_recon_spectrum_fns(cosmo_idx=0, phase_idx=0, n_hod=1):
    base_dir = Path(f'/pscratch/sd/e/epaillas/emc/v1.2/abacus/raw_measurements/recon_spectrum/c{cosmo_idx:03}_ph000/seed0/')
    handle = f'mesh2_recon_spectrum_poles_c{cosmo_idx:03}_hod???.h5'
    fns = sorted(base_dir.glob(handle))
    return fns[:n_hod]

def read_spectrum(filename):
    data = read(filename)
    kmin, kmax = 0.01, 0.7
    data = data.select(k=slice(0, None, 5)).select(k=(kmin, kmax))
    poles = [data.get(ell) for ell in (0, 2, 4)]
    k = poles[0].coords('k')
    return k, poles

def get_tpcf_fns(cosmo_idx=0, phase_idx=0, n_hod=1):
    base_dir = Path(f'/pscratch/sd/e/epaillas/emc/v1.2/abacus/raw_measurements/tpcf/c{cosmo_idx:03}_ph000/seed0/')
    handle = f'tpcf_smu_c{cosmo_idx:03}_hod???.npy'
    fns = sorted(base_dir.glob(handle))
    return fns[:n_hod]

def get_recon_tpcf_fns(cosmo_idx=0, phase_idx=0, n_hod=1):
    base_dir = Path(f'/pscratch/sd/e/epaillas/emc/v1.2/abacus/raw_measurements/recon_tpcf/c{cosmo_idx:03}_ph000/seed0/')
    handle = f'recon_tpcf_smu_c{cosmo_idx:03}_hod???.npy'
    fns = sorted(base_dir.glob(handle))
    return fns[:n_hod]

def read_tpcf(filename):
    data = TwoPointCorrelationFunction.load(filename)[::4]
    s, poles = data(ells=(0, 2, 4), return_sep=True)
    return s, poles

def plot_spectrum():
    pk_fns = np.concatenate([get_spectrum_fns(cosmo_idx=i) for i in range(5)])
    k, _ = read_spectrum(pk_fns[0])
    poles = [read_spectrum(fn)[1] for fn in pk_fns]

    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for pole in poles:
        for ell in (0, 2, 4):
            ax[ell//2].plot(k, k*pole[ell//2], ls='-', lw=1.0)
    for aa in ax:
        aa.set_xlabel(r'$k$ [h/Mpc]', fontsize=13)
    for ell in (0, 2, 4):
        ax[ell//2].set_ylabel(r'$k P_{%d}(k)$ [h$^{-2}$ Mpc$^{2}$]' % ell, fontsize=13)
    plt.tight_layout()
    plt.savefig('fig/spectrum_measurements.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_spectrum_small():
    pk_fns = get_spectrum_fns(sim_type='small')[:20]
    k, _ = read_spectrum(pk_fns[0])
    poles = [read_spectrum(fn)[1] for fn in pk_fns]

    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for pole in poles:
        for ell in (0, 2, 4):
            ax[ell//2].plot(k, k*pole[ell//2], ls='-', lw=0.5, color='gray')
    # plot mean
    mean_poles = np.mean(np.array([[pole[ell//2] for ell in (0, 2, 4)] for pole in poles]), axis=0)
    std_poles = np.std(np.array([[pole[ell//2] for ell in (0, 2, 4)] for pole in poles]), axis=0) / 5
    for ell in (0, 2, 4):
        ax[ell//2].errorbar(k, k*mean_poles[ell//2], yerr=k*std_poles[ell//2], fmt='o', markersize=1, capsize=0.5, elinewidth=1.0, label='Mean', color='k')
    for aa in ax:
        aa.set_xlabel(r'$k$ [h/Mpc]', fontsize=13)
    for ell in (0, 2, 4):
        ax[ell//2].set_ylabel(r'$k P_{%d}(k)$ [h$^{-2}$ Mpc$^{2}$]' % ell, fontsize=13)
    plt.tight_layout()
    plt.savefig('fig/spectrum_measurements_small.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_recon_spectrum():
    pk_fns = np.concatenate([get_recon_spectrum_fns(cosmo_idx=i) for i in range(5)])
    k, _ = read_spectrum(pk_fns[0])
    poles = [read_spectrum(fn)[1] for fn in pk_fns]

    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for pole in poles:
        for ell in (0, 2, 4):
            ax[ell//2].plot(k, k*pole[ell//2], ls='-', lw=1.0)
    for aa in ax:
        aa.set_xlabel(r'$k$ [h/Mpc]', fontsize=13)
    for ell in (0, 2, 4):
        ax[ell//2].set_ylabel(r'$k P_{%d}(k)$ [h$^{-2}$ Mpc$^{2}$]' % ell, fontsize=13)
    plt.tight_layout()
    plt.savefig('fig/recon_spectrum_measurements.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_spectrum_vs_recon_spectrum():
    pk_fns = np.concatenate([get_spectrum_fns(cosmo_idx=i, n_hod=1) for i in range(1)])
    recon_pk_fns = np.concatenate([get_recon_spectrum_fns(cosmo_idx=i, n_hod=1) for i in range(1)])
    k, _ = read_spectrum(pk_fns[0])
    poles = [read_spectrum(fn)[1] for fn in pk_fns]
    recon_poles = [read_spectrum(fn)[1] for fn in recon_pk_fns]

    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for pole, recon_pole in zip(poles, recon_poles):
        for ell in (0, 2, 4):
            ax[ell//2].plot(k, k*pole[ell//2], ls='-', lw=1.0, label='Pre-recon')
            ax[ell//2].plot(k, k*recon_pole[ell//2], ls='--', lw=1.0, label='Post-recon')
    for aa in ax:
        aa.set_xlabel(r'$k$ [h/Mpc]', fontsize=13)
    for ell in (0, 2, 4):
        ax[ell//2].set_ylabel(r'$k P_{%d}(k)$ [h$^{-2}$ Mpc$^{2}$]' % ell, fontsize=13)
        ax[ell//2].legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('fig/compare_spectrum_vs_recon_spectrum_measurements.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_tpcf():
    tpcf_fns = np.concatenate([get_tpcf_fns(cosmo_idx=i) for i in range(5)])
    s, _ = read_tpcf(tpcf_fns[0])
    poles = [read_tpcf(fn)[1] for fn in tpcf_fns]

    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for pole in poles:
        for ell in (0, 2, 4):
            ax[ell//2].plot(s, s*s*pole[ell//2], ls='-', lw=1.0)
    for aa in ax:
        aa.set_xlabel(r'$s$ [Mpc/h]', fontsize=13)
    for ell in (0, 2, 4):
        ax[ell//2].set_ylabel(r'$s^2 \xi_{%d}(s)$ [Mpc/h]' % ell, fontsize=13)
    plt.tight_layout()
    plt.savefig('fig/tpcf_measurements.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_recon_tpcf():
    tpcf_fns = np.concatenate([get_recon_tpcf_fns(cosmo_idx=i) for i in range(5)])
    s, _ = read_tpcf(tpcf_fns[0])
    poles = [read_tpcf(fn)[1] for fn in tpcf_fns]

    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for pole in poles:
        for ell in (0, 2, 4):
            ax[ell//2].plot(s, s*s*pole[ell//2], ls='-', lw=1.0)
    for aa in ax:
        aa.set_xlabel(r'$s$ [Mpc/h]', fontsize=13)
    for ell in (0, 2, 4):
        ax[ell//2].set_ylabel(r'$s^2 \xi_{%d}(s)$ [Mpc/h]' % ell, fontsize=13)
    plt.tight_layout()
    plt.savefig('fig/recon_tpcf_measurements.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_tpcf_vs_recon_tpcf():
    tpcf_fns = np.concatenate([get_tpcf_fns(cosmo_idx=i, n_hod=1) for i in range(1)])
    recon_tpcf_fns = np.concatenate([get_recon_tpcf_fns(cosmo_idx=i, n_hod=1) for i in range(1)])
    s, _ = read_tpcf(tpcf_fns[0])
    poles = [read_tpcf(fn)[1] for fn in tpcf_fns]
    recon_poles = [read_tpcf(fn)[1] for fn in recon_tpcf_fns]

    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for pole, recon_pole in zip(poles, recon_poles):
        for ell in (0, 2, 4):
            ax[ell//2].plot(s, s*s*pole[ell//2], ls='-', lw=1.0, label='Pre-recon')
            ax[ell//2].plot(s, s*s*recon_pole[ell//2], ls='--', lw=1.0, label='Post-recon')
    for aa in ax:
        aa.set_xlabel(r'$s$ [Mpc/h]', fontsize=13)
    for ell in (0, 2, 4):
        ax[ell//2].set_ylabel(r'$s^2 \xi_{%d}(s)$ [Mpc/h]' % ell, fontsize=13)
        ax[ell//2].legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('fig/compare_tpcf_vs_recon_tpcf_measurements.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    plot_spectrum()
    plot_recon_spectrum()
    compare_spectrum_vs_recon_spectrum()
    plot_tpcf()
    plot_recon_tpcf()
    compare_tpcf_vs_recon_tpcf()
    plot_spectrum_small()
