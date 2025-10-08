from jaxpower import read
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_spectrum_fns(cosmo_idx=0, phase_idx=0):
    base_dir = Path(f'/pscratch/sd/e/epaillas/emc/v1.2/abacus/raw_measurements/spectrum/c{cosmo_idx:03}_ph000/seed0/')
    handle = f'mesh2_spectrum_poles_c{cosmo_idx:03}_hod???.h5'
    fns = sorted(base_dir.glob(handle))
    return fns

def read_spectrum(filename):
    data = read(filename)
    kmin, kmax = 0.01, 0.7
    data = data.select(k=slice(0, None, 5)).select(k=(kmin, kmax))
    poles = [data.get(ell) for ell in (0, 2, 4)]
    k = poles[0].coords('k')
    return k, poles

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


if __name__ == '__main__':

    plot_spectrum()
