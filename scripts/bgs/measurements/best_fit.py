"""
Find the best-fit HOD based on chi-squared statistic.

Usage:
    python best_fit.py --cosmology 0 --ells 0 2 --ndof 5 --diag --plot --log_level INFO
"""
import logging
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pycorr import TwoPointEstimator
from acm.utils.logging import setup_logging

def chi2(observed, expected, covariance):
    """Calculate the chi-squared statistic."""
    diff = observed - expected
    inv_cov = np.linalg.inv(covariance)
    chi2 = diff @ inv_cov @ diff
    return chi2

def chi2_ndof(observed, expected, covariance, dof):
    """Calculate the chi-squared per degree of freedom."""
    chi2_value = chi2(observed, expected, covariance)
    return chi2_value / dof

def load_cfs(dir: str|Path):
    """Load the CFs from the specified directory."""
    dir = Path(dir)
    fns = sorted(dir.glob('hod*/tpcf_los_*.npy'))
    cfs = []
    for i in range(0, len(fns), 3):
        cf = [TwoPointEstimator.load(fns[i+j]) for j in range(3)]
        cf = cf[0].concatenate_x(cf)
        cfs.append(cf)
    return cfs

def load_ref(dir: str|Path, phase: int = 0):
    """Load reference measurements from the specified directory."""
    dir = Path(dir)
    fns = dir.glob(f'AbacusSummit_base_c000_ph{phase:03d}/measurements/Mr-20/tpcf_los_*.npy')
    # TODO: Find a way to not encode path structure this strictly ?
    cfs = [TwoPointEstimator.load(fn) for fn in fns]
    cf = cfs[0].concatenate_x(cfs)
    return cf

def load_covariance(dir: str|Path, ells=(0, 2)):
    """Load the covariance matrix from the specified directory."""
    cfs = [load_ref(dir, phase=phase) for phase in range(25)]
    cov = np.cov([cf(ells=ells, return_sep=False).flatten() for cf in cfs], rowvar=False)
    return cov

def plot_best_fit(s, ref_poles, best_poles, ells, hod):
    """Plot the best-fit comparison."""
    fig, ax = plt.subplots()

    for i, p in enumerate(ells):
        ax.plot(s, ref_poles[i]*s**2, ls='--', c=f'C{i}')
        ax.plot(s, best_poles[i]*s**2, c=f'C{i}')
    
    handles = [
        plt.Line2D([0], [0], color='k', label='SecondGen mock'),
        plt.Line2D([0], [0], color='k', ls='--', label=f'Best HOD fit (HOD {hod})'),
    ]
    ax.legend(handles=handles)
    ax.set_title('Best-fit HOD comparison')
    ax.set_xlabel('s [Mpc/h]')
    ax.set_ylabel(r'$s^2 \xi_\ell(s)$ [Mpc/h]$^2$')
    return fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find best-fit HOD based on chi-squared statistic.')
    parser.add_argument('--cosmology', '-c', type=int, default=0, help='Cosmology index to find the fit for (default: 0)')
    parser.add_argument('--ells', '-e', type=int, nargs='+', default=[0], help='Multipole moments to consider (default: 0)')
    parser.add_argument('--ndof', '-n', type=int, help='Calculate chi-squared per degree of freedom')
    parser.add_argument('--diag', '-d', action='store_true', help='Use diagonal covariance matrix only')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the best-fit comparison')
    parser.add_argument('--log_level', type=str, help='Set logging level (e.g., DEBUG, INFO)', default='warning')
    args = parser.parse_args()
    ells = tuple(args.ells)
    
    logger = logging.getLogger(__file__.split('/')[-1])
    setup_logging(level=args.log_level)
    
    logger.info(f'Using multipoles: {ells}')
    logger.info(f'Calculating chi-squared {f"with {args.ndof} degree of freedom" if args.ndof else ""}')

    data_dir = Path(f'/pscratch/sd/s/sbouchar/acm/bgs/measurements/base/c{args.cosmology:03d}_ph000/seed0')
    hods = [f.stem.lstrip('hod') for f in sorted(data_dir.glob('hod*'))]
    cfs = load_cfs(data_dir)
    
    ref_dir = '/pscratch/sd/s/sbouchar/SecondGen/CubicBox/BGS/z0.200'
    ref = load_ref(ref_dir)
    if args.diag:
        cov = np.diag(np.diag(load_covariance(ref_dir, ells=ells))) # Use diagonal covariance
    else:
        cov = load_covariance(ref_dir, ells=ells)
    
    # TODO: Add covariance sanity checks here
    
    chi2_values = []
    for cf in cfs:
        observed = cf(ells=ells, return_sep=False).flatten()
        expected = ref(ells=ells, return_sep=False).flatten()
        if args.ndof:
            dof = len(observed) - args.ndof  # Adjust as necessary for number of fitted parameters
            chi2_value = chi2_ndof(observed, expected, cov, dof)
        else:
            chi2_value = chi2(observed, expected, cov)
        chi2_values.append(chi2_value)
        logger.debug(f'Chi-squared: {chi2_value}')
    best_idx = np.argmin(chi2_values)

    print(f'c{args.cosmology:03d} Best-fit index: {best_idx} (HOD {hods[best_idx]}) with chi-squared: {chi2_values[best_idx]}')

    if args.plot:
        s, ref_poles = ref(ells=ells, return_sep=True)
        s, best_poles = cfs[best_idx](ells=ells, return_sep=True)
        fig, ax = plot_best_fit(s, ref_poles, best_poles, ells, hods[best_idx])
        fig.savefig(f'best_fit_c{args.cosmology:03d}_ells{ells}.png', dpi=300, bbox_inches='tight')