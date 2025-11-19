"""
Find the best-fit HOD based on chi-squared statistic.
Note that this requires all the HOD folders to have all their measurements computed for the given cosmology (otherwise the compression functions will crash).

Usage:
    python best_fit.py --cosmology 0 --ells 0 2 --ndof 5 --diag --plot --log_level INFO
"""
import logging
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pycorr import TwoPointEstimator
from acm.utils.modules import get_class_from_module
from acm.utils.logging import setup_logging
 
def load_ref(dir: str|Path, phase: int = 0, los: list = ['x', 'y', 'z'], rebin: int = 1, **kwargs):
    """
    Load reference measurements from the specified directory.
    
    Parameters
    ----------
    dir : str | Path
        Directory containing the measurement files.
    phase : int
        Phase index to load.
    los : list
        List of line-of-sight directions to sum on.
    rebin : int
        Rebinning factor to apply to the measurements.
    **kwargs
        Additional keyword arguments to pass to the TwoPointEstimator call.
        
    Returns
    -------
    np.ndarray | tuple
        The esimated correlation function multipoles, rebinned as specified.
        If return_sep is True, returns (separations, multipoles), else returns multipoles only.
    """
    # TODO: Find a way to not encode path structure this strictly ? (how?)
    # TODO: add Mr as an argument ?
    dir = Path(dir) / f'AbacusSummit_base_c000_ph{phase:03d}' / 'measurements' / 'Mr-20' 
    fns = [dir / f'tpcf_los_{l}.npy' for l in los]
    cf = sum([TwoPointEstimator.load(fn).normalize() for fn in fns])
    cout = cf[::rebin](**kwargs)
    return cout

def load_covariance(dir: str|Path, n_phases: int = 25, **kwargs):
    """
    Load the covariance matrix from the specified directory.
    Calls load_ref for each phase and computes the covariance.
    
    Parameters
    ----------
    dir : str | Path
        Directory containing the measurement files.
    n_phases : int
        Number of phases to load.
    **kwargs
        Additional keyword arguments to pass to the load_ref call (return_sep will be enforced to False).
    
    Returns
    -------
    np.ndarray
        Covariance matrix computed from the loaded measurements.
    """
    kwargs['return_sep'] = False # Ensure we only get the data arrays
    cfs = [load_ref(dir, phase=phase, **kwargs) for phase in range(n_phases)]
    cov = np.cov([cf.flatten() for cf in cfs], rowvar=False)
    return cov

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
    parser.add_argument('--module', type=str, default='acm.observables.bgs', help='Base module path for observables')
    parser.add_argument('--cosmology', '-c', type=int, default=0, help='Cosmology index to find the fit for (default: 0)')
    parser.add_argument('--ells', '-e', type=int, nargs='+', default=[0], help='Multipole moments to consider (default: 0)')
    parser.add_argument('--ndof', '-n', type=int, help='Calculate chi-squared per degree of freedom')
    parser.add_argument('--diag', '-d', action='store_true', help='Use diagonal covariance matrix only')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the best-fit comparison')
    parser.add_argument('--log_level', type=str, default='warning', help='Set logging level (e.g., DEBUG, INFO)')
    args = parser.parse_args()
    ells = args.ells
    
    logger = logging.getLogger(__file__.split('/')[-1])
    setup_logging(level=args.log_level)
    
    logger.info(f'Using multipoles: {ells}')
    logger.info(f'Calculating chi-squared {f"with {args.ndof} degree of freedom" if args.ndof else ""}')

    cf_kwargs = dict(ells=ells, rebin=1) # Enforce no rebinning as we fit the CFs
    
    # Load compressed measurements for the specified cosmology
    cls = get_class_from_module(args.module, 'tpcf')
    obs = cls()
    data = obs.compress_data(cosmos=[args.cosmology], **cf_kwargs)
    y = obs.flatten_output(data.y, flat_output_dims=2).values # to 2D numpy array
    hods = obs.get_hod_from_files(cosmo_idx=args.cosmology)
    logger.info(f'Loaded data for cosmology c{args.cosmology:03d} with shape {data.y.shape}')
    
    # Load reference measurement and covariance
    ref_dir = '/pscratch/sd/s/sbouchar/SecondGen/CubicBox/BGS/z0.200'
    ref_poles = load_ref(ref_dir, **cf_kwargs)
    cov = load_covariance(ref_dir, **cf_kwargs)
    if args.diag:
        cov = np.diag(np.diag(cov)) # Use diagonal covariance only
    
    # TODO: Add covariance sanity checks here
    
    chi2_values = []
    expected = ref_poles.flatten()
    for cf in y:
        observed = cf
        
        dof = len(observed) - args.ndof if args.ndof else 1 # Default dof=1 if not specified
        chi2_value = chi2_ndof(observed, expected, cov, dof)
        
        chi2_values.append(chi2_value)
        logger.debug(f'Chi-squared: {chi2_value}')
    best_idx = np.argmin(chi2_values)

    print(f'c{args.cosmology:03d} Best-fit index: {best_idx} (HOD {hods[best_idx]}) with chi-squared: {chi2_values[best_idx]}')

    if args.plot:
        s = data['s'].values
        best_poles = y[best_idx].reshape(len(ells), -1)
        fig, ax = plot_best_fit(s, ref_poles, best_poles, ells, hods[best_idx])
        fig.savefig(f'best_fit_c{args.cosmology:03d}_ells{tuple(ells)}.png', dpi=300, bbox_inches='tight')