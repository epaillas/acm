import argparse  # noqa: INP001
from pathlib import Path

import lsstypes
import matplotlib.pyplot as plt
import numpy as np
import yaml
from compress_files import select
from measure_box import get_estimator

from acm.estimators.compression import Compressor
from acm.utils.logging import get_logger_for_script, setup_logging
from acm.utils.scripts import NumpyLoader

logger = get_logger_for_script(__file__)

def chi2(
    observed: np.ndarray[tuple[int], np.dtype[np.float64]],
    expected: np.ndarray[tuple[int], np.dtype[np.float64]],
    covariance: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    dof: int = 1,
) -> float:
    """
    Calculate the chi-squared statistic.

    Parameters
    ----------
    observed : np.ndarray
        The observed data points (1D array).
    expected : np.ndarray
        The expected data points (1D array).
    covariance : np.ndarray
        The covariance matrix of the data (2D array).
    dof : int, optional
        The number of degrees of freedom (number of data points minus number of fitted parameters).
        If not provided, defaults to 1 (i.e., no adjustment for degrees of freedom).

    Returns
    -------
    float
        The chi-squared statistic.
    """
    diff = observed - expected
    inv_cov = np.linalg.inv(covariance)
    chi2 = diff @ inv_cov @ diff
    return chi2 / dof

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate data best-fit statistics")
    parser.add_argument("--root_obs", type=str, required=True, help="Root directory containing the measurements")
    parser.add_argument("--root_exp", type=str, required=True, help="Root directory containing the expected measurements")
    parser.add_argument("--measurement", type=str, required=True, help="Measurement to process")
    parser.add_argument("--estimator_config", type=str, required=True, help="YAML file containing estimator parameters.")
    parser.add_argument('--Mr', type=float, default=-20, help='Magnitude threshold for the measurements (default: -20)')
    parser.add_argument('--cosmo_idx', type=int, default=0, help='Cosmology index to use (default: 0)')
    parser.add_argument('--ndof', type=int, help='Calculate chi-squared per degree of freedom')
    parser.add_argument('--diag', action='store_true', help='Use diagonal covariance matrix only')
    parser.add_argument('--plot', action='store_true', help='Plot the best-fit comparison')
    parser.add_argument("--log_level", type=str, default='warning', help="Set logging level (e.g., DEBUG, INFO)")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    with Path(args.estimator_config).open() as f:
        estimator_config = yaml.load(f, Loader=NumpyLoader)  # noqa: S506

    stat_name = args.measurement
    args = estimator_config.get(stat_name, {})
    load_args = args.get("load", {})
    reader = get_estimator(stat_name).load

    # NOTE: using hardcoded pattern/index structure for those files, as they handle outputs of measure_box.py
    pattern_obs = f"c{args.cosmo_idx}" + r"_ph{phase_idx}/seed{seed}/hod{hod_idx}/" + stat_name + r"_los-{los}.h5"
    pattern_exp = f"AbacusSummit_base_c{args.cosmo_idx}" + r"_ph{phase_idx}/measurements/" + f"Mr{args.Mr}/{stat_name}" + r"_los-{los}.h5"
    ignore_index = ["los"]

    # Compress measurements
    compressor = Compressor(root=Path(args.root_obs), pattern=pattern_obs)
    group_obs = compressor.read(reader=reader, ignore_index=ignore_index, **load_args)
    group_obs = select(stat_name, group_obs)
    group_obs = group_obs.merge(method=lsstypes.mean)  # Merge identical indices
    data = np.array([obj.data for obj in group_obs]) # Flattened arrays of observed data

    # Load expected data and covariance matrix
    compressor = Compressor(root=Path(args.root_exp), pattern=pattern_exp)
    group_exp = compressor.read(reader=reader, ignore_index=ignore_index, **load_args)
    group_exp = select(stat_name, group_exp)
    group_exp = group_exp.merge(method=lsstypes.mean)  # Merge identical indices
    expected = np.asarray(group_exp[0].data) # Flattened array of expected data
    covariance = np.cov(np.array([obj.data for obj in group_exp]), rowvar=False)

    values = []
    for observed in data:
        dof = len(observed) - args.ndof if args.ndof else 1
        chi2_value = chi2(observed, expected, covariance, dof=dof)
        logger.info(f"Chi-squared value: {chi2_value}")

    idx = np.argmin(values)
    str_idx = ", ".join([f"{k}={v}" for k, v in group_obs[idx].indexes.items()])
    logger.info(f"Best-fit found for {str_idx} ({idx=}) with chi-squared value: {values[idx]}")

    if args.plot:
        plotter = get_estimator(stat_name).plot
        fig, ax = plotter(group_obs[idx], ls='-')
        fig, ax = plotter(group_exp[0], ls='--', fig=fig, ax=ax)
        handles = [
            plt.Line2D([0], [0], lw=2, ls='-', label='Observed'),
            plt.Line2D([0], [0], lw=2, ls='--', label='Expected')
        ]
        ax.legend(handles=handles)
        ax.set_title(f'Best-fit comparison for {stat_name} ({str_idx})')
        fig.savefig(f"best_fit_{stat_name}.png", dpi=300, bbox_inches='tight')
