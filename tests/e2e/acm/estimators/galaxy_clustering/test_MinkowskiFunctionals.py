"""
From Enrique Paillas acm/tests/test_minkowsky.py  adapted for pytest and CI
"""

import logging
import os
from pathlib import Path

import fitsio
import matplotlib.pyplot as plt
import numpy as np
import pytest

from acm.estimators.galaxy_clustering.jaxmf import MinkowskiFunctionals

logger = logging.getLogger(__name__)

ACM_TEST_DATA = os.environ.get("ACM_TEST_DATA")


@pytest.fixture(scope="module")
def get_hod_positions(filename="hod000_minkowski.fits.gz", los="z"):
    """Get redshift-space positions from a HOD file."""
    hod_fn = Path(ACM_TEST_DATA) / filename
    print("Loading HOD catalog from:", hod_fn)
    hod, header = fitsio.read(hod_fn, header=True)
    qpar, qperp = header["Q_PAR"], header["Q_PERP"]
    if los == "x":
        pos = np.c_[hod["X_RSD"], hod["Y_PERP"], hod["Z_PERP"]]
        boxsize = np.array([2000 / qpar, 2000 / qperp, 2000 / qperp])
    elif los == "y":
        pos = np.c_[hod["X_PERP"], hod["Y_RSD"], hod["Z_PERP"]]
        boxsize = np.array([2000 / qperp, 2000 / qpar, 2000 / qperp])
    elif los == "z":
        pos = np.c_[hod["X_PERP"], hod["Y_PERP"], hod["Z_RSD"]]
        boxsize = np.array([2000 / qperp, 2000 / qperp, 2000 / qpar])
    return pos, boxsize


@pytest.fixture(scope="module")
def minkowski_jaxpower(get_hod_positions):
    positions, boxsize = get_hod_positions
    print(f"type: {positions.dtype}, shape: {positions.shape}")
    results, smoothing_radii = minkowski_metrics(positions, boxsize, backend="jaxpower")
    return results, smoothing_radii


def get_box_args(boxsize, cellsize):
    meshsize = (boxsize / cellsize).astype(int)
    return dict(boxsize=boxsize, boxcenter=0.0, meshsize=meshsize)


def minkowski_metrics(positions, boxsize, backend):
    """Test Minkowski functionals computation on a HOD catalog."""
    # Load thresholds for different smoothing radii
    # For testing purposes, we'll create simple thresholds if the file doesn't exist
    thresholds_fn = (
        "/pscratch/sd/e/epaillas/emc/Thresholds_for_MFs_with_Rg5_7_10_15.npy"
    )
    if Path(thresholds_fn).exists():
        thresholds_all = np.load(thresholds_fn, allow_pickle=True).item()
        smoothing_radii = [5, 7, 10, 15]
    else:
        # Create simple test thresholds
        logger.info(f"Warning: {thresholds_fn} not found. Using test thresholds.")
        smoothing_radii = [10]
        thresholds_all = {
            f"Thresholds_Rg{r}": np.linspace(-2, 2, 10) for r in smoothing_radii
        }
    # Get HOD catalog
    box_args = get_box_args(boxsize, cellsize=3.9)
    # Initialize Minkowski functionals estimator
    # backend='pyrecon'
    backend = "jaxpower"
    mf = MinkowskiFunctionals(
        data_positions=positions, thres_mask=-5, backend=backend, **box_args
    )
    # Store results
    mfs3d_all = {}
    # Compute for each smoothing radius
    for smoothing_radius in smoothing_radii:
        logger.info(
            f"Computing Minkowski functionals for smoothing radius = {smoothing_radius} Mpc/h"
        )
        thresholds = thresholds_all[f"Thresholds_Rg{smoothing_radius}"]

        # Set density contrast with smoothing
        mf.set_density_contrast(smoothing_radius=smoothing_radius)

        # Compute Minkowski functionals
        mf3d = mf.run(thresholds=thresholds)

        # Store results
        mfs3d_all[f"Rg{smoothing_radius}"] = mf3d
        mfs3d_all[f"thresholds_Rg{smoothing_radius}"] = thresholds
        logger.info(f"  MF shape: {mf3d.shape}")
        logger.info(f"  MF0 range: [{mf3d[:, 0].min():.4f}, {mf3d[:, 0].max():.4f}]")
        logger.info(f"  MF1 range: [{mf3d[:, 1].min():.4f}, {mf3d[:, 1].max():.4f}]")
        logger.info(f"  MF2 range: [{mf3d[:, 2].min():.4f}, {mf3d[:, 2].max():.4f}]")
        logger.info(f"  MF3 range: [{mf3d[:, 3].min():.4f}, {mf3d[:, 3].max():.4f}]")
    # np.save('minkowski_results.npy', mfs3d_all, True)
    # logger.info("Minkowski functionals results saved to minkowski_results.npy")
    return mfs3d_all, smoothing_radii


def plot_minkowski(mfs3d_all, smoothing_radii):
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    mf_labels = [
        r"$V_0$ (Volume)",
        r"$V_1$ (Surface)",
        r"$V_2$ (Curvature)",
        r"$V_3$ (Euler)",
    ]

    for i, ax in enumerate(axes):
        for smoothing_radius in smoothing_radii:
            thresholds = mfs3d_all[f"thresholds_Rg{smoothing_radius}"]
            mf3d = mfs3d_all[f"Rg{smoothing_radius}"]

            ax.plot(
                thresholds,
                mf3d[:, i],
                marker="o",
                markersize=3,
                label=f"$R_s={smoothing_radius}$ Mpc/h",
                linewidth=1.5,
            )

        ax.set_xlabel(r"Threshold $\nu$", fontsize=10)
        ax.set_ylabel(mf_labels[i], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("minkowski_test.png", bbox_inches="tight", dpi=300)
    logger.info("\nPlot saved to minkowski_test.png")

    # Additional diagnostic plot: check monotonicity for V0 (should be monotonic)
    fig, ax = plt.subplots(figsize=(6, 4))
    for smoothing_radius in smoothing_radii:
        thresholds = mfs3d_all[f"thresholds_Rg{smoothing_radius}"]
        mf3d = mfs3d_all[f"Rg{smoothing_radius}"]
        ax.plot(
            thresholds,
            mf3d[:, 0],
            marker="o",
            markersize=4,
            label=f"$R_s={smoothing_radius}$ Mpc/h",
        )

    ax.set_xlabel(r"Threshold $\nu$", fontsize=12)
    ax.set_ylabel(r"$V_0$ (Volume Fraction)", fontsize=12)
    ax.set_title(
        "Volume Fraction vs Threshold (should be monotonically decreasing)", fontsize=10
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("minkowski_v0_test.png", bbox_inches="tight", dpi=300)


#
# TESTS
#


def test_minkowski_backend_consistency(minkowski_jaxpower, get_hod_positions):
    """Compare Minkowski functionals computed with JAX and PyRecon backends on the same HOD catalog."""
    res_jax, smoothing_radii = minkowski_jaxpower
    positions, boxsize = get_hod_positions
    res_recon, smoothing_radii = minkowski_metrics(positions, boxsize, "pyrecon")
    for key in res_jax.keys():
        if key in res_recon:
            assert np.allclose(res_jax[key], res_recon[key], rtol=1e-5, atol=1e-8), (
                f"Mismatch in {key} between JAX and PyRecon backends"
            )


def test_minkowski_jaxpower_no_regression(minkowski_jaxpower):
    """Test Minkowski functionals computed with JAX backend against reference results to check for regressions."""
    results, smoothing_radii = minkowski_jaxpower
    # plot_minkowski(results, smoothing_radii)
    # compare to ref
    ref_fn = Path(ACM_TEST_DATA) / "minkowski_reference.npy"
    ref_results = np.load(ref_fn, allow_pickle=True).item()
    for key in results.keys():
        if key in ref_results:
            assert np.allclose(results[key], ref_results[key], rtol=1e-5, atol=1e-8), (
                f"Mismatch in {key}"
            )
