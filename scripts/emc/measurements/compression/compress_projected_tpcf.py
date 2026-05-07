"""Compress EMC projected TPCF measurements."""

import argparse

import numpy as np

from _common import CompressionSpec, run_cli, single_file_reader


def projected_tpcf_reader(files):
    """Read one projected TPCF measurement."""
    from pycorr import TwoPointCorrelationFunction

    return single_file_reader(files, TwoPointCorrelationFunction.load)


def projected_tpcf_postprocess(data):
    """Convert projected TPCF measurements to an array plus coordinates."""
    rows = []
    r_p = None
    for measurement in data:
        r_p, w_p = measurement(pimax=None, return_sep=True)
        rows.append(w_p)
    return np.asarray(rows), {"r_p": r_p}


def build_spec(args: argparse.Namespace) -> CompressionSpec:
    """Build the compression spec for projected TPCF measurements."""
    del args
    from acm.observables.emc.projected_tpcf_module import (
        ProjectedGalaxyCorrelationFunction,
    )

    return CompressionSpec(
        stat_name="projected_tpcf",
        observable_cls=ProjectedGalaxyCorrelationFunction,
        base_root="base/projected_tpcf",
        base_pattern=(
            "c{cosmo_idx}_ph{phase_idx}/seed{seed}/"
            "tpcf_rppi_*_hod{hod_idx}.npy"
        ),
        reader=projected_tpcf_reader,
        postprocess=projected_tpcf_postprocess,
        covariance_root="small/projected_tpcf",
        covariance_pattern="tpcf_rppi_ph{phase_idx}.npy",
        covariance_reader=projected_tpcf_reader,
        covariance_postprocess=projected_tpcf_postprocess,
    )


def main() -> None:
    """Run projected TPCF compression from the command line."""
    run_cli(
        build_spec,
        description="Compress EMC projected TPCF measurements.",
    )


if __name__ == "__main__":
    main()
