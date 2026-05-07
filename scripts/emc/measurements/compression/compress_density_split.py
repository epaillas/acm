"""Shared density-split compression definitions."""

import argparse
from collections.abc import Sequence

import numpy as np

from _common import CompressionSpec, run_cli, single_file_reader


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add density-split-specific CLI arguments."""
    parser.add_argument("--smin", type=float, default=0.0)
    parser.add_argument("--smax", type=float, default=150.0)
    parser.add_argument("--rebin", type=int, default=4)
    parser.add_argument("--ells", type=str, default="0,2")
    parser.add_argument("--quantiles", type=str, default="0,1,3,4")


def parse_ints(value: str) -> list[int]:
    """Parse integer values from a comma-separated string."""
    return [int(item) for item in value.split(",") if item]


def density_split_reader(files):
    """Read one density-split base measurement."""

    def load(path):
        return np.load(path, allow_pickle=True)

    return single_file_reader(files, load)


def density_split_covariance_reader(files):
    """Read one density-split covariance measurement."""
    from acm.estimators.galaxy_clustering.base import BaseEstimator

    return single_file_reader(files, BaseEstimator.read)


def make_density_split_postprocess(
    smin: float,
    smax: float,
    rebin: int,
    ells: Sequence[int],
    quantiles: Sequence[int],
):
    """Build a postprocessor for density-split base measurements."""

    def postprocess(data):
        rows = []
        s = None
        for measurement in data:
            quantile_rows = []
            for quantile in quantiles:
                result = measurement[quantile][::rebin]
                result.select((smin, smax))
                s, multipoles = result(ells=ells, return_sep=True)
                quantile_rows.append(
                    np.concatenate(multipoles).reshape(len(ells), -1)
                )
            rows.append(quantile_rows)
        return (
            np.asarray(rows),
            {"quantiles": list(quantiles), "ells": list(ells), "s": s},
        )

    return postprocess


def make_density_split_covariance_postprocess(
    smin: float,
    smax: float,
    rebin: int,
    ells: Sequence[int],
    quantiles: Sequence[int],
):
    """Build a postprocessor for density-split covariance measurements."""

    def postprocess(data):
        rows = []
        s = None
        for measurement in data:
            quantile_rows = []
            for quantile in quantiles:
                xi = measurement.get(quantiles=quantile).select(
                    s=slice(0, None, rebin)
                )
                xi = xi.select(s=(smin, smax))
                poles = xi.project(ells=ells)
                s = poles.get(ells=ells[0]).coords("s")
                multipoles = [poles.get(ells=ell).value() for ell in ells]
                quantile_rows.append(
                    np.concatenate(multipoles).reshape(len(ells), -1)
                )
            rows.append(quantile_rows)
        return (
            np.asarray(rows),
            {"quantiles": list(quantiles), "ells": list(ells), "s": s},
        )

    return postprocess


def build_density_split_spec(
    args: argparse.Namespace,
    stat_name: str,
    measurement_root: str,
    observable_cls: type,
) -> CompressionSpec:
    """Build a compression spec for a density-split statistic."""
    ells = parse_ints(args.ells)
    quantiles = parse_ints(args.quantiles)
    return CompressionSpec(
        stat_name=stat_name,
        observable_cls=observable_cls,
        base_root="base/density_split",
        base_pattern=(
            "c{cosmo_idx}_ph{phase_idx}/seed{seed}/"
            f"{measurement_root}_poles_*_hod{{hod_idx}}.npy"
        ),
        reader=density_split_reader,
        postprocess=make_density_split_postprocess(
            args.smin, args.smax, args.rebin, ells, quantiles
        ),
        covariance_root="small/density_split",
        covariance_pattern=f"{measurement_root}_poles_ph{{phase_idx}}.h5",
        covariance_reader=density_split_covariance_reader,
        covariance_postprocess=make_density_split_covariance_postprocess(
            args.smin, args.smax, args.rebin, ells, quantiles
        ),
    )


def run_density_split_cli(
    build_spec,
    description: str,
) -> None:
    """Run a density-split compression CLI."""
    run_cli(build_spec, description=description, add_arguments=add_arguments)
