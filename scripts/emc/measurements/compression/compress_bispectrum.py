"""Compress EMC bispectrum measurements."""

import argparse
from collections.abc import Sequence

import numpy as np

from _common import CompressionSpec, run_cli, single_file_reader


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add bispectrum-specific CLI arguments."""
    parser.add_argument("--kmin", type=float, default=0.016)
    parser.add_argument("--kmax", type=float, default=0.285)
    parser.add_argument("--rebin", type=int, default=3)
    parser.add_argument("--ells", type=str, default="0,2")


def parse_ells(value: str) -> list[int]:
    """Parse multipole values from a comma-separated string."""
    return [int(item) for item in value.split(",") if item]


def bispectrum_reader(files):
    """Read one jaxpower bispectrum measurement."""
    from jaxpower import read

    return single_file_reader(files, read)


def make_bispectrum_postprocess(
    kmin: float,
    kmax: float,
    rebin: int,
    ells: Sequence[int],
):
    """Build a postprocessor for jaxpower bispectrum measurements."""

    def postprocess(data):
        rows = []
        bin_idx = None
        for measurement in data:
            measurement = measurement.select(k=slice(0, None, rebin)).select(
                k=(kmin, kmax)
            )
            poles = [measurement.get(ell) for ell in ells]
            k = poles[0].coords("k")
            weights = k.prod(axis=1) / 1e5
            rows.append([weights * pole.value().real for pole in poles])
            if bin_idx is None:
                bin_idx = np.arange(len(k))
        return np.asarray(rows), {"ells": list(ells), "bin_idx": bin_idx}

    return postprocess


def build_spec(args: argparse.Namespace) -> CompressionSpec:
    """Build the compression spec for bispectrum measurements."""
    from acm.observables.emc.bispectrum_module import GalaxyBispectrumMultipoles

    ells = parse_ells(args.ells)
    postprocess = make_bispectrum_postprocess(args.kmin, args.kmax, args.rebin, ells)
    return CompressionSpec(
        stat_name="bispectrum",
        observable_cls=GalaxyBispectrumMultipoles,
        base_root="base/bispectrum",
        base_pattern=(
            "c{cosmo_idx}_ph{phase_idx}/seed{seed}/"
            "mesh3_spectrum_poles_*_hod{hod_idx}.h5"
        ),
        reader=bispectrum_reader,
        postprocess=postprocess,
        covariance_root="small/bispectrum",
        covariance_pattern="mesh3_spectrum_poles_ph{phase_idx}.h5",
        covariance_reader=bispectrum_reader,
        covariance_postprocess=postprocess,
    )


def main() -> None:
    """Run bispectrum compression from the command line."""
    run_cli(
        build_spec,
        description="Compress EMC bispectrum measurements.",
        add_arguments=add_arguments,
    )


if __name__ == "__main__":
    main()
