"""Compress EMC power spectrum measurements with generic compression utilities."""

import argparse
from collections.abc import Sequence

import numpy as np

from _common import CompressionSpec, run_cli, single_file_reader


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add power-spectrum-specific CLI arguments."""
    parser.add_argument("--kmin", type=float, default=0.0126)
    parser.add_argument("--kmax", type=float, default=0.7)
    parser.add_argument("--rebin", type=int, default=13)
    parser.add_argument("--ells", type=str, default="0,2,4")


def parse_ells(value: str) -> list[int]:
    """Parse multipole values from a comma-separated string."""
    return [int(item) for item in value.split(",") if item]


def spectrum_reader(files):
    """Read one jaxpower power spectrum measurement."""
    from jaxpower import read

    return single_file_reader(files, read)


def make_spectrum_postprocess(kmin: float, kmax: float, rebin: int, ells: Sequence[int]):
    """Build a postprocessor for jaxpower power spectrum measurements."""

    def postprocess(data):
        rows = []
        k = None
        for measurement in data:
            measurement = measurement.select(k=slice(0, None, rebin)).select(
                k=(kmin, kmax)
            )
            poles = [measurement.get(ell) for ell in ells]
            k = poles[0].coords("k")
            rows.append(np.concatenate(poles).reshape(len(ells), -1))
        return np.asarray(rows), {"ells": list(ells), "k": k}

    return postprocess


def build_spec(args: argparse.Namespace) -> CompressionSpec:
    """Build the compression spec for power spectrum measurements."""
    from acm.observables.emc.spectrum_module import GalaxyPowerSpectrumMultipoles

    ells = parse_ells(args.ells)
    postprocess = make_spectrum_postprocess(args.kmin, args.kmax, args.rebin, ells)
    return CompressionSpec(
        stat_name="spectrum",
        observable_cls=GalaxyPowerSpectrumMultipoles,
        base_root="base/spectrum",
        base_pattern=(
            "c{cosmo_idx}_ph{phase_idx}/seed{seed}/"
            "mesh2_spectrum_poles_*_hod{hod_idx}.h5"
        ),
        reader=spectrum_reader,
        postprocess=postprocess,
        covariance_root="small/spectrum",
        covariance_pattern="mesh2_spectrum_poles_ph{phase_idx}.h5",
        covariance_reader=spectrum_reader,
        covariance_postprocess=postprocess,
    )


def main() -> None:
    """Run power spectrum compression from the command line."""
    run_cli(
        build_spec,
        description="Compress EMC power spectrum measurements.",
        add_arguments=add_arguments,
    )


if __name__ == "__main__":
    main()
