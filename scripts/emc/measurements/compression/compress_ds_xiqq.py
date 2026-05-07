"""Compress EMC density-split quantile autocorrelation measurements."""

import argparse

from compress_density_split import build_density_split_spec, run_density_split_cli


def build_spec(args: argparse.Namespace):
    """Build the compression spec for ds_xiqq measurements."""
    from acm.observables.emc.density_split_module import (
        DensitySplitQuantileCorrelationFunctionMultipoles,
    )

    return build_density_split_spec(
        args,
        stat_name="ds_xiqq",
        measurement_root="dsc_xiqq",
        observable_cls=DensitySplitQuantileCorrelationFunctionMultipoles,
    )


def main() -> None:
    """Run ds_xiqq compression from the command line."""
    run_density_split_cli(
        build_spec,
        description="Compress EMC density-split quantile autocorrelation measurements.",
    )


if __name__ == "__main__":
    main()
