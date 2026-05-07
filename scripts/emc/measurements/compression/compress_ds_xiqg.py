"""Compress EMC density-split quantile-galaxy correlation measurements."""

import argparse

from compress_density_split import build_density_split_spec, run_density_split_cli


def build_spec(args: argparse.Namespace):
    """Build the compression spec for ds_xiqg measurements."""
    from acm.observables.emc.density_split_module import (
        DensitySplitQuantileGalaxyCorrelationFunctionMultipoles,
    )

    return build_density_split_spec(
        args,
        stat_name="ds_xiqg",
        measurement_root="dsc_xiqg",
        observable_cls=DensitySplitQuantileGalaxyCorrelationFunctionMultipoles,
    )


def main() -> None:
    """Run ds_xiqg compression from the command line."""
    run_density_split_cli(
        build_spec,
        description="Compress EMC density-split quantile-galaxy measurements.",
    )


if __name__ == "__main__":
    main()
