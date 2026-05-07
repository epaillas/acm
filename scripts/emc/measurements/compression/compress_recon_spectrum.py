"""Compress EMC reconstructed power spectrum measurements."""

import argparse

from compress_spectrum import (
    add_arguments,
    make_spectrum_postprocess,
    parse_ells,
    spectrum_reader,
)
from _common import CompressionSpec, run_cli


def build_spec(args: argparse.Namespace) -> CompressionSpec:
    """Build the compression spec for reconstructed power spectrum measurements."""
    from acm.observables.emc.recon_spectrum_module import (
        ReconstructedGalaxyPowerSpectrumMultipoles,
    )

    ells = parse_ells(args.ells)
    postprocess = make_spectrum_postprocess(args.kmin, args.kmax, args.rebin, ells)
    return CompressionSpec(
        stat_name="recon_spectrum",
        observable_cls=ReconstructedGalaxyPowerSpectrumMultipoles,
        base_root="base/recon_spectrum",
        base_pattern=(
            "c{cosmo_idx}_ph{phase_idx}/seed{seed}/"
            "mesh2_recon_spectrum_poles_*_hod{hod_idx}.h5"
        ),
        reader=spectrum_reader,
        postprocess=postprocess,
        covariance_root="small/recon_spectrum",
        covariance_pattern="mesh2_recon_spectrum_poles_ph{phase_idx}.h5",
        covariance_reader=spectrum_reader,
        covariance_postprocess=postprocess,
    )


def main() -> None:
    """Run reconstructed power spectrum compression from the command line."""
    run_cli(
        build_spec,
        description="Compress EMC reconstructed power spectrum measurements.",
        add_arguments=add_arguments,
    )


if __name__ == "__main__":
    main()
