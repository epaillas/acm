"""Compress EMC wavelet scattering transform measurements."""

import argparse
from pathlib import Path

import numpy as np

from _common import CompressionSpec, parse_str_list, run_cli

DEFAULT_CONFIGS = "J4_L4_q1_sigma0.8,J4_L4_q1_sigma1.0,J5_L3_q0.8_sigma0.4"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add WST-specific CLI arguments."""
    parser.add_argument("--configs", type=parse_str_list, default=parse_str_list(DEFAULT_CONFIGS))


def find_config(path: Path, configs: list[str]) -> str:
    """Find the WST config name embedded in a measurement path."""
    for part in path.parts:
        if part in configs:
            return part
    raise ValueError(f"Could not infer WST config from {path}")


def make_wst_reader(configs: list[str]):
    """Build a reader that concatenates WST measurements across configs."""
    from acm.observables.emc.wst_module import WaveletScatteringTransform

    def reader(files):
        by_config = {find_config(path, configs): path for path in files}
        missing = [config for config in configs if config not in by_config]
        if missing:
            raise ValueError(f"Missing WST config files for {missing}")

        coeffs = []
        for config in configs:
            data = np.load(by_config[config], allow_pickle=True)
            coeffs.append(WaveletScatteringTransform.renorm_wst(data, config=config)[1:])
        return np.concatenate(coeffs)

    return reader


def wst_postprocess(data):
    """Convert concatenated WST coefficients to an array plus coordinates."""
    data_out = np.asarray(data)
    return data_out, {"bin_idx": np.arange(data_out.shape[-1])}


def build_spec(args: argparse.Namespace) -> CompressionSpec:
    """Build the compression spec for WST measurements."""
    from acm.observables.emc.wst_module import WaveletScatteringTransform

    configs = parse_str_list(args.configs)
    reader = make_wst_reader(configs)
    return CompressionSpec(
        stat_name="wst",
        observable_cls=WaveletScatteringTransform,
        base_root="base/wst",
        base_pattern=(
            "{config}/c{cosmo_idx}_ph{phase_idx}/seed{seed}/"
            "wst_*_hod{hod_idx}.npy"
        ),
        base_ignore_index=["config"],
        reader=reader,
        postprocess=wst_postprocess,
        covariance_root="small/wst",
        covariance_pattern="{config}/wst_ph{phase_idx}.npy",
        covariance_ignore_index=["config"],
        covariance_reader=reader,
        covariance_postprocess=wst_postprocess,
    )


def main() -> None:
    """Run WST compression from the command line."""
    run_cli(
        build_spec,
        description="Compress EMC WST measurements.",
        add_arguments=add_arguments,
    )


if __name__ == "__main__":
    main()
