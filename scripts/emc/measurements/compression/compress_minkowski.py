"""Compress EMC Minkowski functional measurements."""

import argparse
from collections.abc import Mapping

import numpy as np

from _common import CompressionSpec, run_cli, single_file_reader

DEFAULT_THRESHOLD_INDEX = (
    "/pscratch/sd/e/epaillas/emc/Threshold_index_for_MFs_with_Rg5_7_10_15.npy"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Minkowski-specific CLI arguments."""
    parser.add_argument("--threshold_index", type=str, default=DEFAULT_THRESHOLD_INDEX)


def minkowski_reader(files):
    """Read one Minkowski measurement."""

    def load(path):
        return np.load(path, allow_pickle=True).item()

    return single_file_reader(files, load)


def make_minkowski_postprocess(threshold_index: Mapping[str, np.ndarray]):
    """Build a postprocessor for Minkowski measurements."""

    def postprocess(data):
        rows = []
        for measurement in data:
            functionals = []
            for radius in [5, 7, 10, 15]:
                smoothing_key = f"Rg{radius}"
                for functional_idx in range(4):
                    functionals.append(
                        measurement[smoothing_key][
                            threshold_index[f"Threshold_index_{smoothing_key}"][
                                functional_idx
                            ],
                            functional_idx,
                        ]
                        * (10 * radius) ** functional_idx
                    )
            rows.append(np.concatenate(functionals))
        data_out = np.asarray(rows)
        return data_out, {"bin_idx": np.arange(data_out.shape[-1])}

    return postprocess


def build_spec(args: argparse.Namespace) -> CompressionSpec:
    """Build the compression spec for Minkowski measurements."""
    from acm.observables.emc.minkowski_module import MinkowskiFunctionals

    threshold_index = np.load(args.threshold_index, allow_pickle=True).item()
    postprocess = make_minkowski_postprocess(threshold_index)
    return CompressionSpec(
        stat_name="minkowski",
        observable_cls=MinkowskiFunctionals,
        base_root="base/minkowski",
        base_pattern=(
            "c{cosmo_idx}_ph{phase_idx}/seed{seed}/"
            "minkowski_*_hod{hod_idx}.npy"
        ),
        reader=minkowski_reader,
        postprocess=postprocess,
        covariance_root="small/minkowski",
        covariance_pattern="minkowski_ph{phase_idx}.npy",
        covariance_reader=minkowski_reader,
        covariance_postprocess=postprocess,
    )


def main() -> None:
    """Run Minkowski compression from the command line."""
    run_cli(
        build_spec,
        description="Compress EMC Minkowski functional measurements.",
        add_arguments=add_arguments,
    )


if __name__ == "__main__":
    main()
