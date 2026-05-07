"""Shared utilities for EMC measurement compression scripts."""

import argparse
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray

from acm import setup_logging
from acm.utils.default import cosmo_list
from acm.utils.paths import lookup_registry_path
from acm.utils.xarray import dataset_to_dict

logger = logging.getLogger(__name__)

TEST_FILTERS = {"cosmo_idx": [0, 1, 2, 3, 4, 13]}

Reader = Callable[[list[Path]], Any]
Postprocessor = Callable[[Sequence[Any]], tuple[np.ndarray, dict]]
BuildSpec = Callable[[argparse.Namespace], "CompressionSpec"]


@dataclass
class CompressionSpec:
    """Configuration for a single EMC statistic compression script."""

    stat_name: str
    observable_cls: type
    base_root: str
    base_pattern: str
    reader: Reader
    postprocess: Postprocessor
    base_ignore_index: list[str] = field(default_factory=list)
    covariance_root: str | None = None
    covariance_pattern: str | None = None
    covariance_reader: Reader | None = None
    covariance_postprocess: Postprocessor | None = None
    covariance_ignore_index: list[str] = field(default_factory=list)
    align_covariance_last_dim: bool = True


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    """Parse comma-separated integer values from CLI input."""
    if isinstance(value, str):
        return [int(item) for item in value.split(",") if item]
    return [int(item) for item in value]


def parse_str_list(value: str | Sequence[str]) -> list[str]:
    """Parse comma-separated string values from CLI input."""
    if isinstance(value, str):
        return [item for item in value.split(",") if item]
    return list(value)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common EMC compression CLI arguments."""
    parser.add_argument("--n_hod", type=int, default=250)
    parser.add_argument("--add_covariance", action="store_true")
    parser.add_argument("--save_to", type=str, default=None)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cosmos", type=parse_int_list, default=None)
    parser.add_argument("--no_test_split", action="store_true")


def single_file_reader(files: list[Path], loader: Callable[[Path], Any]) -> Any:
    """Read a group that is expected to contain exactly one file."""
    if len(files) != 1:
        raise ValueError(f"Expected exactly one file per group, got {len(files)}")
    return loader(files[0])


def filter_measurement_groups(
    groups: dict[tuple, list[Path]],
    index_arrays: dict[str, list],
    cosmos: Sequence[int],
    phase: int,
    seed: int,
) -> tuple[dict[tuple, list[Path]], dict[str, list]]:
    """Filter groups to the requested cosmologies, phase, and seed."""
    requested_cosmos = {f"{cosmo_idx:03d}" for cosmo_idx in cosmos}
    filtered_groups: dict[tuple, list[Path]] = {}
    filtered_index_arrays = {key: [] for key in index_arrays}

    for key, files in groups.items():
        key_dict = dict(key)
        if "cosmo_idx" in key_dict and key_dict["cosmo_idx"] not in requested_cosmos:
            continue
        if "phase_idx" in key_dict and key_dict["phase_idx"] != f"{phase:03d}":
            continue
        if "seed" in key_dict and key_dict["seed"] != str(seed):
            continue

        filtered_groups[key] = files
        for name, values in filtered_index_arrays.items():
            values.append(key_dict[name])

    return filtered_groups, filtered_index_arrays


def limit_hods(
    groups: dict[tuple, list[Path]],
    index_arrays: dict[str, list],
    n_hod: int,
    group_by: Sequence[str] = ("cosmo_idx", "phase_idx", "seed"),
    hod_key: str = "hod_idx",
) -> tuple[dict[tuple, list[Path]], dict[str, list]]:
    """Keep the first ``n_hod`` HOD groups within each sample subgroup."""
    if hod_key not in index_arrays:
        raise ValueError(f"Missing expected HOD index '{hod_key}'")

    counts: dict[tuple, int] = {}
    kept_groups: dict[tuple, list[Path]] = {}
    kept_index_arrays = {key: [] for key in index_arrays}

    for key, files in groups.items():
        key_dict = dict(key)
        group_key = tuple(key_dict[name] for name in group_by if name in key_dict)
        count = counts.get(group_key, 0)
        if count >= n_hod:
            continue

        counts[group_key] = count + 1
        kept_groups[key] = files
        for name, values in kept_index_arrays.items():
            values.append(key_dict[name])

    if not kept_groups:
        raise ValueError("No measurement groups matched the requested filters")

    short_groups = {key: count for key, count in counts.items() if count < n_hod}
    if short_groups:
        raise ValueError(
            f"Some {tuple(group_by)} groups have fewer than n_hod={n_hod} HODs: "
            f"{short_groups}"
        )

    return kept_groups, kept_index_arrays


def align_covariance_last_dim(
    covariance_y: xarray.DataArray, y: xarray.DataArray
) -> xarray.DataArray:
    """Align covariance's last coordinate to the compressed data coordinate."""
    covariance_last_dim = covariance_y.dims[-1]
    y_last_dim = y.dims[-1]
    covariance_coord = covariance_y.coords[covariance_last_dim]
    y_coord = y.coords[y_last_dim]

    if covariance_coord.shape != y_coord.shape:
        raise ValueError(
            "Cannot align covariance last dimension due to shape mismatch: "
            f"{covariance_coord.shape} vs {y_coord.shape}"
        )

    return covariance_y.assign_coords({covariance_last_dim: np.asarray(y_coord)})


def compress_dataset(
    spec: CompressionSpec,
    paths: dict[str, str],
    n_hod: int,
    add_covariance: bool,
    save_to: str | Path | None,
    cosmos: Sequence[int],
    phase: int,
    seed: int,
    test_filters: dict[str, list] | None,
) -> xarray.Dataset:
    """Compress one EMC statistic into an xarray dataset."""
    from acm.utils.compression import (
        collect_measurements,
        compress_measurements,
        split_test_set,
    )

    measurements_dir = Path(paths["measurements_dir"])
    base_groups, base_index_arrays = collect_measurements(
        root_dir=measurements_dir / spec.base_root,
        glob_pattern=spec.base_pattern,
        ignore_index=spec.base_ignore_index,
    )
    base_groups, base_index_arrays = filter_measurement_groups(
        base_groups,
        base_index_arrays,
        cosmos=cosmos,
        phase=phase,
        seed=seed,
    )
    base_groups, base_index_arrays = limit_hods(
        base_groups,
        base_index_arrays,
        n_hod=n_hod,
    )

    y = compress_measurements(
        groups=base_groups,
        index_arrays=base_index_arrays,
        reindex=["hod_idx"],
        reindex_group_by=["cosmo_idx", "phase_idx", "seed"],
        reader=spec.reader,
        postprocess=spec.postprocess,
    ).rename("y")
    x = spec.observable_cls.compress_x(
        paths=paths,
        cosmos=list(cosmos),
        n_hod=n_hod,
        phase=phase,
        seed=seed,
    )

    data_vars: dict[str, xarray.DataArray] = {"x": x, "y": y}
    if add_covariance:
        if spec.covariance_root is None or spec.covariance_pattern is None:
            raise ValueError(f"No covariance compression configured for {spec.stat_name}")
        cov_groups, cov_index_arrays = collect_measurements(
            root_dir=measurements_dir / spec.covariance_root,
            glob_pattern=spec.covariance_pattern,
            ignore_index=spec.covariance_ignore_index,
        )
        covariance_y = compress_measurements(
            groups=cov_groups,
            index_arrays=cov_index_arrays,
            reader=spec.covariance_reader or spec.reader,
            postprocess=spec.covariance_postprocess or spec.postprocess,
        ).rename("covariance_y")
        if spec.align_covariance_last_dim:
            covariance_y = align_covariance_last_dim(covariance_y, y)
        data_vars["covariance_y"] = covariance_y

    dataset = xarray.Dataset(data_vars=data_vars)
    if test_filters is not None:
        dataset = split_test_set(dataset, test_filters)

    if save_to is not None:
        save_to = Path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)
        save_fn = save_to / f"{spec.stat_name}.npy"
        payload = np.array(dataset_to_dict(dataset), dtype=object)
        np.save(save_fn, payload)
        logger.info("Saved compressed %s data to %s", spec.stat_name, save_fn)

    return dataset


def run_cli(
    build_spec: BuildSpec,
    description: str,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    """Parse CLI args, build a compression spec, and run compression."""
    parser = argparse.ArgumentParser(description=description)
    add_common_arguments(parser)
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args()

    setup_logging()
    paths = lookup_registry_path("projects.yaml", "emc")
    save_to = args.save_to if args.save_to is not None else paths["data_dir"]
    cosmos = args.cosmos if args.cosmos is not None else cosmo_list
    test_filters = None if args.no_test_split else TEST_FILTERS

    compress_dataset(
        spec=build_spec(args),
        paths=paths,
        n_hod=args.n_hod,
        add_covariance=args.add_covariance,
        save_to=save_to,
        cosmos=cosmos,
        phase=args.phase,
        seed=args.seed,
        test_filters=test_filters,
    )
