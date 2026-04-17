#!/usr/bin/env python3
"""Create a v1.3 compressed statistic file from a v1.2 compressed source file.

The source file is expected to be a compressed xarray-style dictionary stored at
`v1.2/abacus/base/<statistic>/<statistic>.npy`. The rows are assumed to follow
the first `hod_idx` sorted HOD files in the corresponding v1.2 HOD tree. This
script keeps only the rows whose actual HOD indices are also present in the
v1.3 HOD tree, then truncates every cosmology to the common minimum retained
count so the output remains compatible with the rectangular EMC compressed-file
format.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from acm.utils.xarray import dataset_from_dict, dataset_to_dict

DEFAULT_STATISTIC = "pdf"
DEFAULT_V12_BASE = Path(
    "/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/base"
)
DEFAULT_V13_COMPRESSED = Path(
    "/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.3/abacus/compressed"
)
DEFAULT_V12_HOD_ROOT = Path("/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior")
DEFAULT_V13_HOD_ROOT = Path(
    "/pscratch/sd/n/ntbfin/emulator/hods/v1.3/z0.5/yuan23_prior"
)

HOD_RE = re.compile(r"hod(\d+)")


@dataclass(frozen=True)
class ConversionSummary:
    statistic: str
    source_x_shape: tuple[int, ...]
    source_y_shape: tuple[int, ...] | None
    target_x_shape: tuple[int, ...]
    target_y_shape: tuple[int, ...] | None
    overlap_counts: dict[int, int]
    common_count: int
    target_path: Path | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a v1.3 compressed statistic file by subsetting the v1.2 "
            "compressed training set to HOD indices present in the v1.3 HOD tree."
        )
    )
    parser.add_argument(
        "--statistic",
        type=str,
        default=DEFAULT_STATISTIC,
        help="Statistic name, used to infer default source and target paths.",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        help="Optional explicit path to the source v1.2 compressed .npy file.",
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        help="Optional explicit path to the target v1.3 compressed .npy file.",
    )
    parser.add_argument(
        "--v12-base",
        type=Path,
        default=DEFAULT_V12_BASE,
        help="Base directory for v1.2 compressed source files.",
    )
    parser.add_argument(
        "--v13-compressed-dir",
        type=Path,
        default=DEFAULT_V13_COMPRESSED,
        help="Directory where the v1.3 compressed output will be written.",
    )
    parser.add_argument(
        "--v12-hod-root",
        type=Path,
        default=DEFAULT_V12_HOD_ROOT,
        help="Root directory for the v1.2 HOD files.",
    )
    parser.add_argument(
        "--v13-hod-root",
        type=Path,
        default=DEFAULT_V13_HOD_ROOT,
        help="Root directory for the v1.3 HOD files.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        help="Phase index used to read HOD files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed index used to read HOD files.",
    )
    parser.add_argument(
        "--n-hod-v12",
        type=int,
        help=(
            "Optional override for the number of source HOD rows. By default "
            "the script infers this from x.sizes['hod_idx']."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the conversion and print the target shapes without writing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target file if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-cosmology overlap counts in addition to the summary.",
    )
    return parser.parse_args(argv)


def default_source_path(statistic: str, v12_base: Path) -> Path:
    return v12_base / statistic / f"{statistic}.npy"


def default_target_path(statistic: str, v13_compressed_dir: Path) -> Path:
    return v13_compressed_dir / f"{statistic}.npy"


def extract_hod_index(path: Path) -> int:
    match = HOD_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse a HOD index from {path}")
    return int(match.group(1))


def get_hod_ids(
    hod_root: Path, cosmo_idx: int, phase: int, seed: int, limit: int | None = None
) -> list[int]:
    hod_dir = hod_root / f"c{cosmo_idx:03d}_ph{phase:03d}" / f"seed{seed}"
    if not hod_dir.is_dir():
        raise FileNotFoundError(f"Missing HOD directory: {hod_dir}")

    hod_files = sorted(hod_dir.glob("hod*.fits"))
    if limit is not None:
        if len(hod_files) < limit:
            raise ValueError(
                f"Expected at least {limit} HOD files in {hod_dir}, found {len(hod_files)}"
            )
        hod_files = hod_files[:limit]
    return [extract_hod_index(path) for path in hod_files]


def load_source_dataset(source_path: Path) -> xr.Dataset:
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source compressed file: {source_path}")
    return dataset_from_dict(np.load(source_path, allow_pickle=True).item())


def validate_source_dataset(dataset: xr.Dataset, n_hod_v12: int | None = None) -> int:
    if "x" not in dataset.data_vars:
        raise ValueError("Source dataset is missing required variable 'x'")

    x = dataset["x"]
    if len(x.dims) < 2 or x.dims[:2] != ("cosmo_idx", "hod_idx"):
        raise ValueError(f"Unexpected x dims: {x.dims}")

    inferred_n_hod = int(x.sizes["hod_idx"])
    if n_hod_v12 is None:
        n_hod_v12 = inferred_n_hod
    elif inferred_n_hod != n_hod_v12:
        raise ValueError(
            f"Source x hod_idx size ({inferred_n_hod}) does not match --n-hod-v12 ({n_hod_v12})"
        )

    for var_name, data_array in dataset.data_vars.items():
        dims = data_array.dims
        if "hod_idx" not in dims:
            continue
        if len(dims) < 2 or dims[:2] != ("cosmo_idx", "hod_idx"):
            raise ValueError(
                f"Variable '{var_name}' contains hod_idx in an unsupported layout: {dims}"
            )
        if data_array.sizes["cosmo_idx"] != x.sizes["cosmo_idx"]:
            raise ValueError(
                f"Variable '{var_name}' has a different number of cosmologies than x"
            )
        if data_array.sizes["hod_idx"] != n_hod_v12:
            raise ValueError(
                f"Variable '{var_name}' has hod_idx size {data_array.sizes['hod_idx']}, "
                f"expected {n_hod_v12}"
            )
        if not np.array_equal(
            data_array.coords["cosmo_idx"].values, x.coords["cosmo_idx"].values
        ):
            raise ValueError(
                f"Variable '{var_name}' has different cosmo_idx coordinates than x"
            )

    return n_hod_v12


def build_row_selection(
    dataset: xr.Dataset,
    v12_hod_root: Path,
    v13_hod_root: Path,
    phase: int = 0,
    seed: int = 0,
    n_hod_v12: int = 300,
) -> tuple[list[int], dict[int, list[int]], dict[int, int], int]:
    cosmo_values = [int(value) for value in dataset["x"].coords["cosmo_idx"].values]
    selected_rows: dict[int, list[int]] = {}
    overlap_counts: dict[int, int] = {}

    for cosmo_idx in cosmo_values:
        v12_hod_ids = get_hod_ids(
            v12_hod_root, cosmo_idx=cosmo_idx, phase=phase, seed=seed, limit=n_hod_v12
        )
        v13_hod_ids = set(
            get_hod_ids(v13_hod_root, cosmo_idx=cosmo_idx, phase=phase, seed=seed)
        )
        rows = [row for row, hod_id in enumerate(v12_hod_ids) if hod_id in v13_hod_ids]
        if not rows:
            raise ValueError(f"No overlapping HOD rows found for c{cosmo_idx:03d}")

        selected_rows[cosmo_idx] = rows
        overlap_counts[cosmo_idx] = len(rows)

    common_count = min(overlap_counts.values())
    return cosmo_values, selected_rows, overlap_counts, common_count


def subset_sample_variable(
    data_array: xr.DataArray,
    cosmo_values: list[int],
    selected_rows: dict[int, list[int]],
    common_count: int,
) -> xr.DataArray:
    trailing_dims = data_array.dims[2:]
    sliced = np.stack(
        [
            data_array.values[row, selected_rows[cosmo_idx][:common_count], ...]
            for row, cosmo_idx in enumerate(cosmo_values)
        ],
        axis=0,
    )
    coords = {
        "cosmo_idx": data_array.coords["cosmo_idx"].values,
        "hod_idx": np.arange(common_count),
    }
    for dim in trailing_dims:
        coords[dim] = data_array.coords[dim].values
    return xr.DataArray(
        data=sliced,
        dims=data_array.dims,
        coords=coords,
        attrs=data_array.attrs,
        name=data_array.name,
    )


def build_v13_dataset(
    source_dataset: xr.Dataset,
    statistic: str,
    v12_hod_root: Path,
    v13_hod_root: Path,
    phase: int = 0,
    seed: int = 0,
    n_hod_v12: int | None = None,
) -> tuple[xr.Dataset, ConversionSummary]:
    n_hod_v12 = validate_source_dataset(source_dataset, n_hod_v12=n_hod_v12)

    (
        cosmo_values,
        selected_rows,
        overlap_counts,
        common_count,
    ) = build_row_selection(
        dataset=source_dataset,
        v12_hod_root=v12_hod_root,
        v13_hod_root=v13_hod_root,
        phase=phase,
        seed=seed,
        n_hod_v12=n_hod_v12,
    )

    data_vars: dict[str, xr.DataArray] = {}
    for var_name, data_array in source_dataset.data_vars.items():
        if len(data_array.dims) >= 2 and data_array.dims[:2] == ("cosmo_idx", "hod_idx"):
            data_vars[var_name] = subset_sample_variable(
                data_array=data_array,
                cosmo_values=cosmo_values,
                selected_rows=selected_rows,
                common_count=common_count,
            )
        else:
            data_vars[var_name] = data_array.copy(deep=True)

    target_dataset = xr.Dataset(data_vars=data_vars)

    y = target_dataset.data_vars.get("y")
    summary = ConversionSummary(
        statistic=statistic,
        source_x_shape=tuple(source_dataset["x"].shape),
        source_y_shape=tuple(source_dataset["y"].shape) if "y" in source_dataset else None,
        target_x_shape=tuple(target_dataset["x"].shape),
        target_y_shape=tuple(y.shape) if y is not None else None,
        overlap_counts=overlap_counts,
        common_count=common_count,
        target_path=None,
    )
    return target_dataset, summary


def ensure_writable_target(target_path: Path, force: bool) -> None:
    if target_path.exists():
        if target_path.is_dir():
            raise IsADirectoryError(f"Target path is a directory: {target_path}")
        if not force:
            raise FileExistsError(
                f"Target file already exists: {target_path}. Use --force to overwrite."
            )


def save_dataset(dataset: xr.Dataset, target_path: Path, force: bool = False) -> None:
    ensure_writable_target(target_path, force=force)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_path, dataset_to_dict(dataset))


def format_summary(summary: ConversionSummary, verbose: bool = False) -> str:
    count_distribution = sorted(Counter(summary.overlap_counts.values()).items())
    overlap_min = min(summary.overlap_counts.values())
    overlap_max = max(summary.overlap_counts.values())

    lines = [
        "Summary:",
        f"  Statistic: {summary.statistic}",
        f"  Cosmologies: {len(summary.overlap_counts)}",
        f"  Source x shape: {summary.source_x_shape}",
    ]
    if summary.source_y_shape is not None:
        lines.append(f"  Source y shape: {summary.source_y_shape}")
    lines.extend(
        [
            f"  Overlap range: {overlap_min}..{overlap_max}",
            f"  Common retained HODs: {summary.common_count}",
            f"  Target x shape: {summary.target_x_shape}",
        ]
    )
    if summary.target_y_shape is not None:
        lines.append(f"  Target y shape: {summary.target_y_shape}")
    lines.append(f"  Overlap distribution: {count_distribution}")

    if verbose:
        for cosmo_idx in sorted(summary.overlap_counts):
            lines.append(
                f"  c{cosmo_idx:03d}: {summary.overlap_counts[cosmo_idx]} retained rows"
            )

    if summary.target_path:
        lines.append(f"  Target path: {summary.target_path}")
    return "\n".join(lines)


def convert_compressed_file(
    statistic: str,
    source_path: Path,
    target_path: Path,
    v12_hod_root: Path,
    v13_hod_root: Path,
    phase: int = 0,
    seed: int = 0,
    n_hod_v12: int | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> ConversionSummary:
    source_dataset = load_source_dataset(source_path)
    target_dataset, summary = build_v13_dataset(
        source_dataset=source_dataset,
        statistic=statistic,
        v12_hod_root=v12_hod_root,
        v13_hod_root=v13_hod_root,
        phase=phase,
        seed=seed,
        n_hod_v12=n_hod_v12,
    )
    summary = ConversionSummary(
        statistic=summary.statistic,
        source_x_shape=summary.source_x_shape,
        source_y_shape=summary.source_y_shape,
        target_x_shape=summary.target_x_shape,
        target_y_shape=summary.target_y_shape,
        overlap_counts=summary.overlap_counts,
        common_count=summary.common_count,
        target_path=target_path,
    )

    if not dry_run:
        save_dataset(target_dataset, target_path, force=force)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    source_path = args.source_path or default_source_path(args.statistic, args.v12_base)
    target_path = args.target_path or default_target_path(
        args.statistic, args.v13_compressed_dir
    )

    try:
        summary = convert_compressed_file(
            statistic=args.statistic,
            source_path=source_path,
            target_path=target_path,
            v12_hod_root=args.v12_hod_root,
            v13_hod_root=args.v13_hod_root,
            phase=args.phase,
            seed=args.seed,
            n_hod_v12=args.n_hod_v12,
            dry_run=args.dry_run,
            force=args.force,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        IsADirectoryError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("Dry-run summary:")
        print(format_summary(summary, verbose=args.verbose).replace("Summary:\n", "", 1))
    else:
        print(format_summary(summary, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
