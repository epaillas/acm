#!/usr/bin/env python3
"""Create measurement symlinks for HOD indices available in a target HOD tree.

This script mirrors source measurement files into a target directory by creating
absolute symlinks for the HOD indices present in the selected HOD catalog tree.
It is path-driven so it can be reused across statistics with different filename
conventions or extra directory layers.

Examples
--------
Create bispectrum links for all available cosmologies and seeds:

    python scripts/emc/measurements/create_measurement_symlinks.py \
        --source-root /global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/base/bispectrum \
        --target-root /global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.3/abacus/base/bispectrum \
        --hod-root /pscratch/sd/n/ntbfin/emulator/hods/v1.3/z0.5/yuan23_prior

Dry-run a single cosmology/seed:

    python scripts/emc/measurements/create_measurement_symlinks.py \
        --source-root .../v1.2/abacus/base/bispectrum \
        --target-root .../v1.3/abacus/base/bispectrum \
        --hod-root .../hods/v1.3/z0.5/yuan23_prior \
        --cosmo 0 --phase 0 --seed 0 --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Sequence

HOD_TOKEN_RE = re.compile(r"hod(\d+)")
COSMO_PHASE_RE = re.compile(r"c(?P<cosmo>\d+)_ph(?P<phase>\d+)$")
SEED_RE = re.compile(r"seed(?P<seed>\d+)$")


class ConflictError(RuntimeError):
    """Raised when an existing target conflicts with the requested symlink."""


@dataclass(frozen=True, order=True)
class ContextKey:
    """Uniquely identifies a cosmology/phase/seed combination."""

    cosmo: int
    phase: int
    seed: int

    @classmethod
    def from_relative_dir(cls, relative_dir: Path) -> "ContextKey | None":
        if len(relative_dir.parts) < 2:
            return None

        cosmo_phase_match = COSMO_PHASE_RE.fullmatch(relative_dir.parts[-2])
        seed_match = SEED_RE.fullmatch(relative_dir.parts[-1])
        if not cosmo_phase_match or not seed_match:
            return None

        return cls(
            cosmo=int(cosmo_phase_match.group("cosmo")),
            phase=int(cosmo_phase_match.group("phase")),
            seed=int(seed_match.group("seed")),
        )

    @property
    def suffix(self) -> Path:
        return Path(f"c{self.cosmo:03d}_ph{self.phase:03d}") / f"seed{self.seed}"


@dataclass
class RunSummary:
    hod_contexts: int = 0
    source_directories: int = 0
    matched_directories: int = 0
    created_links: int = 0
    skipped_existing: int = 0
    missing_source_hods: int = 0
    missing_source_directories: int = 0
    warnings: list[str] = field(default_factory=list)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create absolute symlinks under a target measurement tree for the "
            "HOD indices present in a reference HOD directory."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Source measurement root, e.g. the v1.2 statistic directory.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="Target measurement root, e.g. the v1.3 statistic directory.",
    )
    parser.add_argument(
        "--hod-root",
        type=Path,
        required=True,
        help="HOD root whose hodNNN.fits files define the allowed HOD indices.",
    )
    parser.add_argument(
        "--include-glob",
        action="append",
        default=[],
        help=(
            "Optional filename or relative-path glob to restrict source files. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--cosmo",
        type=int,
        nargs="+",
        help="Optional list of cosmology indices to process.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        nargs="+",
        help="Optional list of phase indices to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        help="Optional list of seed indices to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be linked without writing to disk.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-link actions and warnings.",
    )
    return parser.parse_args(argv)


def extract_hod_index(filename: str) -> int | None:
    match = HOD_TOKEN_RE.search(filename)
    if match is None:
        return None
    return int(match.group(1))


def matches_patterns(relative_path: Path, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True

    path_str = relative_path.as_posix()
    file_name = relative_path.name
    return any(
        fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(file_name, pattern)
        for pattern in patterns
    )


def context_selected(
    context: ContextKey,
    cosmo_filter: set[int] | None,
    phase_filter: set[int] | None,
    seed_filter: set[int] | None,
) -> bool:
    if cosmo_filter is not None and context.cosmo not in cosmo_filter:
        return False
    if phase_filter is not None and context.phase not in phase_filter:
        return False
    if seed_filter is not None and context.seed not in seed_filter:
        return False
    return True


def build_hod_index(
    hod_root: Path,
    cosmo_filter: set[int] | None = None,
    phase_filter: set[int] | None = None,
    seed_filter: set[int] | None = None,
) -> dict[ContextKey, set[int]]:
    index: DefaultDict[ContextKey, set[int]] = defaultdict(set)

    for hod_file in hod_root.rglob("hod*.fits"):
        if not hod_file.is_file():
            continue

        relative_parent = hod_file.relative_to(hod_root).parent
        context = ContextKey.from_relative_dir(relative_parent)
        if context is None or not context_selected(
            context, cosmo_filter, phase_filter, seed_filter
        ):
            continue

        hod_index = extract_hod_index(hod_file.name)
        if hod_index is None:
            continue
        index[context].add(hod_index)

    return dict(index)


def build_source_index(
    source_root: Path,
    include_globs: Sequence[str],
    cosmo_filter: set[int] | None = None,
    phase_filter: set[int] | None = None,
    seed_filter: set[int] | None = None,
) -> dict[Path, dict[int, list[Path]]]:
    index: DefaultDict[Path, DefaultDict[int, list[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # Measurements are stored directly inside seed directories, possibly
    # underneath extra statistic-specific path components.
    for source_dir in source_root.rglob("seed*"):
        if not source_dir.is_dir():
            continue

        relative_dir = source_dir.relative_to(source_root)
        context = ContextKey.from_relative_dir(relative_dir)
        if context is None or not context_selected(
            context, cosmo_filter, phase_filter, seed_filter
        ):
            continue

        for source_file in source_dir.iterdir():
            if not source_file.is_file():
                continue

            relative_path = source_file.relative_to(source_root)
            if not matches_patterns(relative_path, include_globs):
                continue

            hod_index = extract_hod_index(source_file.name)
            if hod_index is None:
                continue
            index[relative_dir][hod_index].append(relative_path)

    return {
        source_dir: {
            hod_index: sorted(paths)
            for hod_index, paths in files_by_hod.items()
        }
        for source_dir, files_by_hod in index.items()
    }


def resolve_existing_target(target_path: Path) -> Path | None:
    if not target_path.is_symlink():
        return None

    linked_path = Path(os.readlink(target_path))
    if not linked_path.is_absolute():
        linked_path = target_path.parent / linked_path
    return linked_path.resolve()


def ensure_symlink(
    source_path: Path,
    target_path: Path,
    dry_run: bool = False,
) -> str:
    expected_target = source_path.resolve()

    if target_path.exists() or target_path.is_symlink():
        existing_target = resolve_existing_target(target_path)
        if existing_target == expected_target:
            return "skipped"

        raise ConflictError(
            f"Target already exists with a different value: {target_path}"
        )

    if dry_run:
        return "created"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.symlink_to(source_path.resolve())
    return "created"


def create_symlinks(
    source_root: Path,
    target_root: Path,
    hod_root: Path,
    include_globs: Sequence[str] | None = None,
    cosmo_filter: set[int] | None = None,
    phase_filter: set[int] | None = None,
    seed_filter: set[int] | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> RunSummary:
    include_globs = include_globs or []
    source_root = source_root.expanduser().resolve()
    target_root = target_root.expanduser().resolve()
    hod_root = hod_root.expanduser().resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if not hod_root.is_dir():
        raise FileNotFoundError(f"HOD root does not exist: {hod_root}")
    if target_root.exists() and not target_root.is_dir():
        raise NotADirectoryError(f"Target root is not a directory: {target_root}")

    hod_index = build_hod_index(
        hod_root,
        cosmo_filter=cosmo_filter,
        phase_filter=phase_filter,
        seed_filter=seed_filter,
    )
    source_index = build_source_index(
        source_root,
        include_globs=include_globs,
        cosmo_filter=cosmo_filter,
        phase_filter=phase_filter,
        seed_filter=seed_filter,
    )

    summary = RunSummary(
        hod_contexts=len(hod_index),
        source_directories=len(source_index),
    )

    source_contexts = {
        ContextKey.from_relative_dir(source_dir): source_dir for source_dir in source_index
    }
    for context in sorted(hod_index):
        if context not in source_contexts:
            summary.missing_source_directories += 1
            message = f"Missing source directory for {context.suffix.as_posix()}"
            summary.warnings.append(message)
            if verbose:
                print(f"WARNING: {message}", file=sys.stderr)

    for source_dir in sorted(source_index):
        context = ContextKey.from_relative_dir(source_dir)
        if context is None:
            continue

        allowed_hods = hod_index.get(context)
        if not allowed_hods:
            continue

        summary.matched_directories += 1
        files_by_hod = source_index[source_dir]

        for hod_id in sorted(allowed_hods):
            matching_files = files_by_hod.get(hod_id)
            if not matching_files:
                summary.missing_source_hods += 1
                message = (
                    f"Missing source files for hod{hod_id:03d} in "
                    f"{source_dir.as_posix()}"
                )
                summary.warnings.append(message)
                if verbose:
                    print(f"WARNING: {message}", file=sys.stderr)
                continue

            for relative_path in matching_files:
                source_path = source_root / relative_path
                target_path = target_root / relative_path
                status = ensure_symlink(
                    source_path=source_path,
                    target_path=target_path,
                    dry_run=dry_run,
                )
                if status == "created":
                    summary.created_links += 1
                elif status == "skipped":
                    summary.skipped_existing += 1

                if verbose:
                    action = "would link" if dry_run else status
                    print(
                        f"{action}: {target_path} -> {source_path.resolve()}",
                        file=sys.stderr,
                    )

    return summary


def format_summary(summary: RunSummary, dry_run: bool = False) -> str:
    prefix = "Dry-run summary" if dry_run else "Summary"
    lines = [
        f"{prefix}:",
        f"  HOD contexts scanned: {summary.hod_contexts}",
        f"  Source directories scanned: {summary.source_directories}",
        f"  Matched source directories: {summary.matched_directories}",
        f"  Links {'planned' if dry_run else 'created'}: {summary.created_links}",
        f"  Existing correct links skipped: {summary.skipped_existing}",
        f"  Missing source HOD matches: {summary.missing_source_hods}",
        f"  Missing source directories: {summary.missing_source_directories}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        summary = create_symlinks(
            source_root=args.source_root,
            target_root=args.target_root,
            hod_root=args.hod_root,
            include_globs=args.include_glob,
            cosmo_filter=set(args.cosmo) if args.cosmo else None,
            phase_filter=set(args.phase) if args.phase else None,
            seed_filter=set(args.seed) if args.seed else None,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except (ConflictError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(format_summary(summary, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
