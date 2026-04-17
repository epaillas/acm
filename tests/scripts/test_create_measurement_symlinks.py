"""Tests for the measurement symlink bootstrap script."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_script_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "emc"
        / "measurements"
        / "create_measurement_symlinks.py"
    )
    spec = importlib.util.spec_from_file_location(
        "create_measurement_symlinks", script_path
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def symlink_module() -> ModuleType:
    return load_script_module()


def write_file(path: Path, contents: str = "data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def test_create_symlinks_mirrors_nested_paths(
    tmp_path: Path, symlink_module: ModuleType
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    hod_root = tmp_path / "hods"

    source_plain_hod006 = (
        source_root
        / "c000_ph000"
        / "seed0"
        / "mesh3_spectrum_poles_c000_hod006.h5"
    )
    source_plain_hod008 = (
        source_root
        / "c000_ph000"
        / "seed0"
        / "mesh3_spectrum_poles_c000_hod008.h5"
    )
    source_plain_hod999 = (
        source_root
        / "c000_ph000"
        / "seed0"
        / "mesh3_spectrum_poles_c000_hod999.h5"
    )
    nested_source_hod006 = (
        source_root
        / "J5_L3_q0.8_sigma0.4"
        / "c000_ph000"
        / "seed0"
        / "wst_c000_hod006.npy"
    )
    nested_source_hod999 = (
        source_root
        / "J5_L3_q0.8_sigma0.4"
        / "c000_ph000"
        / "seed0"
        / "wst_c000_hod999.npy"
    )

    for path in [
        source_plain_hod006,
        source_plain_hod008,
        source_plain_hod999,
        nested_source_hod006,
        nested_source_hod999,
    ]:
        write_file(path)

    write_file(hod_root / "c000_ph000" / "seed0" / "hod006.fits")
    write_file(hod_root / "c000_ph000" / "seed0" / "hod008.fits")

    existing_target = (
        target_root
        / "c000_ph000"
        / "seed0"
        / "mesh3_spectrum_poles_c000_hod008.h5"
    )
    existing_target.parent.mkdir(parents=True, exist_ok=True)
    existing_target.symlink_to(source_plain_hod008.resolve())

    summary = symlink_module.create_symlinks(
        source_root=source_root,
        target_root=target_root,
        hod_root=hod_root,
    )

    created_plain = (
        target_root
        / "c000_ph000"
        / "seed0"
        / "mesh3_spectrum_poles_c000_hod006.h5"
    )
    created_nested = (
        target_root
        / "J5_L3_q0.8_sigma0.4"
        / "c000_ph000"
        / "seed0"
        / "wst_c000_hod006.npy"
    )

    assert created_plain.is_symlink()
    assert created_nested.is_symlink()
    assert Path(os.readlink(created_plain)).is_absolute()
    assert Path(os.readlink(created_nested)).is_absolute()
    assert created_plain.resolve() == source_plain_hod006.resolve()
    assert created_nested.resolve() == nested_source_hod006.resolve()

    assert not (
        target_root
        / "c000_ph000"
        / "seed0"
        / "mesh3_spectrum_poles_c000_hod999.h5"
    ).exists()
    assert not (
        target_root
        / "J5_L3_q0.8_sigma0.4"
        / "c000_ph000"
        / "seed0"
        / "wst_c000_hod999.npy"
    ).exists()

    assert summary.created_links == 2
    assert summary.skipped_existing == 1
    assert summary.matched_directories == 2
    assert summary.missing_source_hods == 1
    assert summary.missing_source_directories == 0


def test_main_dry_run_respects_filters(
    tmp_path: Path, symlink_module: ModuleType
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    hod_root = tmp_path / "hods"

    write_file(
        source_root / "c001_ph000" / "seed1" / "minkowski_c001_hod001.npy"
    )
    write_file(
        source_root / "c002_ph000" / "seed1" / "minkowski_c002_hod001.npy"
    )
    write_file(hod_root / "c001_ph000" / "seed1" / "hod001.fits")
    write_file(hod_root / "c002_ph000" / "seed1" / "hod001.fits")

    return_code = symlink_module.main(
        [
            "--source-root",
            str(source_root),
            "--target-root",
            str(target_root),
            "--hod-root",
            str(hod_root),
            "--cosmo",
            "1",
            "--seed",
            "1",
            "--dry-run",
            "--include-glob",
            "*.npy",
        ]
    )

    assert return_code == 0
    assert not any(target_root.rglob("*"))


def test_create_symlinks_raises_on_conflicting_target(
    tmp_path: Path, symlink_module: ModuleType
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    hod_root = tmp_path / "hods"

    source_file = (
        source_root / "c002_ph000" / "seed0" / "minkowski_c002_hod001.npy"
    )
    target_file = (
        target_root / "c002_ph000" / "seed0" / "minkowski_c002_hod001.npy"
    )

    write_file(source_file)
    write_file(hod_root / "c002_ph000" / "seed0" / "hod001.fits")
    write_file(target_file, contents="conflict")

    with pytest.raises(symlink_module.ConflictError):
        symlink_module.create_symlinks(
            source_root=source_root,
            target_root=target_root,
            hod_root=hod_root,
        )
