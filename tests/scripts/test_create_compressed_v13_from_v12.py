"""Tests for the generic v1.3 compressed-file generation helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import xarray as xr

from acm.utils.xarray import dataset_to_dict


def load_script_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "emc"
        / "measurements"
        / "create_compressed_v13_from_v12.py"
    )
    spec = importlib.util.spec_from_file_location(
        "create_compressed_v13_from_v12", script_path
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def compressed_module() -> ModuleType:
    return load_script_module()


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data")


def build_source_dataset() -> xr.Dataset:
    x_data = np.array(
        [
            [[0, 0], [0, 1], [0, 2], [0, 3]],
            [[1, 0], [1, 1], [1, 2], [1, 3]],
        ]
    )
    y_data = np.array(
        [
            [
                [[100, 101, 102], [103, 104, 105]],
                [[110, 111, 112], [113, 114, 115]],
                [[120, 121, 122], [123, 124, 125]],
                [[130, 131, 132], [133, 134, 135]],
            ],
            [
                [[200, 201, 202], [203, 204, 205]],
                [[210, 211, 212], [213, 214, 215]],
                [[220, 221, 222], [223, 224, 225]],
                [[230, 231, 232], [233, 234, 235]],
            ],
        ]
    )
    covariance_y = np.arange(15).reshape(5, 3)

    return xr.Dataset(
        data_vars={
            "x": xr.DataArray(
                data=x_data,
                dims=("cosmo_idx", "hod_idx", "parameters"),
                coords={
                    "cosmo_idx": np.array([0, 1]),
                    "hod_idx": np.arange(4),
                    "parameters": np.array(["p0", "p1"]),
                },
                attrs={"sample": ["cosmo_idx", "hod_idx"], "features": ["parameters"]},
                name="x",
            ),
            "y": xr.DataArray(
                data=y_data,
                dims=("cosmo_idx", "hod_idx", "multipoles", "k"),
                coords={
                    "cosmo_idx": np.array([0, 1]),
                    "hod_idx": np.arange(4),
                    "multipoles": np.array([0, 2]),
                    "k": np.arange(3),
                },
                attrs={"sample": ["cosmo_idx", "hod_idx"], "features": ["multipoles", "k"]},
                name="y",
            ),
            "covariance_y": xr.DataArray(
                data=covariance_y,
                dims=("phase_idx", "bin_idx"),
                coords={"phase_idx": np.arange(5), "bin_idx": np.arange(3)},
                attrs={"sample": ["phase_idx"], "features": ["bin_idx"]},
                name="covariance_y",
            ),
        }
    )


def setup_fake_inputs(tmp_path: Path) -> tuple[Path, Path, Path, xr.Dataset]:
    source_dataset = build_source_dataset()
    source_path = tmp_path / "source" / "pdf.npy"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(source_path, dataset_to_dict(source_dataset))

    v12_root = tmp_path / "hods_v12"
    v13_root = tmp_path / "hods_v13"

    v12_rows = {
        0: [1, 5, 10, 20],
        1: [2, 4, 6, 8],
    }
    v13_rows = {
        0: [5, 20],
        1: [4, 6, 8],
    }

    for cosmo_idx, hod_ids in v12_rows.items():
        for hod_id in hod_ids:
            touch(
                v12_root / f"c{cosmo_idx:03d}_ph000" / "seed0" / f"hod{hod_id:03d}.fits"
            )

    for cosmo_idx, hod_ids in v13_rows.items():
        for hod_id in hod_ids:
            touch(
                v13_root / f"c{cosmo_idx:03d}_ph000" / "seed0" / f"hod{hod_id:03d}.fits"
            )

    return source_path, v12_root, v13_root, source_dataset


def test_build_v13_dataset_truncates_and_preserves_trailing_dims(
    tmp_path: Path, compressed_module: ModuleType
) -> None:
    source_path, v12_root, v13_root, source_dataset = setup_fake_inputs(tmp_path)
    loaded = compressed_module.load_source_dataset(source_path)

    output_dataset, summary = compressed_module.build_v13_dataset(
        source_dataset=loaded,
        statistic="pdf",
        v12_hod_root=v12_root,
        v13_hod_root=v13_root,
        phase=0,
        seed=0,
        n_hod_v12=None,
    )

    expected_x = np.array(
        [
            [[0, 1], [0, 3]],
            [[1, 1], [1, 2]],
        ]
    )
    expected_y = np.array(
        [
            [
                [[110, 111, 112], [113, 114, 115]],
                [[130, 131, 132], [133, 134, 135]],
            ],
            [
                [[210, 211, 212], [213, 214, 215]],
                [[220, 221, 222], [223, 224, 225]],
            ],
        ]
    )

    assert summary.statistic == "pdf"
    assert summary.overlap_counts == {0: 2, 1: 3}
    assert summary.common_count == 2
    assert summary.target_x_shape == (2, 2, 2)
    assert summary.target_y_shape == (2, 2, 2, 3)
    assert np.array_equal(output_dataset["x"].values, expected_x)
    assert np.array_equal(output_dataset["y"].values, expected_y)
    assert output_dataset["y"].dims == ("cosmo_idx", "hod_idx", "multipoles", "k")
    assert np.array_equal(
        output_dataset["covariance_y"].values, source_dataset["covariance_y"].values
    )
    assert np.array_equal(output_dataset["x"].coords["hod_idx"].values, np.arange(2))


def test_main_dry_run_writes_nothing(
    tmp_path: Path, compressed_module: ModuleType
) -> None:
    source_path, v12_root, v13_root, _ = setup_fake_inputs(tmp_path)
    target_path = tmp_path / "target" / "pdf.npy"

    return_code = compressed_module.main(
        [
            "--statistic",
            "pdf",
            "--source-path",
            str(source_path),
            "--target-path",
            str(target_path),
            "--v12-hod-root",
            str(v12_root),
            "--v13-hod-root",
            str(v13_root),
            "--dry-run",
        ]
    )

    assert return_code == 0
    assert not target_path.exists()


def test_convert_compressed_file_requires_force_to_overwrite(
    tmp_path: Path, compressed_module: ModuleType
) -> None:
    source_path, v12_root, v13_root, _ = setup_fake_inputs(tmp_path)
    target_path = tmp_path / "target" / "pdf.npy"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("existing")

    with pytest.raises(FileExistsError):
        compressed_module.convert_compressed_file(
            statistic="pdf",
            source_path=source_path,
            target_path=target_path,
            v12_hod_root=v12_root,
            v13_hod_root=v13_root,
            phase=0,
            seed=0,
            n_hod_v12=None,
            force=False,
        )

    summary = compressed_module.convert_compressed_file(
        statistic="pdf",
        source_path=source_path,
        target_path=target_path,
        v12_hod_root=v12_root,
        v13_hod_root=v13_root,
        phase=0,
        seed=0,
        n_hod_v12=None,
        force=True,
    )

    output_dict = np.load(target_path, allow_pickle=True).item()
    assert summary.statistic == "pdf"
    assert summary.target_x_shape == (2, 2, 2)
    assert output_dict["x"]["data"].shape == (2, 2, 2)
