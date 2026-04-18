import numpy as np
import pytest

try:
    from cosmoprimo.fiducial import AbacusSummit

    from acm.hod.cutsky import BaseCutskyCatalog
except ImportError as exc:  # pragma: no cover - optional dependency guard
    pytest.skip(f"cutsky optional dependencies not available: {exc}", allow_module_level=True)


class _DummyMPIComm:
    rank = 0


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None


class DummyCutskyCatalog(BaseCutskyCatalog):
    def __init__(self, z):
        self.catalog = {"Z": np.array(z, dtype=float)}
        self.cosmo = AbacusSummit(0)
        self.logger = _DummyLogger()
        self.mpicomm = _DummyMPIComm()
        self.mpiroot = 0
        self.sky_fraction = 1.0


def _write_nz_file(path, rows):
    np.savetxt(path, np.asarray(rows, dtype=float))


def test_apply_radial_mask_raises_when_target_exceeds_raw(tmp_path):
    catalog = DummyCutskyCatalog([0.101, 0.103, 0.105, 0.107, 0.109])
    original_z = catalog.catalog["Z"].copy()
    nz_path = tmp_path / "target_nz.txt"
    _write_nz_file(
        nz_path,
        [
            [0, 0.10, 0.11, 1.0e3],
        ],
    )

    with pytest.raises(ValueError, match="too sparse"):
        catalog.apply_radial_mask(str(nz_path))

    np.testing.assert_allclose(catalog.catalog["Z"], original_z)
    assert "NZ" not in catalog.catalog


def test_apply_radial_mask_shape_only_skips_density_guard(tmp_path):
    catalog = DummyCutskyCatalog([0.101, 0.103, 0.105, 0.107, 0.109])
    nz_path = tmp_path / "target_nz.txt"
    _write_nz_file(
        nz_path,
        [
            [0, 0.10, 0.11, 1.0e3],
        ],
    )

    catalog.apply_radial_mask(str(nz_path), shape_only=True)

    assert "NZ" in catalog.catalog
    assert len(catalog.catalog["Z"]) <= 5
