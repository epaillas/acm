import pytest
import numpy as np
import pandas as pd

from acm.catalogs.products.snapshot import (
    _apply_rsd,
    _apply_ap,
    _apply_downsample,
)


class TestApplyRsd:
    def test_shifts_los_column(self):
        """RSD transform should shift the los column according to the formula z' = z + vz / (H * az)."""
        data = pd.DataFrame({"z": [0.0], "vz": [100.0]})
        result = _apply_rsd(data, los="z", hubble=100.0, az=0.5)
        assert result["z"].iloc[0] == pytest.approx(0.0 + 100.0 / (100.0 * 0.5))

    def test_does_not_mutate_input(self):
        """RSD transform should not mutate the input DataFrame."""
        data = pd.DataFrame({"z": [0.0], "vz": [100.0]})
        original = data.copy()
        _apply_rsd(data, los="z", hubble=100.0, az=0.5)
        pd.testing.assert_frame_equal(data, original)

    def test_only_los_column_modified(self):
        """RSD transform should only modify the los column, leaving other columns unchanged."""
        data = pd.DataFrame({"x": [1.0], "z": [0.0], "vz": [100.0]})
        result = _apply_rsd(data, los="z", hubble=100.0, az=0.5)
        assert result["x"].iloc[0] == pytest.approx(1.0)


class TestApplyAp:
    def test_scales_los_by_qpar(self):
        """AP transform should scale the los column by q_par."""
        data = pd.DataFrame({"x": [1.0], "y": [1.0], "z": [1.0]})
        result = _apply_ap(data, los="z", q_par=1.2, q_perp=0.9, pos_columns=("x", "y", "z"))
        assert result["z"].iloc[0] == pytest.approx(1.2)

    def test_scales_transverse_by_qperp(self):
        """AP transform should scale the transverse columns by q_perp."""
        data = pd.DataFrame({"x": [1.0], "y": [1.0], "z": [1.0]})
        result = _apply_ap(data, los="z", q_par=1.2, q_perp=0.9, pos_columns=("x", "y", "z"))
        assert result["x"].iloc[0] == pytest.approx(0.9)
        assert result["y"].iloc[0] == pytest.approx(0.9)

    def test_does_not_mutate_input(self):
        """AP transform should not mutate the input DataFrame."""
        data = pd.DataFrame({"x": [1.0], "y": [1.0], "z": [1.0]})
        original = data.copy()
        _apply_ap(data, los="z", q_par=1.2, q_perp=0.9, pos_columns=("x", "y", "z"))
        pd.testing.assert_frame_equal(data, original)


class TestApplyDownsample:
    @pytest.fixture
    def data(self):
        return pd.DataFrame({"x": np.arange(100, dtype=float)})

    def test_by_ngal(self, data):
        """Downsampling by n_gal should reduce the number of galaxies to the target."""
        result = _apply_downsample(data, tracer="FOO", n_gal=50, f_gal=None, nbar=None)
        assert len(result) == 50

    def test_by_fgal(self, data):
        """Downsampling by f_gal should reduce the number of galaxies to the target."""
        result = _apply_downsample(data, tracer="FOO", n_gal=None, f_gal=0.5, nbar=None)
        assert len(result) == 50

    def test_by_nbar(self, data):
        """Downsampling by nbar should reduce the number of galaxies to the target, using volume to compute current nbar."""
        volume = lambda: np.prod([10., 10., 10.])
        target_nbar = 50 / 1000.0
        result = _apply_downsample(data, tracer="FOO", n_gal=None, f_gal=None, nbar=target_nbar, volume=volume)
        assert len(result) == 50

    def test_nbar_without_volume_raises(self, data):
        """Downsampling by nbar without providing volume should raise an error."""
        with pytest.raises(ValueError, match="volume"):
            _apply_downsample(data, tracer="FOO", n_gal=None, f_gal=None, nbar=0.1)

    def test_multiple_params_raises(self, data):
        """Specifying multiple downsampling parameters should raise an error."""
        with pytest.raises(ValueError, match="Exactly one"):
            _apply_downsample(data, tracer="FOO", n_gal=50, f_gal=0.5, nbar=None)

    def test_target_geq_current_returns_unchanged(self, data):
        """Downsampling should be skipped (returning unchanged data) if the target number density is greater than or equal to the current number density."""
        result = _apply_downsample(data, tracer="FOO", n_gal=200, f_gal=None, nbar=None)
        assert len(result) == 100
        
    def test_random_seed_reproducibility(self, data):
        """Downsampling with a fixed random seed should produce the same result across multiple calls."""
        result1 = _apply_downsample(data, tracer="FOO", n_gal=50, f_gal=None, nbar=None, seed=42)
        result2 = _apply_downsample(data, tracer="FOO", n_gal=50, f_gal=None, nbar=None, seed=42)
        pd.testing.assert_frame_equal(result1.reset_index(drop=True), result2.reset_index(drop=True))
        
    def test_random_seed_different_seeds(self, data):
        """Downsampling with different random seeds should produce different results."""
        result1 = _apply_downsample(data, tracer="FOO", n_gal=50, f_gal=None, nbar=None, seed=42)
        result2 = _apply_downsample(data, tracer="FOO", n_gal=50, f_gal=None, nbar=None, seed=43)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(result1.reset_index(drop=True), result2.reset_index(drop=True))
