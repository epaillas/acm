import numpy as np
import pandas as pd
import pytest

from acm.catalogs.dataclasses import Tracer
from acm.catalogs.products.snapshot import (
    RandomSnapshotCatalog,
    SnapshotCatalog,
    boundary_check,
)

# ruff: noqa: ANN001, ANN201, D103, INP001, S101

#%% Fixtures

def make_tracer_data(n: int = 100) -> pd.DataFrame:
    """Generate minimal valid tracer data."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "x": rng.uniform(0, 500, n),
        "y": rng.uniform(0, 500, n),
        "z": rng.uniform(0, 500, n),
        "vx": rng.normal(0, 1, n),
        "vy": rng.normal(0, 1, n),
        "vz": rng.normal(0, 1, n),
    })

@pytest.fixture
def boxsize():
    return 500.0

@pytest.fixture
def catalog(cosmo_mock1, cosmo_mock2, boxsize):
    return SnapshotCatalog(redshift=0.5, cosmo=cosmo_mock1, cosmo_fid=cosmo_mock2, boxsize=boxsize)

@pytest.fixture
def tracer():
    return Tracer(name="FOO", params={"a": 1})

@pytest.fixture
def valid_data():
    """Minimal valid DataFrame with required position and velocity columns."""
    return make_tracer_data()

@pytest.fixture
def populated_catalog(catalog, tracer, valid_data):
    catalog.set_tracer_data(tracer, valid_data)
    return catalog


#%% Testing SnapshotCatalog construction

def test_redshift_stored(catalog):
    assert catalog.redshift == 0.5

def test_boxsize_scalar_broadcast(catalog):
    """Scalar boxsize should be broadcast to all three dimensions."""
    np.testing.assert_array_equal(catalog._boxsize, [500.0, 500.0, 500.0])

def test_boxsize_array(cosmo_mock1, cosmo_mock2):
    """Array boxsize should be stored as-is."""
    cat = SnapshotCatalog(redshift=0.5, cosmo=cosmo_mock1, cosmo_fid=cosmo_mock2, boxsize=[100., 200., 300.])
    np.testing.assert_array_equal(cat._boxsize, [100., 200., 300.])

def test_boxsize_invalid_shape_raises(cosmo_mock1, cosmo_mock2):
    """Boxsize with invalid shape should raise an error."""
    with pytest.raises(ValueError, match="shape"):
        SnapshotCatalog(redshift=0.5, cosmo=cosmo_mock1, cosmo_fid=cosmo_mock2, boxsize=[1., 2.])

def test_repr(populated_catalog):
    """Repr should include class name, redshift, and tracer names."""
    r = repr(populated_catalog)
    assert "SnapshotCatalog" in r
    assert "0.5" in r
    assert "FOO" in r


#%% Cosmological properties

def test_az(catalog):
    assert catalog.az == pytest.approx(1.0 / 1.5)

def test_hubble(catalog):
    assert catalog.hubble == pytest.approx(130.0)

def test_hubble_fid(catalog):
    assert catalog.hubble_fid == pytest.approx(150.0)

def test_q_par(catalog):
    """q_par = hubble_fid / hubble; both cosmologies identical here so q_par = 1."""
    assert catalog.q_par == pytest.approx(150/130)

def test_q_perp(catalog, cosmo_mock1, cosmo_mock2):
    """q_perp = DA / DA_fid; both cosmologies identical here so q_perp = 1."""
    z = catalog.redshift
    qperp = cosmo_mock1.angular_diameter_distance(z) / cosmo_mock2.angular_diameter_distance(z)
    assert catalog.q_perp == pytest.approx(qperp)


#%% boxsize with AP

def test_boxsize_without_ap(catalog):
    """Without AP transform, boxsize should be unchanged."""
    np.testing.assert_array_equal(catalog.boxsize, [500., 500., 500.])

def test_boxsize_with_ap(populated_catalog):
    """AP transform should scale boxsize by q_par along los and q_perp transversely."""
    populated_catalog.ap(los="z")
    boxsize = populated_catalog.boxsize
    assert boxsize[2] == pytest.approx(500.0 / populated_catalog.q_par)
    assert boxsize[0] == pytest.approx(500.0 / populated_catalog.q_perp)


#%% _check_data_columns

def test_check_data_columns_valid(catalog, valid_data):
    assert catalog._check_data_columns(valid_data) is True

def test_check_data_columns_missing(catalog):
    """DataFrame missing required columns should fail the check."""
    bad = pd.DataFrame({"x": [1.], "y": [2.], "z": [3.]})  # missing velocities
    assert catalog._check_data_columns(bad) is False

def test_set_tracer_data_missing_columns_raises(catalog, tracer):
    """Setting tracer data with missing required columns should raise an error."""
    bad = pd.DataFrame({"x": [1.], "y": [2.]})
    with pytest.raises(ValueError, match="missing required columns"):
        catalog.set_tracer_data(tracer, bad)


#%% nbar

def test_nbar(populated_catalog):
    volume = 500.0 ** 3
    assert populated_catalog.nbar == pytest.approx(100 / volume)

def test_nbar_empty_raises(catalog):
    """Nbar property should raise if no tracers are registered."""
    with pytest.raises(RuntimeError, match="No tracers"):
        _ = catalog.nbar


#%% RSD transform

def test_rsd_adds_transform(populated_catalog):
    """RSD transform should be added to the pipeline when rsd() is called."""
    populated_catalog.rsd(los="z")
    assert "rsd" in populated_catalog._transforms

def test_rsd_invalid_los_raises(populated_catalog):
    """Invalid los should raise an error."""
    with pytest.raises(ValueError, match="los"):
        populated_catalog.rsd(los="w")

def rsd_formula(populated_catalog, raw):
    """RSD transform should shift positions along the los according to the formula z' = z + vz / (H * az)."""
    populated_catalog.rsd(los="z")
    result = populated_catalog.get_tracer_data("FOO")
    expected_z = raw["z"] + raw["vz"] / (populated_catalog.hubble * populated_catalog.az)
    pd.testing.assert_series_equal(result["z"], expected_z, check_names=False)

def test_rsd_with_wrap(populated_catalog, boxsize):
    """RSD with wrap should apply periodic wrapping after shifting."""
    raw = populated_catalog.get_tracer_data("FOO", raw=True).copy()
    populated_catalog.rsd(los="z", wrap=True)
    result = populated_catalog.get_tracer_data("FOO")
    expected_z = (raw["z"] + raw["vz"] / (populated_catalog.hubble * populated_catalog.az)) % boxsize
    pd.testing.assert_series_equal(result["z"], expected_z, check_names=False)

def test_rsd_with_offset(populated_catalog, boxsize):
    """RSD with wrap and offset should apply periodic wrapping with the offset correction."""
    raw = populated_catalog.get_tracer_data("FOO", raw=True).copy()
    populated_catalog.rsd(los="z", wrap=True, offset=boxsize/2)
    result = populated_catalog.get_tracer_data("FOO")
    shifted_z = raw["z"] + boxsize / 2
    expected_z = (shifted_z + raw["vz"] / (populated_catalog.hubble * populated_catalog.az)) % boxsize - boxsize / 2
    pd.testing.assert_series_equal(result["z"], expected_z, check_names=False)

def test_rsd_does_not_mutate_raw(populated_catalog):
    """RSD transform should not mutate the raw data stored in the catalog."""
    raw_before = populated_catalog.get_tracer_data("FOO", raw=True).copy()
    populated_catalog.rsd(los="z")
    populated_catalog.get_tracer_data("FOO")
    pd.testing.assert_frame_equal(populated_catalog._data["FOO"], raw_before)

def test_rsd_after_ap_warns(populated_catalog, caplog):
    with caplog.at_level("WARNING"):
        populated_catalog.ap(los="z")
        populated_catalog.rsd(los="z")
    assert "AP transform exists" in caplog.text


#%% AP transform

def test_ap_adds_transform(populated_catalog):
    """AP transform should be added to the pipeline when ap() is called."""
    populated_catalog.ap(los="z")
    assert "ap" in populated_catalog._transforms

def test_ap_invalid_los_raises(populated_catalog):
    """Invalid los should raise an error."""
    with pytest.raises(ValueError, match="los"):
        populated_catalog.ap(los="w")

def test_ap_scales_positions(populated_catalog):
    """AP transform should scale positions by q_par along the los and q_perp transversely."""
    raw = populated_catalog.get_tracer_data("FOO", raw=True).copy()
    populated_catalog.ap(los="z")
    result = populated_catalog.get_tracer_data("FOO")
    assert result["z"].values == pytest.approx(raw["z"].values / populated_catalog.q_par)
    assert result["x"].values == pytest.approx(raw["x"].values / populated_catalog.q_perp)


#%% Downsample transform

def test_downsample_adds_transform(populated_catalog):
    """Downsample transform should be added to the pipeline when downsample() is called."""
    populated_catalog.downsample("FOO", n_gal=50)
    assert "downsample_FOO" in populated_catalog._transforms

def test_downsample_multiple_params_raises(populated_catalog):
    """Specifying multiple downsampling parameters should raise an error."""
    with pytest.raises(ValueError, match="Exactly one"):
        populated_catalog.downsample("FOO", n_gal=50, f_gal=0.5)

def test_downsample_no_params_raises(populated_catalog):
    """Specifying no downsampling parameters should raise an error."""
    with pytest.raises(ValueError, match="Exactly one"):
        populated_catalog.downsample("FOO")

def test_downsample_by_ngal(populated_catalog):
    """Downsampling by n_gal should reduce the number of galaxies to the target."""
    populated_catalog.downsample("FOO", n_gal=50)
    assert len(populated_catalog.get_tracer_data("FOO")) == 50

def test_downsample_by_fgal(populated_catalog):
    """Downsampling by f_gal should reduce the number of galaxies to the target."""
    populated_catalog.downsample("FOO", f_gal=0.5)
    assert len(populated_catalog.get_tracer_data("FOO")) == 50

def test_downsample_by_nbar(populated_catalog):
    """Downsampling by nbar should reduce the number of galaxies to the target."""
    target_nbar = 50 / 500.0 ** 3
    populated_catalog.downsample("FOO", nbar=target_nbar)
    assert len(populated_catalog.get_tracer_data("FOO")) == 50

def test_downsample_nbar_uses_ap_boxsize(populated_catalog):
    """Nbar downsampling should use AP-scaled boxsize when AP is in the pipeline."""
    populated_catalog.ap(los="z")
    ap_volume = np.prod(populated_catalog.boxsize)
    target_nbar = 50 / ap_volume
    populated_catalog.downsample("FOO", nbar=target_nbar)
    assert len(populated_catalog.get_tracer_data("FOO")) == 50

def test_downsample_skips_if_target_geq_current(populated_catalog, caplog):
    """Downsampling should be skipped (with a warning) if the target number density is greater than or equal to the current number density."""
    with caplog.at_level("WARNING"):
        populated_catalog.downsample("FOO", n_gal=200)
        populated_catalog.get_tracer_data("FOO")
    assert "skipping" in caplog.text

#%% Positions tests

def test_positions_returns_dataframe(populated_catalog):
    """get_tracer_data with raw=False should return a DataFrame."""
    result = populated_catalog.positions()
    assert isinstance(result, pd.DataFrame)

def test_positions_columns_match_pos_columns(populated_catalog):
    """Position columns in the data should match the pos_columns specified in the catalog."""
    result = populated_catalog.positions()
    assert list(result.columns) == list(SnapshotCatalog.pos_columns)

def test_positions_no_velocity_columns(populated_catalog):
    """Position data should not include velocity columns."""
    result = populated_catalog.positions()
    assert not any(col in result.columns for col in SnapshotCatalog.vel_columns)

def test_positions_length_equals_ngal(populated_catalog):
    """Total rows should equal ngal."""
    assert len(populated_catalog.positions()) == populated_catalog.ngal

def test_empty_catalog_raises(catalog):
    """Calling positions on a catalog with no tracers should raise."""
    with pytest.raises(RuntimeError, match="No tracers"):
        catalog.positions()

#%% Serialization

def test_save_creates_file(populated_catalog, tmp_path):
    path = tmp_path / "snapshot.h5"
    populated_catalog.save(path)
    assert path.exists()

def test_save_load_roundtrip(populated_catalog, tmp_path):
    path = tmp_path / "snapshot.h5"
    populated_catalog.save(path)
    loaded = SnapshotCatalog.load(path)
    assert loaded.redshift == pytest.approx(0.5)
    np.testing.assert_array_equal(loaded._boxsize, populated_catalog._boxsize)

def test_save_load_tracer_data(populated_catalog, tmp_path, valid_data):
    """Test that tracer data is preserved through a save/load roundtrip, including column order."""
    path = tmp_path / "snapshot.h5"
    populated_catalog.save(path)
    loaded = SnapshotCatalog.load(path)
    pd.testing.assert_frame_equal(
        loaded.get_tracer_data("FOO", raw=True).reset_index(drop=True),
        valid_data.reset_index(drop=True),
    )

def test_save_load_preserves_tracer_names(populated_catalog, tmp_path):
    """Tracer names should be preserved through a save/load roundtrip."""
    path = tmp_path / "snapshot.h5"
    populated_catalog.save(path)
    loaded = SnapshotCatalog.load(path)
    assert set(loaded.tracers.keys()) == set(populated_catalog.tracers.keys())

def test_transforms_not_persisted(populated_catalog, tmp_path):
    """Transforms registered before saving should not be present after loading."""
    populated_catalog.ap(los="z")
    path = tmp_path / "snapshot.h5"
    populated_catalog.save(path)
    loaded = SnapshotCatalog.load(path)
    assert "ap" not in loaded.transform_pipeline

def test_save_load_preserves_cosmo(populated_catalog, tmp_path):
    """Cosmology values should be preserved through a save/load roundtrip."""
    path = tmp_path / "snapshot.h5"
    populated_catalog.save(path)
    loaded = SnapshotCatalog.load(path)
    assert loaded.cosmo != populated_catalog.cosmo  # Different instances
    assert loaded.cosmo.__class__ == populated_catalog.cosmo.__class__  # Same class
    for attr in ("_efunc", "_add"): # Same attributes should be equal
        assert getattr(loaded.cosmo, attr) == getattr(populated_catalog.cosmo, attr)


#%% RandomSnapshotCatalog

@pytest.fixture
def random_catalog(populated_catalog):
    return RandomSnapshotCatalog.from_snapshot(populated_catalog)

def test_from_snapshot_inherits_redshift(populated_catalog, random_catalog):
    """Redshift of the random catalog should match that of the original snapshot catalog."""
    assert random_catalog.redshift == populated_catalog.redshift

def test_from_snapshot_inherits_boxsize(populated_catalog, random_catalog):
    """Boxsize of the random catalog should match that of the original snapshot catalog."""
    np.testing.assert_array_equal(random_catalog._boxsize, populated_catalog._boxsize)

def test_from_snapshot_inherits_tracers(populated_catalog, random_catalog):
    """Tracers of the random catalog should match those of the original snapshot catalog."""
    assert set(random_catalog.tracers.keys()) == set(populated_catalog.tracers.keys())

def test_from_snapshot_same_ngal(populated_catalog, random_catalog):
    """Random catalog should have the same number of galaxies as the original snapshot catalog."""
    assert len(random_catalog.get_tracer_data("FOO", raw=True)) == len(populated_catalog._data["FOO"])

def test_from_snapshot_positions_differ(populated_catalog, random_catalog):
    """Random positions should differ from the original (with overwhelming probability)."""
    orig = populated_catalog._data["FOO"][["x", "y", "z"]].values
    rand = random_catalog.get_tracer_data("FOO", raw=True)[["x", "y", "z"]].values
    assert not np.allclose(orig, rand)

def test_random_positions_within_box(random_catalog):
    """Random positions should be within the box defined by the boxsize."""
    data = random_catalog.get_tracer_data("FOO", raw=True)
    boxsize = random_catalog._boxsize
    for i, col in enumerate(("x", "y", "z")):
        assert data[col].between(0, boxsize[i]).all()

def test_random_catalog_no_velocity_columns(random_catalog):
    """Random catalog should not have velocity columns since they are not meaningful for randoms."""
    data = random_catalog.get_tracer_data("FOO", raw=True)
    assert "vx" not in data.columns

def test_random_catalog_rsd_raises(random_catalog):
    """RSD transform should not be implemented for RandomSnapshotCatalog and should raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="RSD"):
        random_catalog.rsd()

#%%Multi-tracer catalogs

@pytest.fixture
def tracer_bar():
    return Tracer(name="BAR", params={})

@pytest.fixture
def valid_data_bar():
    """Smaller dataset for BAR tracer to allow testing different counts."""
    return make_tracer_data(n=50)

@pytest.fixture
def multi_tracer_catalog(catalog, tracer, tracer_bar, valid_data, valid_data_bar):
    catalog.set_tracer_data(tracer, valid_data)       # 100 galaxies
    catalog.set_tracer_data(tracer_bar, valid_data_bar)  # 50 galaxies
    return catalog

def test_nbar_multi_tracer(multi_tracer_catalog):
    """Nbar should use total ngal over total volume."""
    volume = 500.0 ** 3
    assert multi_tracer_catalog.nbar == pytest.approx(150 / volume)

def test_nbar_per_tracer(multi_tracer_catalog):
    """_nbar should return independent densities per tracer."""
    volume = 500.0 ** 3
    assert multi_tracer_catalog._nbar("FOO") == pytest.approx(100 / volume)
    assert multi_tracer_catalog._nbar("BAR") == pytest.approx(50 / volume)
    assert multi_tracer_catalog._nbar("FOO", "BAR") == pytest.approx(150 / volume)

def test_nbar_per_tracer_differ(multi_tracer_catalog):
    """Per-tracer nbar values should differ when tracers have different counts."""
    assert multi_tracer_catalog._nbar("FOO") != multi_tracer_catalog._nbar("BAR")

def test_rsd_applies_to_all_tracers(multi_tracer_catalog):
    """RSD is a catalog-level transform and should affect all tracers."""
    raw_foo = multi_tracer_catalog.get_tracer_data("FOO", raw=True)["z"].copy()
    raw_bar = multi_tracer_catalog.get_tracer_data("BAR", raw=True)["z"].copy()
    multi_tracer_catalog.rsd(los="z")
    assert not np.allclose(multi_tracer_catalog.get_tracer_data("FOO")["z"].values, raw_foo.values)
    assert not np.allclose(multi_tracer_catalog.get_tracer_data("BAR")["z"].values, raw_bar.values)

def test_ap_applies_to_all_tracers(multi_tracer_catalog):
    """AP is a catalog-level transform and should affect all tracers."""
    raw_foo = multi_tracer_catalog.get_tracer_data("FOO", raw=True)["z"].copy()
    raw_bar = multi_tracer_catalog.get_tracer_data("BAR", raw=True)["z"].copy()
    multi_tracer_catalog.ap(los="z")
    assert not np.allclose(multi_tracer_catalog.get_tracer_data("FOO")["z"].values, raw_foo.values)
    assert not np.allclose(multi_tracer_catalog.get_tracer_data("BAR")["z"].values, raw_bar.values)

def test_downsample_affects_only_target_tracer(multi_tracer_catalog):
    """Downsampling FOO should not change the number of galaxies in BAR."""
    multi_tracer_catalog.downsample("FOO", n_gal=50)
    assert len(multi_tracer_catalog.get_tracer_data("FOO")) == 50
    assert len(multi_tracer_catalog.get_tracer_data("BAR")) == 50

def test_downsample_independent_per_tracer(multi_tracer_catalog):
    """Each tracer can be downsampled independently."""
    multi_tracer_catalog.downsample("FOO", n_gal=60)
    multi_tracer_catalog.downsample("BAR", n_gal=30)
    assert len(multi_tracer_catalog.get_tracer_data("FOO")) == 60
    assert len(multi_tracer_catalog.get_tracer_data("BAR")) == 30

def test_downsample_ngal_multi_tracer_updates_total(multi_tracer_catalog):
    """Ngal should reflect downsampled counts across all tracers."""
    multi_tracer_catalog.downsample("FOO", n_gal=60)
    multi_tracer_catalog.downsample("BAR", n_gal=30)
    assert multi_tracer_catalog.ngal == 90

def test_downsample_nbar_uses_full_volume(multi_tracer_catalog):
    """Nbar downsampling should use catalog boxsize, not per-tracer extent."""
    volume = 500.0 ** 3
    target_nbar = 60 / volume
    multi_tracer_catalog.downsample("FOO", nbar=target_nbar)
    assert len(multi_tracer_catalog.get_tracer_data("FOO")) == 60
    assert len(multi_tracer_catalog.get_tracer_data("BAR")) == 50  # BAR unchanged

def test_downsample_does_not_affect_raw_data(multi_tracer_catalog):
    """Downsampling should not mutate _data for any tracer."""
    multi_tracer_catalog.downsample("FOO", n_gal=50)
    multi_tracer_catalog.get_tracer_data("FOO")
    assert len(multi_tracer_catalog._data["FOO"]) == 100
    assert len(multi_tracer_catalog._data["BAR"]) == 50

def test_multi_tracer_concatenates_positions(multi_tracer_catalog):
    """Positions from all tracers should be concatenated."""
    result = multi_tracer_catalog.positions()
    assert len(result) == multi_tracer_catalog.ngal

def test_multi_tracer_positions_reset_index(multi_tracer_catalog):
    """Index should be reset after concatenation — no duplicate indices."""
    result = multi_tracer_catalog.positions()
    assert list(result.index) == list(range(len(result)))


#%% Helpers
def make_positions(n=10, low=0.0, high=100.0, seed=0):
    """Create a random catalog of positions in a box."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, 3)).astype(np.float64)

class TestBoundaryCheck:
    """Test the boundary_check function with various scenarios."""

    def test_valid_catalog(self):
        """Test that a valid catalog passes the checks."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        boxsize = 100.0
        boundary_check(positions, boxsize)

    def test_array_boxsize(self):
        """Test that an array boxsize is accepted."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        boxsize = np.array([100.0, 100.0, 100.0])
        boundary_check(positions, boxsize)

    def test_list_boxsize(self):
        """Test that a list boxsize is accepted."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        boxsize = [100.0, 100.0, 100.0]
        boundary_check(positions, boxsize)

    def test_center_at_zero(self):
        """Test that positions centered at zero are accepted when center_at_zero is True."""
        positions = make_positions(n=100, low=-50.0, high=50.0)
        boxsize = 100.0
        boundary_check(positions, boxsize, center_at_zero=True)

    def test_float32_precision(self):
        """Test that checks can be performed in float32 precision."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        boxsize = 100.0
        boundary_check(positions, boxsize, dtype=np.float32)

    def test_left_edge_inclusive(self):
        """Test that positions on the left edge are accepted."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        positions[0, 0] = 0.0  # Set one position to the left edge
        boxsize = 100.0
        boundary_check(positions, boxsize)

    def test_right_edge_exclusive(self):
        """Test that positions on the right edge are rejected."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        positions[0, 0] = 100.0  # Set one position to the right edge
        boxsize = 100.0
        with pytest.raises(ValueError, match="right edge"):
            boundary_check(positions, boxsize)

    def test_invalid_boxsize_shape(self):
        """Test that an invalid boxsize shape raises an error."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        boxsize = np.array([100.0, 100.0])  # Invalid shape
        with pytest.raises(ValueError, match="boxsize"):
            boundary_check(positions, boxsize)

    def test_out_of_bounds_left(self):
        """Test that positions outside the left boundary raise an error."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        positions[0, 0] = -1.0  # Set one position outside the left boundary
        boxsize = 100.0
        with pytest.raises(ValueError, match="left edge"):
            boundary_check(positions, boxsize)

    def test_out_of_bounds_right(self):
        """Test that positions outside the right boundary raise an error."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        positions[0, 0] = 101.0  # Set one position outside the right boundary
        boxsize = 100.0
        with pytest.raises(ValueError, match="right edge"):
            boundary_check(positions, boxsize)

    def test_out_of_bounds_both_edges(self):
        """Test that positions outside both boundaries raise an error."""
        positions = make_positions(n=100, low=0.0, high=100.0)
        positions[0, 0] = -1.0  # Set one position outside the left boundary
        positions[1, 0] = 101.0  # Set another position outside the right boundary
        boxsize = 100.0
        with pytest.raises(ValueError) as exc_info:  # noqa: PT011
            boundary_check(positions, boxsize)
        assert "left edge" in str(exc_info.value)
        assert "right edge" in str(exc_info.value)

    def test_out_of_bounds_centered(self):
        """Test that positions outside the boundaries raise an error when center_at_zero is True."""
        positions = make_positions(n=100, low=-50.0, high=50.0)
        positions[0, 0] = -51.0  # Set one position outside the left boundary
        positions[1, 0] = 51.0   # Set another position outside the right boundary
        boxsize = 100.0
        with pytest.raises(ValueError) as exc_info:  # noqa: PT011
            boundary_check(positions, boxsize, center_at_zero=True)
        assert "left edge" in str(exc_info.value)
        assert "right edge" in str(exc_info.value)

    # Edge cases
    def test_single_position(self):
        """Test that a catalog with a single position is checked correctly."""
        positions = np.array([[50.0, 50.0, 50.0]])
        boxsize = 100.0
        boundary_check(positions, boxsize)

    def test_asymmetric_box(self):
        """Test that an asymmetric box size is handled correctly."""
        # Symmetric positions in the smallest box dimension
        positions = make_positions(n=100, low=0.0, high=25.0)
        boxsize = [100.0, 50.0, 25.0]
        boundary_check(positions, boxsize)

    def test_asymmetric_box_centered(self):
        """Test that an asymmetric box size with centered positions is handled correctly."""
        # Symmetric positions in the smallest box dimension
        positions = make_positions(n=100, low=-12.5, high=12.5)
        boxsize = [100.0, 50.0, 25.0]
        boundary_check(positions, boxsize, center_at_zero=True)

    def test_asymmetric_box_out_of_bounds(self):
        """Test that positions outside the boundaries of an asymmetric box raise an error."""
        # Symmetric positions in the smallest box dimension
        positions = make_positions(n=100, low=0.0, high=25.0)
        positions[0, 1] = 51.0  # Set one position outside the second dimension boundary
        boxsize = [100.0, 50.0, 25.0]
        with pytest.raises(ValueError, match="right edge"):
            boundary_check(positions, boxsize)
