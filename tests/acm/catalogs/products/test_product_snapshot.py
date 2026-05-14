import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from acm.catalogs.dataclasses import Tracer
from acm.catalogs.products.snapshot import (
    SnapshotCatalog,
    RandomSnapshotCatalog,
)

#%% Fixtures

def make_tracer_data(n: int = 100) -> pd.DataFrame:
    """Generate minimal valid tracer data."""
    return pd.DataFrame({
        "x": np.random.uniform(0, 500, n),
        "y": np.random.uniform(0, 500, n),
        "z": np.random.uniform(0, 500, n),
        "vx": np.random.normal(0, 1, n),
        "vy": np.random.normal(0, 1, n),
        "vz": np.random.normal(0, 1, n),
    })

@pytest.fixture
def cosmo():
    m = MagicMock()
    m.efunc.return_value = 1.0
    m.angular_diameter_distance.return_value = 1000.0
    return m

@pytest.fixture
def cosmo_fid():
    m = MagicMock()
    m.efunc.return_value = 1.0
    m.angular_diameter_distance.return_value = 1000.0
    return m

@pytest.fixture
def boxsize():
    return 500.0

@pytest.fixture
def catalog(cosmo, cosmo_fid, boxsize):
    return SnapshotCatalog(redshift=0.5, cosmo=cosmo, cosmo_fid=cosmo_fid, boxsize=boxsize)

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

def test_boxsize_array(cosmo, cosmo_fid):
    """Array boxsize should be stored as-is."""
    cat = SnapshotCatalog(redshift=0.5, cosmo=cosmo, cosmo_fid=cosmo_fid, boxsize=[100., 200., 300.])
    np.testing.assert_array_equal(cat._boxsize, [100., 200., 300.])

def test_boxsize_invalid_shape_raises(cosmo, cosmo_fid):
    """Boxsize with invalid shape should raise an error."""
    with pytest.raises(ValueError, match="shape"):
        SnapshotCatalog(redshift=0.5, cosmo=cosmo, cosmo_fid=cosmo_fid, boxsize=[1., 2.])

def test_repr(populated_catalog):
    """Repr should include class name, redshift, and tracer names."""
    r = repr(populated_catalog)
    assert "SnapshotCatalog" in r
    assert "0.5" in r
    assert "FOO" in r


#%% Cosmological properties

def test_az(catalog):
    assert catalog.az == pytest.approx(1.0 / 1.5)

def test_hubble(catalog, cosmo):
    cosmo.efunc.return_value = 0.8
    assert catalog.hubble == pytest.approx(80.0)

def test_hubble_fid(catalog, cosmo_fid):
    cosmo_fid.efunc.return_value = 0.9
    assert catalog.hubble_fid == pytest.approx(90.0)

def test_q_par(catalog):
    """q_par = hubble_fid / hubble; both cosmologies identical here so q_par = 1."""
    assert catalog.q_par == pytest.approx(1.0)

def test_q_perp(catalog):
    """q_perp = DA / DA_fid; both cosmologies identical here so q_perp = 1."""
    assert catalog.q_perp == pytest.approx(1.0)


#%% boxsize with AP

def test_boxsize_without_ap(catalog):
    """Without AP transform, boxsize should be unchanged."""
    np.testing.assert_array_equal(catalog.boxsize, [500., 500., 500.])

def test_boxsize_with_ap(populated_catalog, cosmo, cosmo_fid):
    """AP transform should scale boxsize by q_par along los and q_perp transversely."""
    cosmo.efunc.return_value = 0.8
    cosmo_fid.efunc.return_value = 1.0
    cosmo.angular_diameter_distance.return_value = 800.0
    cosmo_fid.angular_diameter_distance.return_value = 1000.0
    populated_catalog.ap(los="z")
    boxsize = populated_catalog.boxsize
    assert boxsize[2] == pytest.approx(500.0 * populated_catalog.q_par)
    assert boxsize[0] == pytest.approx(500.0 * populated_catalog.q_perp)


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
    """nbar property should raise if no tracers are registered."""
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

def test_rsd_shifts_positions(populated_catalog):
    """RSD transform should shift positions along the los according to the formula z' = z + vz / (H * az)."""
    raw = populated_catalog.get_tracer_data("FOO", raw=True).copy()
    populated_catalog.rsd(los="z")
    result = populated_catalog.get_tracer_data("FOO")
    expected_z = raw["z"] + raw["vz"] / (populated_catalog.hubble * populated_catalog.az)
    pd.testing.assert_series_equal(result["z"], expected_z, check_names=False)

def test_rsd_does_not_mutate_raw(populated_catalog):
    """RSD transform should not mutate the raw data stored in the catalog."""
    raw_before = populated_catalog.get_tracer_data("FOO", raw=True).copy()
    populated_catalog.rsd(los="z")
    populated_catalog.get_tracer_data("FOO")
    pd.testing.assert_frame_equal(populated_catalog._data["FOO"], raw_before)


#%% AP transform

def test_ap_adds_transform(populated_catalog):
    """AP transform should be added to the pipeline when ap() is called."""
    populated_catalog.ap(los="z")
    assert "ap" in populated_catalog._transforms

def test_ap_invalid_los_raises(populated_catalog):
    """Invalid los should raise an error."""
    with pytest.raises(ValueError, match="los"):
        populated_catalog.ap(los="w")

def test_ap_scales_positions(populated_catalog, cosmo, cosmo_fid):
    """AP transform should scale positions by q_par along the los and q_perp transversely."""
    cosmo.efunc.return_value = 0.8
    cosmo_fid.efunc.return_value = 1.0
    raw = populated_catalog.get_tracer_data("FOO", raw=True).copy()
    populated_catalog.ap(los="z")
    result = populated_catalog.get_tracer_data("FOO")
    assert result["z"].values == pytest.approx(raw["z"].values * populated_catalog.q_par)
    assert result["x"].values == pytest.approx(raw["x"].values * populated_catalog.q_perp)


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

def test_downsample_nbar_uses_ap_boxsize(populated_catalog, cosmo, cosmo_fid):
    """nbar downsampling should use AP-scaled boxsize when AP is in the pipeline."""
    cosmo.efunc.return_value = 0.8
    cosmo_fid.efunc.return_value = 1.0
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

def test_save_creates_file(populated_catalog, tmp_path, cosmo, cosmo_fid):
    path = tmp_path / "snapshot.h5"
    populated_catalog.save(path)
    assert path.exists()

def test_save_load_roundtrip(populated_catalog, tmp_path, cosmo, cosmo_fid):
    path = tmp_path / "snapshot.h5"
    populated_catalog.save(path)
    loaded = SnapshotCatalog.load(path, cosmo, cosmo_fid)
    assert loaded.redshift == pytest.approx(0.5)
    np.testing.assert_array_equal(loaded._boxsize, populated_catalog._boxsize)

def test_save_load_tracer_data(populated_catalog, tmp_path, cosmo, cosmo_fid, valid_data):
    """Test that tracer data is preserved through a save/load roundtrip, including column order."""
    path = tmp_path / "snapshot.h5"
    populated_catalog.save(path)
    loaded = SnapshotCatalog.load(path, cosmo, cosmo_fid)
    pd.testing.assert_frame_equal(
        loaded.get_tracer_data("FOO", raw=True).reset_index(drop=True),
        valid_data.reset_index(drop=True),
    )




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
    """nbar should use total ngal over total volume."""
    volume = 500.0 ** 3
    assert multi_tracer_catalog.nbar == pytest.approx(150 / volume)

def test_nbar_per_tracer(multi_tracer_catalog):
    """_nbar should return independent densities per tracer."""
    volume = 500.0 ** 3
    assert multi_tracer_catalog._nbar("FOO") == pytest.approx(100 / volume)
    assert multi_tracer_catalog._nbar("BAR") == pytest.approx(50 / volume)

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

def test_ap_applies_to_all_tracers(multi_tracer_catalog, cosmo, cosmo_fid):
    """AP is a catalog-level transform and should affect all tracers."""
    cosmo.efunc.return_value = 0.8
    cosmo_fid.efunc.return_value = 1.0
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
    """ngal should reflect downsampled counts across all tracers."""
    multi_tracer_catalog.downsample("FOO", n_gal=60)
    multi_tracer_catalog.downsample("BAR", n_gal=30)
    assert multi_tracer_catalog.ngal == 90

def test_downsample_nbar_uses_full_volume(multi_tracer_catalog):
    """nbar downsampling should use catalog boxsize, not per-tracer extent."""
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