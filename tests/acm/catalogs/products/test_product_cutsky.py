import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from acm.catalogs.dataclasses import Tracer
from acm.catalogs.products.cutsky import (
    CutskyCatalog,
    RandomCutskyCatalog,
    _fsky,
    _shell_volume,
)

#%% Fixtures 

def make_tracer_data(n: int = 200) -> pd.DataFrame:
    """Generate minimal valid cutsky tracer data with known angular and redshift ranges."""
    rng = np.random.default_rng(seed=0)
    return pd.DataFrame({
        "ra":  rng.uniform(10.0, 50.0, n),
        "dec": rng.uniform(-20.0, 20.0, n),
        "z":   rng.uniform(0.5, 1.0, n),
    })

@pytest.fixture
def cosmo():
    m = MagicMock()
    m.comoving_radial_distance.side_effect = lambda z: 1000.0 * np.asarray(z)
    return m

@pytest.fixture
def cosmo_fid():
    m = MagicMock()
    m.comoving_radial_distance.side_effect = lambda z: 1000.0 * np.asarray(z)
    return m

@pytest.fixture
def catalog(cosmo, cosmo_fid):
    return CutskyCatalog(cosmo=cosmo, cosmo_fid=cosmo_fid)

@pytest.fixture
def tracer():
    return Tracer(name="FOO", params={})

@pytest.fixture
def valid_data():
    return make_tracer_data()

@pytest.fixture
def populated_catalog(catalog, tracer, valid_data):
    catalog.set_tracer_data(tracer, valid_data)
    return catalog


#%% Module-level helpers

class TestFsky:
    def test_fullsky_returns_one(self):
        """Densely sampled full-sky positions should approach fsky=1."""
        rng = np.random.default_rng(0)
        n = 500_000
        ra = rng.uniform(0, 360, n)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
        result = _fsky(ra, dec, nside=64)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_small_patch_less_than_one(self):
        """A small sky patch should produce fsky well below 1."""
        rng = np.random.default_rng(0)
        ra = rng.uniform(10, 20, 5000)
        dec = rng.uniform(-5, 5, 5000)
        result = _fsky(ra, dec, nside=64)
        assert result < 0.1

    def test_returns_float_in_unit_interval(self):
        """fsky should always be in [0, 1]."""
        rng = np.random.default_rng(0)
        ra = rng.uniform(0, 360, 1000)
        dec = rng.uniform(-90, 90, 1000)
        result = _fsky(ra, dec)
        assert 0.0 <= result <= 1.0


class TestShellVolume:
    def test_output_shape(self, cosmo):
        """Output shape should be (n_bins,) for (n_bins+1,) input edges."""
        z = np.linspace(0.0, 1.0, 6)
        result = _shell_volume(cosmo, z)
        assert result.shape == (5,)

    def test_positive_volumes(self, cosmo):
        """All shell volumes should be positive for increasing redshift edges."""
        z = np.linspace(0.1, 1.0, 5)
        result = _shell_volume(cosmo, z)
        assert np.all(result > 0)

    def test_single_shell(self, cosmo):
        """A two-edge input should return a single shell volume."""
        z = np.array([0.0, 1.0])
        result = _shell_volume(cosmo, z)
        assert result.shape == (1,)
        expected = 4 / 3 * np.pi * (1000.0 ** 3 - 0.0 ** 3)
        assert result[0] == pytest.approx(expected)

    def test_volumes_increase_with_redshift(self, cosmo):
        """Shells at higher redshift should have larger volume."""
        z = np.array([0.0, 0.5, 1.0, 1.5])
        result = _shell_volume(cosmo, z)
        assert result[1] > result[0]
        assert result[2] > result[1]


#%% CutskyCatalog construction 

def test_default_hp_res(catalog):
    assert catalog.hp_res == 256

def test_custom_hp_res(cosmo, cosmo_fid):
    cat = CutskyCatalog(cosmo=cosmo, cosmo_fid=cosmo_fid, hp_res=128)
    assert cat.hp_res == 128

def test_caches_initialised_empty(catalog):
    assert catalog._fsky_cache == {}
    assert catalog._interpolate_nz_cache == {}


#%% _check_data_columns 

def test_check_data_columns_valid(catalog, valid_data):
    assert catalog._check_data_columns(valid_data) is True

def test_check_data_columns_missing_ra(catalog):
    bad = pd.DataFrame({"dec": [0.0], "z": [0.5]})
    assert catalog._check_data_columns(bad) is False

def test_check_data_columns_missing_all(catalog):
    assert catalog._check_data_columns(pd.DataFrame()) is False

def test_set_tracer_data_missing_columns_raises(catalog, tracer):
    bad = pd.DataFrame({"ra": [10.0], "dec": [5.0]})  # missing z
    with pytest.raises(ValueError, match="missing required columns"):
        catalog.set_tracer_data(tracer, bad)


#%% Coordinate ranges

def test_zrange_bounds(populated_catalog, valid_data):
    zmin, zmax = populated_catalog.zrange
    assert zmin == pytest.approx(valid_data["z"].min())
    assert zmax == pytest.approx(valid_data["z"].max())

def test_zrange_per_tracer(populated_catalog, valid_data):
    zmin, zmax = populated_catalog._zrange("FOO")
    assert zmin == pytest.approx(valid_data["z"].min())
    assert zmax == pytest.approx(valid_data["z"].max())

def test_zrange_none_equals_global(populated_catalog):
    assert populated_catalog._zrange() == populated_catalog.zrange

def test_rarange_wrapping(cosmo, cosmo_fid):
    """RA values stored beyond 360 (e.g. from box periodicity) should produce
    a wrap-around range where ra_min > ra_max after mod."""
    tracer = Tracer(name="FOO", params={})
    cat = CutskyCatalog(cosmo=cosmo, cosmo_fid=cosmo_fid)
    rng = np.random.default_rng(0)
    # Simulate raw RA values that cross 360 without being wrapped first
    data = pd.DataFrame({
        "ra":  np.concatenate([rng.uniform(350, 370, 100), rng.uniform(370, 380, 100)]),
        "dec": rng.uniform(-10, 10, 200),
        "z":   rng.uniform(0.5, 1.0, 200),
    })
    cat.set_tracer_data(tracer, data)
    ra_min, ra_max = cat._range("ra", periodic_wrap=360.0)
    assert ra_min > ra_max


#%% fsky and area 

def test_fsky_no_tracers_raises(catalog):
    with pytest.raises(RuntimeError, match="No tracers"):
        _ = catalog.fsky

def test_fsky_in_unit_interval(populated_catalog):
    assert 0.0 < populated_catalog.fsky <= 1.0

def test_fsky_fullsky_catalog(cosmo, cosmo_fid):
    """A full-sky catalog should return fsky close to 1."""
    tracer = Tracer(name="FOO", params={})
    cat = CutskyCatalog(cosmo=cosmo, cosmo_fid=cosmo_fid, hp_res=64)
    rng = np.random.default_rng(0)
    n = 500_000
    data = pd.DataFrame({
        "ra":  rng.uniform(0, 360, n),
        "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
        "z":   rng.uniform(0.5, 1.0, n),
    })
    cat.set_tracer_data(tracer, data)
    assert cat.fsky == pytest.approx(1.0, abs=0.01)

def test_fsky_cached(populated_catalog):
    """Repeated fsky calls should return the same value from cache."""
    first = populated_catalog.fsky
    second = populated_catalog.fsky
    assert first == second

def test_area_consistent_with_fsky(populated_catalog):
    full_sky_deg2 = 4 * np.pi * (180 / np.pi) ** 2
    assert populated_catalog.area == pytest.approx(populated_catalog.fsky * full_sky_deg2)

def test_clear_caches(populated_catalog):
    _ = populated_catalog.fsky
    populated_catalog.clear_caches()
    assert populated_catalog._fsky_cache == {}
    assert populated_catalog._interpolate_nz_cache == {}


#%% nbar

def test_nbar_positive(populated_catalog):
    assert populated_catalog.nbar > 0.0

def test_nbar_delegates_to_nbar_method(populated_catalog):
    assert populated_catalog.nbar == populated_catalog._nbar()

def test_nbar_per_tracer_positive(populated_catalog):
    assert populated_catalog._nbar("FOO") > 0.0

def test_nbar_per_tracer_matches_global_single_tracer(populated_catalog):
    """With a single tracer, per-tracer and global nbar should agree."""
    assert populated_catalog._nbar("FOO") == pytest.approx(populated_catalog.nbar)


#%% n(z) interpolation

def test_n_returns_positive_within_range(populated_catalog, valid_data):
    z_mid = valid_data["z"].mean()
    assert populated_catalog.n(z_mid) > 0.0

def test_n_returns_zero_outside_range(populated_catalog):
    assert populated_catalog.n(0.0) == pytest.approx(0.0)
    assert populated_catalog.n(10.0) == pytest.approx(0.0)

def test_n_array_input(populated_catalog, valid_data):
    z = np.linspace(valid_data["z"].min(), valid_data["z"].max(), 10)
    result = populated_catalog.n(z)
    assert result.shape == (10,)
    assert np.all(result >= 0.0)

def test_interpolate_nz_cached(populated_catalog):
    """Repeated _interpolate_nz calls with the same arguments should return the cached object."""
    f1 = populated_catalog._interpolate_nz(bins=30)
    f2 = populated_catalog._interpolate_nz(bins=30)
    assert f1 is f2

def test_interpolate_nz_different_bins_not_cached(populated_catalog):
    """Different bin counts should produce independent cache entries."""
    f1 = populated_catalog._interpolate_nz(bins=30)
    f2 = populated_catalog._interpolate_nz(bins=50)
    assert f1 is not f2

def test_interpolate_nz_invalidated_after_transform(populated_catalog):
    """Cache should be invalidated when a new transform is added."""
    f1 = populated_catalog._interpolate_nz(bins=30)
    populated_catalog.add_distance_column()
    f2 = populated_catalog._interpolate_nz(bins=30)
    assert f1 is not f2


#%% add_distance_column transform 

def test_add_distance_column_adds_transform(populated_catalog):
    populated_catalog.add_distance_column()
    assert "add_distance" in populated_catalog.transform_pipeline

def test_add_distance_column_present_in_output(populated_catalog):
    populated_catalog.add_distance_column()
    result = populated_catalog.get_tracer_data("FOO")
    assert "distance" in result.columns

def test_add_distance_column_does_not_mutate_raw(populated_catalog):
    raw_before = populated_catalog.get_tracer_data("FOO", raw=True).copy()
    populated_catalog.add_distance_column()
    populated_catalog.get_tracer_data("FOO")
    pd.testing.assert_frame_equal(
        populated_catalog._data["FOO"], raw_before
    )


#%% downsample transform 

def test_downsample_adds_transform(populated_catalog):
    populated_catalog.downsample("FOO", n_gal=100)
    assert "downsample_FOO" in populated_catalog.transform_pipeline

def test_downsample_multiple_params_raises(populated_catalog):
    with pytest.raises(ValueError, match="Exactly one"):
        populated_catalog.downsample("FOO", n_gal=100, f_gal=0.5)

def test_downsample_no_params_raises(populated_catalog):
    with pytest.raises(ValueError, match="Exactly one"):
        populated_catalog.downsample("FOO")

def test_downsample_by_ngal(populated_catalog):
    populated_catalog.downsample("FOO", n_gal=100)
    assert len(populated_catalog.get_tracer_data("FOO")) == 100

def test_downsample_by_fgal(populated_catalog):
    populated_catalog.downsample("FOO", f_gal=0.5)
    assert len(populated_catalog.get_tracer_data("FOO")) == 100

def test_downsample_does_not_mutate_raw(populated_catalog):
    n_before = len(populated_catalog._data["FOO"])
    populated_catalog.downsample("FOO", n_gal=100)
    populated_catalog.get_tracer_data("FOO")
    assert len(populated_catalog._data["FOO"]) == n_before


#%% Serialization

def test_save_creates_file(populated_catalog, tmp_path, cosmo, cosmo_fid):
    path = tmp_path / "cutsky.h5"
    populated_catalog.save(path)
    assert path.exists()

def test_save_load_roundtrip(populated_catalog, tmp_path, cosmo, cosmo_fid):
    path = tmp_path / "cutsky.h5"
    populated_catalog.save(path)
    loaded = CutskyCatalog.load(path, cosmo, cosmo_fid)
    assert isinstance(loaded, CutskyCatalog)

def test_save_load_tracer_data(populated_catalog, tmp_path, cosmo, cosmo_fid, valid_data):
    """Tracer data should be preserved exactly through a save/load roundtrip."""
    path = tmp_path / "cutsky.h5"
    populated_catalog.save(path)
    loaded = CutskyCatalog.load(path, cosmo, cosmo_fid)
    pd.testing.assert_frame_equal(
        loaded.get_tracer_data("FOO", raw=True).reset_index(drop=True),
        valid_data.reset_index(drop=True),
    )
    
def test_save_load_preserves_tracer_names(populated_catalog, tmp_path, cosmo, cosmo_fid):
    """Tracer names should be preserved through a save/load roundtrip."""
    path = tmp_path / "cutsky.h5"
    populated_catalog.save(path)
    loaded = CutskyCatalog.load(path, cosmo, cosmo_fid)
    assert set(loaded.tracers.keys()) == set(populated_catalog.tracers.keys())
    
def test_transforms_not_persisted(populated_catalog, tmp_path, cosmo, cosmo_fid):
    """Transforms registered before saving should not be present after loading."""
    populated_catalog.add_distance_column()
    path = tmp_path / "cutsky.h5"
    populated_catalog.save(path)
    loaded = CutskyCatalog.load(path, cosmo, cosmo_fid)
    assert "add_distance" not in loaded.transform_pipeline


#%% Multi-tracer 

@pytest.fixture
def tracer_bar():
    return Tracer(name="BAR", params={})

@pytest.fixture
def valid_data_bar():
    return make_tracer_data(n=100)

@pytest.fixture
def multi_tracer_catalog(catalog, tracer, tracer_bar, valid_data, valid_data_bar):
    catalog.set_tracer_data(tracer, valid_data)        # 200 galaxies
    catalog.set_tracer_data(tracer_bar, valid_data_bar)  # 100 galaxies
    return catalog

def test_zrange_aggregates_across_tracers(multi_tracer_catalog, valid_data, valid_data_bar):
    all_z = pd.concat([valid_data["z"], valid_data_bar["z"]])
    zmin, zmax = multi_tracer_catalog.zrange
    assert zmin == pytest.approx(all_z.min())
    assert zmax == pytest.approx(all_z.max())

def test_zrange_per_tracer_independent(multi_tracer_catalog, valid_data):
    assert multi_tracer_catalog._zrange("FOO") == (
        pytest.approx(valid_data["z"].min()),
        pytest.approx(valid_data["z"].max()),
    )

def test_nbar_multi_tracer_uses_total_ngal(multi_tracer_catalog):
    assert multi_tracer_catalog.nbar == pytest.approx(multi_tracer_catalog._nbar())

def test_nbar_per_tracer_differ(multi_tracer_catalog):
    """Per-tracer nbar should differ when tracers have different counts."""
    assert multi_tracer_catalog._nbar("FOO") != multi_tracer_catalog._nbar("BAR")

def test_downsample_affects_only_target_tracer(multi_tracer_catalog):
    multi_tracer_catalog.downsample("FOO", n_gal=150)
    assert len(multi_tracer_catalog.get_tracer_data("FOO")) == 150
    assert len(multi_tracer_catalog.get_tracer_data("BAR")) == 100

def test_add_distance_column_applies_to_all_tracers(multi_tracer_catalog):
    multi_tracer_catalog.add_distance_column()
    assert "distance" in multi_tracer_catalog.get_tracer_data("FOO").columns
    assert "distance" in multi_tracer_catalog.get_tracer_data("BAR").columns


#%% RandomCutskyCatalog 

@pytest.fixture
def random_catalog(populated_catalog):
    return RandomCutskyCatalog.from_snapshot(populated_catalog, seed=42)

def test_from_snapshot_inherits_tracers(populated_catalog, random_catalog):
    assert set(random_catalog.tracers.keys()) == set(populated_catalog.tracers.keys())

def test_from_snapshot_same_ngal(populated_catalog, random_catalog):
    assert len(random_catalog.get_tracer_data("FOO", raw=True)) == len(populated_catalog._data["FOO"])

def test_from_snapshot_positions_differ(populated_catalog, random_catalog):
    orig = populated_catalog._data["FOO"][["ra", "dec", "z"]].values
    rand = random_catalog.get_tracer_data("FOO", raw=True)[["ra", "dec", "z"]].values
    assert not np.allclose(orig, rand)

def test_random_positions_ra_in_range(random_catalog):
    ra = random_catalog.get_tracer_data("FOO", raw=True)["ra"]
    assert ra.between(0, 360).all()

def test_random_positions_dec_in_range(random_catalog):
    dec = random_catalog.get_tracer_data("FOO", raw=True)["dec"]
    assert dec.between(-90, 90).all()

def test_random_positions_z_in_source_range(populated_catalog, random_catalog):
    zmin, zmax = populated_catalog._zrange("FOO")
    z = random_catalog.get_tracer_data("FOO", raw=True)["z"]
    assert z.between(zmin, zmax).all()

def test_from_snapshot_reproducible(populated_catalog):
    r1 = RandomCutskyCatalog.from_snapshot(populated_catalog, seed=7)
    r2 = RandomCutskyCatalog.from_snapshot(populated_catalog, seed=7)
    pd.testing.assert_frame_equal(
        r1.get_tracer_data("FOO", raw=True).reset_index(drop=True),
        r2.get_tracer_data("FOO", raw=True).reset_index(drop=True),
    )

def test_from_snapshot_different_seeds_differ(populated_catalog):
    r1 = RandomCutskyCatalog.from_snapshot(populated_catalog, seed=7)
    r2 = RandomCutskyCatalog.from_snapshot(populated_catalog, seed=8)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            r1.get_tracer_data("FOO", raw=True).reset_index(drop=True),
            r2.get_tracer_data("FOO", raw=True).reset_index(drop=True),
        )

def test_from_snapshot_tracers_independent_seeds(populated_catalog, tracer_bar, valid_data_bar):
    """Two tracers generated from the same root seed should have different positions."""
    populated_catalog.set_tracer_data(tracer_bar, valid_data_bar)
    random_cat = RandomCutskyCatalog.from_snapshot(populated_catalog, seed=42)
    foo_z = random_cat.get_tracer_data("FOO", raw=True)["z"].values[:100]
    bar_z = random_cat.get_tracer_data("BAR", raw=True)["z"].values
    assert not np.allclose(foo_z, bar_z)


#%% RandomCutskyCatalog._random_positions 

class TestRandomPositions:
    def test_output_shape(self):
        result = RandomCutskyCatalog._random_positions(
            500, rarange=(0, 360), decrange=(-90, 90), zrange=(0.5, 1.0), seed=0
        )
        assert result.shape == (500, 3)

    def test_required_columns(self):
        result = RandomCutskyCatalog._random_positions(
            10, rarange=(0, 360), decrange=(-90, 90), zrange=(0.5, 1.0)
        )
        assert set(result.columns) == {"ra", "dec", "z"}

    def test_ra_wrapping(self):
        """RA values should always be in [0, 360) even for wrap-around ranges."""
        result = RandomCutskyCatalog._random_positions(
            1000, rarange=(300, 40), decrange=(-10, 10), zrange=(0.5, 1.0), seed=0
        )
        assert result["ra"].between(0, 360).all()

    def test_ra_wrapping_covers_both_sides(self):
        """Wrap-around RA generation should produce values both above 300 and below 40."""
        result = RandomCutskyCatalog._random_positions(
            5000, rarange=(300, 40), decrange=(-10, 10), zrange=(0.5, 1.0), seed=0
        )
        assert (result["ra"] > 300).any()
        assert (result["ra"] < 40).any()

    # def test_dec_uniform_on_sphere(self):
    #     """sin(dec) should be approximately uniformly distributed for large samples."""
    #     result = RandomCutskyCatalog._random_positions(
    #         50_000, rarange=(0, 360), decrange=(-90, 90), zrange=(0.5, 1.0), seed=0
    #     )
    #     sin_dec = np.sin(np.radians(result["dec"]))
    #     _, p = scipy.stats.kstest(sin_dec, "uniform", args=(-1, 2))
    #     assert p > 0.01

    def test_z_within_range(self):
        result = RandomCutskyCatalog._random_positions(
            500, rarange=(0, 360), decrange=(-90, 90), zrange=(0.3, 0.8), seed=0
        )
        assert result["z"].between(0.3, 0.8).all()

    def test_reproducible_with_seed(self):
        r1 = RandomCutskyCatalog._random_positions(100, (0, 360), (-90, 90), (0.5, 1.0), seed=1)
        r2 = RandomCutskyCatalog._random_positions(100, (0, 360), (-90, 90), (0.5, 1.0), seed=1)
        pd.testing.assert_frame_equal(r1, r2)