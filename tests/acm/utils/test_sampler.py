"""Tests for acm.utils.sampler module."""
import pandas as pd
import pytest

from acm.utils.sampler import LatinHyperCubeSampler

# ruff: noqa: ANN001, ANN201, D101, D102, D103, S101

RANGES = {"alpha": (0.0, 1.0), "beta": (10.0, 20.0), "gamma": (-5.0, 5.0)}


@pytest.fixture
def sampler():
    return LatinHyperCubeSampler(RANGES, seed=42)

@pytest.fixture
def sample(sampler):
    return sampler.sample(10)


class TestSample:
    def test_returns_dataframe(self, sample):
        assert isinstance(sample, pd.DataFrame)

    def test_shape(self, sample):
        assert sample.shape == (10, len(RANGES))

    def test_columns_match_ranges(self, sample):
        assert list(sample.columns) == list(RANGES)

    def test_values_within_bounds(self, sample):
        for col, (lo, hi) in RANGES.items():
            assert sample[col].between(lo, hi).all()

    def test_reproducible_with_same_seed(self):
        s1 = LatinHyperCubeSampler(RANGES, seed=0).sample(5)
        s2 = LatinHyperCubeSampler(RANGES, seed=0).sample(5)
        pd.testing.assert_frame_equal(s1, s2)

    def test_different_seeds_differ(self):
        s1 = LatinHyperCubeSampler(RANGES, seed=0).sample(5)
        s2 = LatinHyperCubeSampler(RANGES, seed=1).sample(5)
        assert not s1.equals(s2)

    def test_single_sample(self, sampler):
        result = sampler.sample(1)
        assert result.shape == (1, len(RANGES))


class TestSplit:
    def test_returns_dict(self, sample):
        result = LatinHyperCubeSampler.split(sample, ["a", "b"])
        assert isinstance(result, dict)

    def test_keys_match(self, sample):
        keys = ["a", "b", "c"]
        result = LatinHyperCubeSampler.split(sample, keys)
        assert set(result.keys()) == set(keys)

    def test_total_rows_preserved(self, sample):
        result = LatinHyperCubeSampler.split(sample, ["a", "b"])
        assert sum(len(v) for v in result.values()) == len(sample)

    def test_columns_preserved(self, sample):
        result = LatinHyperCubeSampler.split(sample, ["a", "b"])
        for df in result.values():
            assert list(df.columns) == list(sample.columns)

    def test_single_split(self, sample):
        """Splitting into one key should return the full sample."""
        result = LatinHyperCubeSampler.split(sample, ["all"])
        pd.testing.assert_frame_equal(result["all"].reset_index(drop=True), sample)

    def test_uneven_split(self):
        """With 3 keys and 10 rows numpy splits as 4, 3, 3."""
        sample = pd.DataFrame({"x": range(10)})
        result = LatinHyperCubeSampler.split(sample, ["a", "b", "c"])
        assert [len(v) for v in result.values()] == [4, 3, 3]


class TestAddColumns:
    def test_new_columns_added(self, sample):
        extra = pd.DataFrame({"z": [99.0]})
        result = LatinHyperCubeSampler.add_columns(sample, extra)
        assert "z" in result.columns

    def test_original_columns_preserved(self, sample):
        extra = pd.DataFrame({"z": [99.0]})
        result = LatinHyperCubeSampler.add_columns(sample, extra)
        for col in sample.columns:
            assert col in result.columns

    def test_extra_values_repeated(self, sample):
        extra = pd.DataFrame({"z": [7.0]})
        result = LatinHyperCubeSampler.add_columns(sample, extra)
        assert (result["z"] == 7.0).all()

    def test_shape(self, sample):
        extra = pd.DataFrame({"z": [0.0], "w": [1.0]})
        result = LatinHyperCubeSampler.add_columns(sample, extra)
        assert result.shape == (len(sample), len(sample.columns) + 2)


class TestSave:
    def test_save_dataframe(self, sample, tmp_path):
        fn = tmp_path / "out.csv"
        LatinHyperCubeSampler.save(sample, fn)
        assert fn.exists()
        loaded = pd.read_csv(fn)
        assert list(loaded.columns) == list(sample.columns)

    def test_save_creates_parent_dirs(self, sample, tmp_path):
        fn = tmp_path / "nested" / "dir" / "out.csv"
        LatinHyperCubeSampler.save(sample, fn)
        assert fn.exists()

    def test_save_dict(self, sample, tmp_path):
        splits = LatinHyperCubeSampler.split(sample, ["train", "test"])
        fn = tmp_path / "split_{key}.csv"
        LatinHyperCubeSampler.save(splits, fn)
        assert (tmp_path / "split_train.csv").exists()
        assert (tmp_path / "split_test.csv").exists()

    def test_save_with_order(self, sample, tmp_path):
        """Order parameter should control column order in the saved CSV."""
        fn = tmp_path / "ordered.csv"
        order = ["beta", "alpha", "gamma"]
        LatinHyperCubeSampler.save(sample, fn, order=order)
        loaded = pd.read_csv(fn)
        assert list(loaded.columns) == order

    def test_save_float_format(self, sample, tmp_path):
        """Values should be saved with 5 decimal places."""
        fn = tmp_path / "out.csv"
        LatinHyperCubeSampler.save(sample, fn)
        raw = fn.read_text()
        for line in raw.strip().split("\n")[1:]:  # skip header
            for val in line.split(","):
                decimals = len(val.split(".")[-1]) if "." in val else 0
                assert decimals <= 5
