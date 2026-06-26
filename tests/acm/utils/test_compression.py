import numpy as np
import pytest
import xarray

from acm.utils.compression import (
    cast_coords,
    collect_measurements,
    compress_measurements,
    reindex_samples,
    reshape_to_coords,
    split_test_set,
)

# ruff: noqa: ANN001, ANN201, ANN202, ARG001, INP001, S101

class TestReshapeToCoords:
    """Tests for the reshape_to_coords function."""

    def test_basic(self):
        """Test that an array can be reshaped to match the provided coordinate lengths."""
        arr = np.arange(6)
        coords = {"a": [0, 1], "b": [0, 1, 2]}
        result = reshape_to_coords(arr, coords)
        assert result.shape == (2, 3)

    def test_single_dim(self):
        """Test that an array can be reshaped to a single dimension."""
        arr = np.arange(3)
        coords = {"a": [0, 1, 2]}
        result = reshape_to_coords(arr, coords)
        assert result.shape == (3,)

    def test_mismatch_raises(self):
        """Test that a ValueError is raised if the array size does not match the product of coordinate lengths."""
        arr = np.arange(5)
        coords = {"a": [0, 1], "b": [0, 1, 2]}
        with pytest.raises(ValueError, match="Cannot reshape"):
            reshape_to_coords(arr, coords)


class TestCastCoords:
    """Tests for the cast_coords function."""

    def test_int_strings(self):
        """Test that strings representing integers are cast to int."""
        d = {"idx": ["000", "001", "002"]}
        result = cast_coords(d)
        assert result["idx"].dtype == int
        np.testing.assert_array_equal(result["idx"], [0, 1, 2])

    def test_float_strings(self):
        """Test that strings representing floats are cast to float."""
        d = {"k": ["0.1", "0.2", "0.35"]}
        result = cast_coords(d)
        assert result["k"].dtype == float
        np.testing.assert_allclose(result["k"], [0.1, 0.2, 0.35])

    def test_non_numeric_strings(self):
        """Test that non-numeric strings are left as strings."""
        d = {"label": ["foo", "bar"]}
        result = cast_coords(d)
        assert result["label"].dtype.kind in ("U", "O")  # string or object

    def test_whole_floats_cast_to_int(self):
        """Test that strings representing whole-number floats are cast to int."""
        d = {"ells": ["0.0", "2.0", "4.0"]}
        result = cast_coords(d)
        assert result["ells"].dtype == int

    def test_float_not_rounded(self):
        """Test that strings representing non-whole floats are not rounded to int."""
        d = {"k": ["0.1", "0.15", "0.2"]}
        result = cast_coords(d)
        assert result["k"].dtype == float

    def test_mixed(self):
        """Test that each key is cast independently based on its values."""
        d = {"idx": ["000", "001"], "k": ["0.1", "0.2"], "label": ["foo", "bar"]}
        result = cast_coords(d)
        assert result["idx"].dtype == int
        assert result["k"].dtype == float
        assert result["label"].dtype.kind in ("U", "O")

    def test_empty(self):
        """Test that an empty dictionary is handled without error."""
        result = cast_coords({})
        assert result == {}


class TestReindexSamples:
    """Tests for the reindex_samples function."""

    def test_global(self):
        """Test global reindexing without grouping."""
        index_arrays = {
            "cosmo_idx": ["000", "000", "001", "001"],
            "hod_idx":   ["006", "008", "006", "010"],
        }
        result = reindex_samples(index_arrays, reindex=["hod_idx"])
        # Global ordering: 006->0, 008->1, 010->2
        assert result["hod_idx"] == [0, 1, 0, 2]
        assert result["cosmo_idx"] == ["000", "000", "001", "001"]  # unchanged

    def test_group_by(self):
        """Test reindexing within groups defined by other index arrays."""
        index_arrays = {
            "cosmo_idx": ["000", "000", "000", "001", "001", "001"],
            "hod_idx":   ["006", "008", "010", "008", "010", "014"],
        }
        result = reindex_samples(index_arrays, reindex=["hod_idx"], group_by=["cosmo_idx"])
        # Within each cosmo group, hod_idx is re-indexed from 0
        assert result["hod_idx"] == [0, 1, 2, 0, 1, 2]

    def test_missing_index_raises(self):
        """Test that a ValueError is raised if a reindex key is not found in the index arrays."""
        index_arrays = {"cosmo_idx": ["000", "001"]}
        with pytest.raises(ValueError, match="not found"):
            reindex_samples(index_arrays, reindex=["hod_idx"])

    def test_preserves_order(self):
        """Test that the re-indexing preserves the original order of samples within each group, not sorted order."""
        index_arrays = {
            "cosmo_idx": ["000", "000", "000"],
            "hod_idx":   ["010", "006", "008"],  # not sorted
        }
        result = reindex_samples(index_arrays, reindex=["hod_idx"], group_by=["cosmo_idx"])
        assert result["hod_idx"] == [0, 1, 2]

    def test_no_group_by_single_group(self):
        """Test reindexing without grouping when all samples belong to a single group."""
        index_arrays = {"hod_idx": ["006", "008", "010"]}
        result = reindex_samples(index_arrays, reindex=["hod_idx"])
        assert result["hod_idx"] == [0, 1, 2]

    def test_multiple_reindex(self):
        """Test reindexing multiple index arrays at once."""
        index_arrays = {
            "cosmo_idx": ["000", "001"],
            "hod_idx":   ["006", "008"],
            "phase_idx": ["000", "001"],
        }
        result = reindex_samples(index_arrays, reindex=["hod_idx", "phase_idx"])
        assert result["hod_idx"] == [0, 1]
        assert result["phase_idx"] == [0, 1]
        assert result["cosmo_idx"] == ["000", "001"]  # untouched

    def test_already_zero_indexed(self):
        """Test that if the index is already zero-indexed, it remains unchanged."""
        index_arrays = {"hod_idx": ["000", "001", "002"]}
        result = reindex_samples(index_arrays, reindex=["hod_idx"])
        assert result["hod_idx"] == [0, 1, 2]

    def test_multiple_group_by_keys(self):
        """Test grouping by multiple keys simultaneously."""
        index_arrays = {
            "cosmo_idx": ["000", "000", "001", "001"],
            "phase_idx": ["000", "000", "001", "001"],
            "hod_idx":   ["006", "008", "006", "010"],
        }
        result = reindex_samples(
            index_arrays, reindex=["hod_idx"], group_by=["cosmo_idx", "phase_idx"]
        )
        assert result["hod_idx"] == [0, 1, 0, 1]

# %% split_test_set

@pytest.fixture
def simple_dataset():
    """2D dataset with cosmo and hod dimensions."""
    x = xarray.DataArray(
        np.random.rand(3, 4),  # noqa: NPY002
        dims=["cosmo_idx", "hod_idx"],
        coords={"cosmo_idx": [0, 1, 2], "hod_idx": [0, 1, 2, 3]},
        name="x",
    )
    y = xarray.DataArray(
        np.random.rand(3, 4),  # noqa: NPY002
        dims=["cosmo_idx", "hod_idx"],
        coords={"cosmo_idx": [0, 1, 2], "hod_idx": [0, 1, 2, 3]},
        name="y",
    )
    return xarray.Dataset({"x": x, "y": y})

class TestSplitTestSet:
    """Tests for the split_test_set function."""

    def test_adds_test_train_variables(self, simple_dataset):
        """Test that the output contains x_test, x_train, y_test, y_train."""
        result = split_test_set(simple_dataset, filters={"cosmo_idx": [0, 1]})
        assert "x_test" in result
        assert "x_train" in result
        assert "y_test" in result
        assert "y_train" in result

    def test_nan_dims_attr(self, simple_dataset):
        """Test that the 'nan_dims' attribute is set correctly on the test and train variables."""
        result = split_test_set(simple_dataset, filters={"cosmo_idx": [0, 1]})
        assert result["x_test"].attrs["nan_dims"] == ["cosmo_idx"]
        assert result["x_train"].attrs["nan_dims"] == ["cosmo_idx"]

    def test_missing_variable_raises(self, simple_dataset):
        """Test that if a variable in to_split is missing from the dataset, a ValueError is raised."""
        with pytest.raises(ValueError, match="not found"):
            split_test_set(simple_dataset, filters={"cosmo_idx": [0]}, to_split=["z"])

    def test_custom_to_split(self, simple_dataset):
        """Test that only variables specified in to_split are split, and others are left unchanged."""
        result = split_test_set(simple_dataset, filters={"cosmo_idx": [0, 1]}, to_split=["x"])
        assert "x_test" in result
        assert "x_train" in result
        assert "y_test" not in result
        assert "y_train" not in result

    def test_preserves_original_variables(self, simple_dataset):
        """Test that the original variables are still present in the output dataset."""
        result = split_test_set(simple_dataset, filters={"cosmo_idx": [0, 1]})
        assert "x" in result
        assert "y" in result

# %% collect_measurements

@pytest.fixture
def mock_file_tree(tmp_path):
    """Create a minimal mock file tree and return the root path."""
    files = [
        "c000_ph000/seed0/hod006/power_spectrum_los_x.h5",
        "c000_ph000/seed0/hod006/power_spectrum_los_y.h5",
        "c000_ph000/seed0/hod008/power_spectrum_los_x.h5",
        "c000_ph000/seed0/hod008/power_spectrum_los_y.h5",
        "c001_ph000/seed0/hod006/power_spectrum_los_x.h5",
        "c001_ph000/seed0/hod006/power_spectrum_los_y.h5",
        "c001_ph000/seed0/hod008/power_spectrum_los_x.h5",
        "c001_ph000/seed0/hod008/power_spectrum_los_y.h5",
    ]
    for f in files:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return tmp_path


GLOB_PATTERN = "c{cosmo_idx}_ph{phase_idx}/seed{seed}/hod{hod_idx}/power_spectrum_los_{los}.h5"

class TestCollectMeasurements:
    """Tests for the collect_measurements function."""

    def test_groups_by_index(self, mock_file_tree):
        """Test that files are grouped by unique combinations of index values, ignoring specified indexes."""
        groups, _ = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        # 2*2 unique combinations of cosmo_idx and hod_idx, so 4 groups
        assert len(groups) == 4

    def test_ignored_index_files_are_grouped(self, mock_file_tree):
        """Test that files differing only in the ignored index (los) are grouped together."""
        groups, _ = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        # Each group should contain 2 files (los_x and los_y)
        assert all(len(files) == 2 for files in groups.values())

    def test_ignored_index_not_in_index_arrays(self, mock_file_tree):
        """Test that the ignored index is not included in the returned index arrays."""
        _, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        assert "los" not in index_arrays

    def test_index_arrays_aligned(self, mock_file_tree):
        """Test that the returned index arrays are aligned with the groups (same length and order)."""
        _, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        lengths = [len(v) for v in index_arrays.values()]
        assert len(set(lengths)) == 1  # all same length

    def test_correct_index_values(self, mock_file_tree):
        """Test that the index arrays contain the correct unique values from the file paths."""
        _, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        assert set(index_arrays["cosmo_idx"]) == {"000", "001"}
        assert set(index_arrays["hod_idx"]) == {"006", "008"}

    def test_no_ignore_index(self, mock_file_tree):
        """Test that if no indexes are ignored, each unique combination of all indexes is its own group."""
        groups, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN)
        # Without ignoring 'los', each (cosmo, phase, seed, hod, los) combo is its own group
        assert "los" in index_arrays
        assert all(len(files) == 1 for files in groups.values())

    def test_all_indexes_tracked(self, mock_file_tree):
        """Test that all tracked indexes exist in the index array."""
        _, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        assert set(index_arrays.keys()) == {"cosmo_idx", "phase_idx", "seed", "hod_idx"}

    def test_empty_dir(self, tmp_path):
        """Test that if the directory contains no matching files, the function returns empty groups and index arrays."""
        groups, index_arrays = collect_measurements(tmp_path, GLOB_PATTERN, ignore_index=["los"])
        assert len(groups) == 0
        assert all(len(v) == 0 for v in index_arrays.values())

    def test_sorted_files(self, mock_file_tree):
        """Test that the files within each group are sorted (for consistent ordering)."""
        groups, _ = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        for files in groups.values():
            assert files == sorted(files)

# %% compress_measurements

def _dummy_reader(files):
    """Return a simple sentinel object (just the count of files)."""
    return len(files)


def _dummy_postprocess(data, **kwargs):
    """Return a flat array of ones with shape (n_samples, n_features=2)."""
    arr = np.ones((len(data), 2))
    coords = {"feature": [0, 1]}
    return arr, coords

class TestCompressMeasurements:
    """Tests for the compress_measurements function."""

    def test_sparse_grid_raises(self, tmp_path):
        """Sparse grids (missing index combinations) should raise a ValueError."""
        files = [
            "c000_ph000/seed0/hod006/power_spectrum_los_x.h5",
            "c001_ph000/seed0/hod008/power_spectrum_los_x.h5",  # c001/hod006 and c000/hod008 are missing
        ]
        for f in files:
            p = tmp_path / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

        groups, index_arrays = collect_measurements(tmp_path, GLOB_PATTERN, ignore_index=["los"])
        with pytest.raises(ValueError, match="sparse"):
            compress_measurements(groups, index_arrays, reader=_dummy_reader, postprocess=_dummy_postprocess)

    def test_correct_shape(self, mock_file_tree):
        """Test that the output DataArray has the expected shape based on the number of unique index values and features."""
        groups, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los", "seed", "phase_idx"])
        result = compress_measurements(groups, index_arrays, reader=_dummy_reader, postprocess=_dummy_postprocess)
        # (n_hod=2, n_cosmo=2, n_features=2) with singleton dims dropped
        assert result.shape == (2, 2, 2)

    def test_reader_called_once_per_group(self, mock_file_tree):
        """Test that the reader function is called exactly once for each group of files, not once per file."""
        groups, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los", "seed", "phase_idx"])
        call_count = 0

        def counting_reader(files):
            nonlocal call_count
            call_count += 1
            return len(files)

        compress_measurements(groups, index_arrays, reader=counting_reader, postprocess=_dummy_postprocess)
        assert call_count == len(groups)

    def test_output_is_dataarray(self, mock_file_tree):
        """Test that the output of compress_measurements is an xarray DataArray."""
        groups, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        result = compress_measurements(
            groups, index_arrays,
            reader=_dummy_reader,
            postprocess=_dummy_postprocess,
        )
        assert isinstance(result, xarray.DataArray)

    def test_sample_dims_in_coords(self, mock_file_tree):
        """Test that the sample index arrays (e.g. cosmo_idx, hod_idx) are included in the output DataArray's coordinates."""
        groups, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        result = compress_measurements(
            groups, index_arrays,
            reader=_dummy_reader,
            postprocess=_dummy_postprocess,
        )
        for idx in ["cosmo_idx", "hod_idx"]:
            assert idx in result.coords

    def test_feature_dims_in_coords(self, mock_file_tree):
        """Test that the feature dimension from the postprocess output is included in the coordinates."""
        groups, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        result = compress_measurements(
            groups, index_arrays,
            reader=_dummy_reader,
            postprocess=_dummy_postprocess,
        )
        assert "feature" in result.coords

    def test_attrs(self, mock_file_tree):
        """Test that the output DataArray has a 'sample' attribute containing the sample index arrays, and a 'features' attribute describing the features."""
        groups, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        result = compress_measurements(
            groups, index_arrays,
            reader=_dummy_reader,
            postprocess=_dummy_postprocess,
        )
        assert "sample" in result.attrs
        assert "features" in result.attrs

    def test_with_reindex(self, mock_file_tree):
        """Test that when reindexing is applied, the sample coordinates are re-indexed to contiguous integers starting from 0 within each group."""
        groups, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        result = compress_measurements(
            groups, index_arrays,
            reindex=["hod_idx"],
            reindex_group_by=["cosmo_idx"],
            reader=_dummy_reader,
            postprocess=_dummy_postprocess,
        )
        # After reindexing, hod_idx coords should be contiguous integers from 0
        hod_coords = result.coords["hod_idx"].values
        assert set(hod_coords) == set(range(len(hod_coords)))

    def test_drop_singleton_dims(self, mock_file_tree):
        """Test that when drop_singleton_dims=True, any sample index dimensions that have only one unique value are dropped from the coordinates and attributes."""
        groups, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        result = compress_measurements(
            groups, index_arrays,
            drop_singleton_dims=True,
            reader=_dummy_reader,
            postprocess=_dummy_postprocess,
        )
        assert all(s > 1 for s in result.shape)
        assert "phase_idx" not in result.coords
        assert "seed" not in result.coords
        assert "phase_idx" not in result.attrs["sample"]
        assert "seed" not in result.attrs["sample"]

    def test_no_drop_singleton_dims(self, mock_file_tree):
        """Test that when drop_singleton_dims=False, sample index dimensions that have only one unique value are retained in the coordinates and attributes."""
        groups, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        result = compress_measurements(
            groups, index_arrays,
            drop_singleton_dims=False,
            reader=_dummy_reader,
            postprocess=_dummy_postprocess,
        )
        # Singleton dims (phase, seed) should still be present
        assert "phase_idx" in result.coords
        assert "seed" in result.coords
        assert "phase_idx" in result.attrs["sample"]
        assert "seed" in result.attrs["sample"]

    def test_data_values(self, mock_file_tree):
        """Test that the data values in the output DataArray match the expected output from the reader and postprocess functions."""
        groups, index_arrays = collect_measurements(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
        result = compress_measurements(
            groups, index_arrays,
            reader=_dummy_reader,
            postprocess=_dummy_postprocess,
        )
        np.testing.assert_array_equal(result.values, np.ones(result.shape))
