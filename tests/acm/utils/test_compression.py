import numpy as np
import pytest
import xarray

from acm.utils.compression import (
    cast_coords,
    collect_mocks,
    compress_mocks,
    reindex_samples,
    split_test_set,
    reshape_to_coords,
)

# %% reshape_to_coords

def test_reshape_to_coords_basic():
    arr = np.arange(6)
    coords = {"a": [0, 1], "b": [0, 1, 2]}
    result = reshape_to_coords(arr, coords)
    assert result.shape == (2, 3)

def test_reshape_to_coords_single_dim():
    arr = np.arange(3)
    coords = {"a": [0, 1, 2]}
    result = reshape_to_coords(arr, coords)
    assert result.shape == (3,)

def test_reshape_to_coords_mismatch_raises():
    arr = np.arange(5)
    coords = {"a": [0, 1], "b": [0, 1, 2]}
    with pytest.raises(ValueError, match="Cannot reshape"):
        reshape_to_coords(arr, coords)


# %% cast_coords

def test_cast_coords_int_strings():
    d = {"idx": ["000", "001", "002"]}
    result = cast_coords(d)
    assert result["idx"].dtype == int
    np.testing.assert_array_equal(result["idx"], [0, 1, 2])


def test_cast_coords_float_strings():
    d = {"k": ["0.1", "0.2", "0.35"]}
    result = cast_coords(d)
    assert result["k"].dtype == float
    np.testing.assert_allclose(result["k"], [0.1, 0.2, 0.35])


def test_cast_coords_non_numeric_strings():
    d = {"label": ["foo", "bar"]}
    result = cast_coords(d)
    assert result["label"].dtype.kind in ("U", "O")  # string or object


def test_cast_coords_whole_floats_cast_to_int():
    # Whole-number floats should be downcast to int
    d = {"ells": ["0.0", "2.0", "4.0"]}
    result = cast_coords(d)
    assert result["ells"].dtype == int


def test_cast_coords_float_not_rounded():
    # Non-whole floats must not be rounded to int
    d = {"k": ["0.1", "0.15", "0.2"]}
    result = cast_coords(d)
    assert result["k"].dtype == float

def test_cast_coords_mixed():
    # Each key is cast independently
    d = {"idx": ["000", "001"], "k": ["0.1", "0.2"], "label": ["foo", "bar"]}
    result = cast_coords(d)
    assert result["idx"].dtype == int
    assert result["k"].dtype == float
    assert result["label"].dtype.kind in ("U", "O")


def test_cast_coords_empty():
    result = cast_coords({})
    assert result == {}

# %% reindex_samples

def test_reindex_samples_global():
    index_arrays = {
        "cosmo_idx": ["000", "000", "001", "001"],
        "hod_idx":   ["006", "008", "006", "010"],
    }
    result = reindex_samples(index_arrays, reindex=["hod_idx"])
    # Global ordering: 006->0, 008->1, 010->2
    assert result["hod_idx"] == [0, 1, 0, 2]
    assert result["cosmo_idx"] == ["000", "000", "001", "001"]  # unchanged


def test_reindex_samples_group_by():
    index_arrays = {
        "cosmo_idx": ["000", "000", "000", "001", "001", "001"],
        "hod_idx":   ["006", "008", "010", "008", "010", "014"],
    }
    result = reindex_samples(index_arrays, reindex=["hod_idx"], group_by=["cosmo_idx"])
    # Within each cosmo group, hod_idx is re-indexed from 0
    assert result["hod_idx"] == [0, 1, 2, 0, 1, 2]


def test_reindex_samples_missing_index_raises():
    index_arrays = {"cosmo_idx": ["000", "001"]}
    with pytest.raises(ValueError, match="not found"):
        reindex_samples(index_arrays, reindex=["hod_idx"])


def test_reindex_samples_preserves_order():
    # Re-indexing should follow insertion order, not sorted order
    index_arrays = {
        "cosmo_idx": ["000", "000", "000"],
        "hod_idx":   ["010", "006", "008"],  # not sorted
    }
    result = reindex_samples(index_arrays, reindex=["hod_idx"], group_by=["cosmo_idx"])
    assert result["hod_idx"] == [0, 1, 2]

def test_reindex_samples_no_group_by_single_group():
    index_arrays = {"hod_idx": ["006", "008", "010"]}
    result = reindex_samples(index_arrays, reindex=["hod_idx"])
    assert result["hod_idx"] == [0, 1, 2]


def test_reindex_samples_multiple_reindex():
    index_arrays = {
        "cosmo_idx": ["000", "001"],
        "hod_idx":   ["006", "008"],
        "phase_idx": ["000", "001"],
    }
    result = reindex_samples(index_arrays, reindex=["hod_idx", "phase_idx"])
    assert result["hod_idx"] == [0, 1]
    assert result["phase_idx"] == [0, 1]
    assert result["cosmo_idx"] == ["000", "001"]  # untouched


def test_reindex_samples_already_zero_indexed():
    index_arrays = {"hod_idx": ["000", "001", "002"]}
    result = reindex_samples(index_arrays, reindex=["hod_idx"])
    assert result["hod_idx"] == [0, 1, 2]


def test_reindex_samples_multiple_group_by_keys():
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

@pytest.fixture()
def simple_dataset():
    """A simple 2D dataset with cosmo and hod dimensions."""
    x = xarray.DataArray(
        np.random.rand(3, 4),
        dims=["cosmo_idx", "hod_idx"],
        coords={"cosmo_idx": [0, 1, 2], "hod_idx": [0, 1, 2, 3]},
        name="x",
    )
    y = xarray.DataArray(
        np.random.rand(3, 4),
        dims=["cosmo_idx", "hod_idx"],
        coords={"cosmo_idx": [0, 1, 2], "hod_idx": [0, 1, 2, 3]},
        name="y",
    )
    return xarray.Dataset({"x": x, "y": y})


def test_split_test_set_adds_test_train_variables(simple_dataset):
    result = split_test_set(simple_dataset, filters={"cosmo_idx": [0, 1]})
    assert "x_test" in result
    assert "x_train" in result
    assert "y_test" in result
    assert "y_train" in result


def test_split_test_set_nan_dims_attr(simple_dataset):
    result = split_test_set(simple_dataset, filters={"cosmo_idx": [0, 1]})
    assert result["x_test"].attrs["nan_dims"] == ["cosmo_idx"]
    assert result["x_train"].attrs["nan_dims"] == ["cosmo_idx"]


def test_split_test_set_missing_variable_raises(simple_dataset):
    with pytest.raises(ValueError, match="not found"):
        split_test_set(simple_dataset, filters={"cosmo_idx": [0]}, to_split=["z"])


def test_split_test_set_custom_to_split(simple_dataset):
    result = split_test_set(simple_dataset, filters={"cosmo_idx": [0, 1]}, to_split=["x"])
    assert "x_test" in result
    assert "x_train" in result
    assert "y_test" not in result
    assert "y_train" not in result


def test_split_test_set_preserves_original_variables(simple_dataset):
    result = split_test_set(simple_dataset, filters={"cosmo_idx": [0, 1]})
    assert "x" in result
    assert "y" in result

# %% collect_mocks

@pytest.fixture()
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


def test_collect_mocks_groups_by_index(mock_file_tree):
    groups, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    # 2*2 unique combinations of cosmo_idx and hod_idx, so 4 groups
    assert len(groups) == 4


def test_collect_mocks_ignored_index_files_are_grouped(mock_file_tree):
    groups, _ = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    # Each group should contain 2 files (los_x and los_y)
    assert all(len(files) == 2 for files in groups.values())


def test_collect_mocks_ignored_index_not_in_index_arrays(mock_file_tree):
    _, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    assert "los" not in index_arrays


def test_collect_mocks_index_arrays_aligned(mock_file_tree):
    _, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    lengths = [len(v) for v in index_arrays.values()]
    assert len(set(lengths)) == 1  # all same length


def test_collect_mocks_correct_index_values(mock_file_tree):
    _, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    assert set(index_arrays["cosmo_idx"]) == {"000", "001"}
    assert set(index_arrays["hod_idx"]) == {"006", "008"}

def test_collect_mocks_no_ignore_index(mock_file_tree):
    groups, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN)
    # Without ignoring 'los', each (cosmo, phase, seed, hod, los) combo is its own group
    assert "los" in index_arrays
    assert all(len(files) == 1 for files in groups.values())


def test_collect_mocks_all_indexes_tracked(mock_file_tree):
    _, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    assert set(index_arrays.keys()) == {"cosmo_idx", "phase_idx", "seed", "hod_idx"}


def test_collect_mocks_empty_dir(tmp_path):
    groups, index_arrays = collect_mocks(tmp_path, GLOB_PATTERN, ignore_index=["los"])
    assert len(groups) == 0
    assert all(len(v) == 0 for v in index_arrays.values())


def test_collect_mocks_sorted_files(mock_file_tree):
    # Files within each group should be sorted
    groups, _ = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    for files in groups.values():
        assert files == sorted(files)

# %% compress_mocks

def _dummy_reader(files):
    """Returns a simple sentinel object (just the count of files)."""
    return len(files)


def _dummy_postprocess(data, **kwargs):
    """Returns a flat array of ones with shape (n_samples, n_features=2)."""
    arr = np.ones((len(data), 2))
    coords = {"feature": [0, 1]}
    return arr, coords

def test_compress_mocks_sparse_grid_raises(tmp_path):
    """Sparse grids (missing index combinations) should raise a ValueError."""
    files = [
        "c000_ph000/seed0/hod006/power_spectrum_los_x.h5",
        "c001_ph000/seed0/hod008/power_spectrum_los_x.h5",  # c001/hod006 and c000/hod008 are missing
    ]
    for f in files:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    groups, index_arrays = collect_mocks(tmp_path, GLOB_PATTERN, ignore_index=["los"])
    with pytest.raises(ValueError, match="sparse"):
        compress_mocks(groups, index_arrays, reader=_dummy_reader, postprocess=_dummy_postprocess)

def test_compress_mocks_correct_shape(mock_file_tree):
    groups, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los", "seed", "phase_idx"])
    result = compress_mocks(groups, index_arrays, reader=_dummy_reader, postprocess=_dummy_postprocess)
    # (n_hod=2, n_cosmo=2, n_features=2) with singleton dims dropped
    assert result.shape == (2, 2, 2)


def test_compress_mocks_reader_called_once_per_group(mock_file_tree):
    groups, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los", "seed", "phase_idx"])
    call_count = 0

    def counting_reader(files):
        nonlocal call_count
        call_count += 1
        return len(files)

    compress_mocks(groups, index_arrays, reader=counting_reader, postprocess=_dummy_postprocess)
    assert call_count == len(groups)

def test_compress_mocks_output_is_dataarray(mock_file_tree):
    groups, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    result = compress_mocks(
        groups, index_arrays,
        reader=_dummy_reader,
        postprocess=_dummy_postprocess,
    )
    assert isinstance(result, xarray.DataArray)


def test_compress_mocks_sample_dims_in_coords(mock_file_tree):
    groups, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    result = compress_mocks(
        groups, index_arrays,
        reader=_dummy_reader,
        postprocess=_dummy_postprocess,
    )
    for idx in ["cosmo_idx", "hod_idx"]:
        assert idx in result.coords


def test_compress_mocks_feature_dims_in_coords(mock_file_tree):
    groups, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    result = compress_mocks(
        groups, index_arrays,
        reader=_dummy_reader,
        postprocess=_dummy_postprocess,
    )
    assert "feature" in result.coords


def test_compress_mocks_attrs(mock_file_tree):
    groups, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    result = compress_mocks(
        groups, index_arrays,
        reader=_dummy_reader,
        postprocess=_dummy_postprocess,
    )
    assert "sample" in result.attrs
    assert "features" in result.attrs


def test_compress_mocks_with_reindex(mock_file_tree):
    groups, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    result = compress_mocks(
        groups, index_arrays,
        reindex=["hod_idx"],
        reindex_group_by=["cosmo_idx"],
        reader=_dummy_reader,
        postprocess=_dummy_postprocess,
    )
    # After reindexing, hod_idx coords should be contiguous integers from 0
    hod_coords = result.coords["hod_idx"].values
    assert set(hod_coords) == set(range(len(hod_coords)))


def test_compress_mocks_drop_singleton_dims(mock_file_tree):
    groups, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    result = compress_mocks(
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

def test_compress_mocks_no_drop_singleton_dims(mock_file_tree):
    groups, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    result = compress_mocks(
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

def test_compress_mocks_data_values(mock_file_tree):
    groups, index_arrays = collect_mocks(mock_file_tree, GLOB_PATTERN, ignore_index=["los"])
    result = compress_mocks(
        groups, index_arrays,
        reader=_dummy_reader,
        postprocess=_dummy_postprocess,
    )
    np.testing.assert_array_equal(result.values, np.ones(result.shape))