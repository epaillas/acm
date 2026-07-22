import itertools
import logging
import re
from pathlib import Path
from unittest.mock import patch

import lsstypes
import numpy as np
import pytest
import xarray

from acm.estimators.compression import (
    Compressor,
    IndexedObject,
    ObjectGroup,
    Pattern,
    downcast,
    split_test_set,
)

# ruff: noqa: ANN001, ANN201, ARG002, D102, INP001, S101

def make_lsstype_object() -> lsstypes.ObservableTree:
    """Create a mock lsstypes.ObservableTree object for testing. Borrowed from lsstypes notebook examples."""
    s = np.linspace(0., 200., 51)
    mu = np.linspace(-1., 1., 101)
    rng = np.random.RandomState(seed=42)
    labels = ['DD', 'DR', 'RR']
    leaves = []
    for _label in labels:
        counts = 1. + rng.uniform(size=(s.size, mu.size))
        leaves.append(lsstypes.ObservableLeaf(
            counts=counts,
            s=s,
            mu=mu,
            coords=['s', 'mu'],
            attrs=dict(los='x'),
        ))
    tree = lsstypes.ObservableTree(leaves, pairs=labels)
    return tree

@pytest.fixture
def object_group() -> ObjectGroup:
    """Fixture for creating an ObjectGroup instance."""
    # Create a simple ObjectGroup with mock objects for testing
    obj1 = IndexedObject(indexes={"i": 1, "j": 2}, data=make_lsstype_object())
    obj2 = IndexedObject(indexes={"i": 2, "j": 3}, data=make_lsstype_object())
    return ObjectGroup(objects=[obj1, obj2])

class TestPattern:
    """Test the Pattern dataclass."""

    ROOT = Path("/data")
    PAT = "I{i}/J{j}_K{k}/file.h5"

    @pytest.fixture
    def pattern(self) -> Pattern:
        return Pattern(self.ROOT, self.PAT)

    def test_names(self, pattern):
        """Test that the names property returns the correct index names in order."""
        assert pattern.names == ["i", "j", "k"]

    def test_to_glob(self, pattern):
        assert pattern.to_glob() == "I*/J*_K*/file.h5"

    def test_to_glob_with_adjacent_braces(self):
        """Avoid recursive glob patterns for adjacent braces."""
        pattern = Pattern(self.ROOT, "I{i}J{j}K{k}{l}/file.h5")
        assert pattern.to_glob() == "I*J*K*/file.h5"

    def test_to_regex_captures_all_groups(self, pattern):
        m = pattern.to_regex().match(str(self.ROOT / "I1/J2_K3/file.h5"))
        if m:
            assert m.groupdict() == {"i": "1", "j": "2", "k": "3"}
        else:
            pytest.fail("Regex did not match the expected path.")

    def test_to_regex_ignore_index_non_capturing(self, pattern):
        m = pattern.to_regex(ignore_index=["k"]).match(str(self.ROOT / "I1/J2_K3/file.h5"))
        if m:
            assert "k" not in m.groupdict()
        else:
            pytest.fail("Regex did not match the expected path.")

    def test_to_regex_raises_for_identical_index_names(self):
        """Test that to_regex raises a ValueError when identical index names are present in the pattern, unless ignored."""
        pattern = Pattern(self.ROOT, "I{i}/J{i}_K{k}/file.h5")
        with pytest.raises(re.error, match="redefinition of group name 'i'"):
            pattern.to_regex()
        assert pattern.to_regex(ignore_index=["i"]).match(str(self.ROOT / "I1/J2_K3/file.h5")) is not None

    def test_to_regex_escapes_special_characters(self):
        """Test that special characters in the root and pattern are properly escaped in the regex."""
        root = Path("/data.( 3 )") # Special characters in the root path
        special_pattern = Pattern(root, "I{i}/J{j}_K{k}/file[1].h5")
        regex = special_pattern.to_regex()
        assert regex.match(str(root / "I1/J2_K3/file[1].h5")) is not None
        assert regex.match(str(root / "I1/J2_K3/file[2].h5")) is None

    def test_to_regex_all_ignored(self):
        """Test that all index names can be ignored, resulting in a non-capturing regex."""
        pattern = Pattern(self.ROOT, "I{i}/J{j}_K{k}/file.h5")
        regex = pattern.to_regex(ignore_index=["i", "j", "k"])
        m = regex.match(str(self.ROOT / "I1/J2_K3/file.h5"))
        if m:
            assert m.groupdict() == {}
        else:
            pytest.fail("Regex did not match the expected path.")

    def test_to_regex_no_match_returns_none(self, pattern):
        assert pattern.to_regex().match("/wrong/path/file.h5") is None

    def test_empty_pattern(self):
        """Edge case: Empty pattern."""
        pattern = Pattern(self.ROOT, "")
        assert pattern.names == []
        assert pattern.to_glob() == ""
        assert pattern.to_regex().pattern == re.escape(str(self.ROOT))


# NOTE: not testing IndexedObject here because it is a simple dataclass and has no logic to test


class TestObjectGroup:
    """Test the ObjectGroup class."""

    def test_init_empty(self):
        group = ObjectGroup()
        assert group.objects == []
        with pytest.raises(ValueError, match="No object stored"):
            _ = group.names

    def test_init_with_objects(self):
        obj1 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 2}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        assert len(group.objects) == 2
        assert obj1 in group.objects
        assert obj2 in group.objects

    def test_init_with_mismatched_keys_raises(self):
        """NOTE: Also tests :func:`~ObjectGroup._match_names`."""
        obj1 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"j": 2}, data=make_lsstype_object())
        with pytest.raises(ValueError, match="Expected matching indexes"):
            ObjectGroup(objects=[obj1, obj2])

    def test_repr(self):
        obj1 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 2}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        repr_str = repr(group)
        assert "ObjectGroup" in repr_str
        assert str(group.names) in repr_str
        assert str(len(group)) in repr_str

    def test_repr_empty(self):
        group = ObjectGroup()
        repr_str = repr(group)
        assert repr_str == "ObjectGroup()"

    def test_append(self):
        obj1 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 2}, data=make_lsstype_object())
        group = ObjectGroup()
        group.append(obj1)
        group.append(obj2)
        assert len(group.objects) == 2
        assert obj1 in group.objects
        assert obj2 in group.objects

    def test_append_unordered_keys_raises(self):
        obj1 = IndexedObject(indexes={"i": 1, "j": 2}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"j": 3, "i": 4}, data=make_lsstype_object())
        group = ObjectGroup()
        group.append(obj1)
        with pytest.raises(ValueError, match="Expected matching indexes"):
            group.append(obj2)

    def test_append_mismatched_keys_raises(self):
        obj1 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"j": 2}, data=make_lsstype_object())
        group = ObjectGroup()
        group.append(obj1)
        with pytest.raises(ValueError, match="Expected matching indexes"):
            group.append(obj2)

    def test_len(self):
        obj1 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 2}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        assert len(group) == 2

    def test_getitem(self):
        obj1 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 2}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        assert group[0] == obj1
        assert group[1] == obj2
        with pytest.raises(IndexError):
            _ = group[2]  # Out of bounds

    def test_get(self):
        obj1 = IndexedObject(indexes={"i": 1, "j": 2}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 2, "j": 3}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        result = group.get(i=1)
        assert len(result) == 1
        assert result == [obj1.data]

    def test_get_no_args(self):
        obj1 = IndexedObject(indexes={"i": 1, "j": 2}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 2, "j": 3}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        result = group.get()
        assert result == [o.data for o in group.objects] # All objects data

    def test_get_no_match_returns_empty(self):
        obj1 = IndexedObject(indexes={"i": 1, "j": 2}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 2, "j": 3}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        result = group.get(i=3)
        assert result == []

    def test_get_multiple_matches(self):
        obj1 = IndexedObject(indexes={"i": 1, "j": 2}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 1, "j": 3}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        result = group.get(i=1)
        assert len(result) == 2
        assert obj1.data in result
        assert obj2.data in result

    def test_get_multiple_keys(self):
        """Multiple keys in get should apply AND logic."""
        obj1 = IndexedObject(indexes={"i": 1, "j": 2}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 1, "j": 3}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        result = group.get(i=1, j=2)
        assert len(result) == 1
        assert result == [obj1.data]

    def test_get_unknown_key_returns_empty(self):
        obj1 = IndexedObject(indexes={"i": 1, "j": 2}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1])
        result = group.get(k=3)  # 'k' is not a valid index name
        assert result == []

    def test_merge(self):
        """Test the merge method of ObjectGroup, including kwargs forwarding."""
        obj1 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        # Result must have the same indexes as the duplicate objects!
        result = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        with patch("lsstypes.mean", return_value=result.data) as mock_mean:
            merged_group = group.merge(method=mock_mean, arg='somearg')
            mock_mean.assert_called_with([obj1.data, obj2.data], arg='somearg')
            assert mock_mean.call_count == 1 # One duplicate object
        assert len(merged_group) == 1
        assert merged_group.objects == [result]

    def test_merge_no_duplicates(self):
        """Test that merge does not change the group when there are no duplicates."""
        obj1 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 2}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        merged_group = group.merge(method=lsstypes.mean)
        assert len(merged_group) == 2
        assert merged_group.objects == [obj1, obj2] # No duplicates, so unchanged

    def test_merge_with_duplicates_and_singles(self):
        """Test that merge correctly merges duplicates and leaves single objects unchanged."""
        obj1 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj3 = IndexedObject(indexes={"i": 2}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2, obj3])
        merged_group = group.merge(method=lsstypes.mean)
        merged_obj = IndexedObject(indexes={"i": 1}, data=lsstypes.mean([obj1.data, obj2.data]))
        assert merged_obj in merged_group.objects
        assert obj3 in merged_group.objects # lsstypes.mean should not change obj3 since it's unique
        assert len(merged_group) == 2
        assert any(obj.indexes["i"] == 1 for obj in merged_group.objects)
        assert any(obj.indexes["i"] == 2 for obj in merged_group.objects)

    def test_getattr_guard_raises(self):
        """Test that accessing 'objects' before __init__ is complete raises AttributeError."""
        group = ObjectGroup.__new__(ObjectGroup)  # Create an instance without calling __init__
        with pytest.raises(AttributeError, match="objects"):
            _ = group.objects

    def test_getattr_empty_group_raises(self):
        """Test that accessing any attribute on an empty ObjectGroup raises AttributeError."""
        group = ObjectGroup()
        with pytest.raises(AttributeError, match="empty"):
            _ = group.select(k_min=0.01, k_max=0.5)  # calls .select() on all data objects

    def test_getattr_unknown_method(self, object_group):
        """Test that accessing an unknown method raises AttributeError."""
        with pytest.raises(AttributeError, match="not found in data objects"):
            _ = object_group.unknown_method()

    def test_getattr_non_callable_attribute(self, object_group):
        """Test that accessing a non-callable attribute raises AttributeError."""
        with pytest.raises(TypeError, match="not callable"):
            _ = object_group.attrs

    def test_getattr(self, object_group):
        """Test getattr default behavior."""
        assert callable(object_group.select)  # Should return a callable
        # Do we have to mock each data object method separately ?
        patch1 = patch.object(object_group.objects[0].data, "select", wraps=object_group.objects[0].data.select)
        patch2 = patch.object(object_group.objects[1].data, "match", wraps=object_group.objects[1].data.match)
        with patch1 as mock_select, patch2 as mock_match:
            result_group = object_group.select(s=slice(0, 100, 1))
            mock_select.assert_called_once_with(s=slice(0, 100, 1)) # Kwargs are passed to the first object's method
            mock_match.assert_called_once_with(object_group.objects[0].data) # Match is used for subsequent objects
        assert isinstance(result_group, ObjectGroup)
        assert len(result_group) == len(object_group)  # Should return the same number of objects

    def test_getattr_fail_on_first_object(self, object_group):
        """Test that if the method fails on the first object, it raises an error."""
        p = patch.object(object_group.objects[0].data, "select", side_effect=ValueError("Test error"))
        with p as mock_select:
            with pytest.raises(ValueError, match="Test error"):
                _ = object_group.select(s=slice(0, 100, 1))
            mock_select.assert_called_once()  # Ensure the method was only called on the first object

    def test_getattr_fail_on_subsequent_object(self, object_group):
        """Test that if the method fails on a subsequent object, it raises an error."""
        p1 = patch.object(object_group.objects[0].data, "select", wraps=object_group.objects[0].data.select)
        p2 = patch.object(object_group.objects[1].data, "match", side_effect=ValueError("Test error"))
        with p1 as mock_select, p2 as mock_match:
            with pytest.raises(ValueError, match="Test error"):
                _ = object_group.select(s=slice(0, 100, 1))
            mock_select.assert_called_once()  # Ensure the method was called on the first object
            mock_match.assert_called_once()  # Ensure the method was called on the second object

    def test_sort(self, object_group):
        """Test the sort method."""
        # Create a new ObjectGroup with known indexes
        obj1 = IndexedObject(indexes={"i": 2}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        ordered_group = group.sort("i")
        assert ordered_group.objects[0].indexes["i"] == 1
        assert ordered_group.objects[1].indexes["i"] == 2

    def test_sort_no_names(self):
        obj1 = IndexedObject(indexes={"i": 2}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        ordered_group = group.sort()
        assert ordered_group == group  # Should return the same group if no names are provided

    def test_sort_multiple_keys(self):
        """Test the sort method with multiple keys."""
        obj1 = IndexedObject(indexes={"i": 2, "j": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 1, "j": 2}, data=make_lsstype_object())
        obj3 = IndexedObject(indexes={"i": 1, "j": 1}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2, obj3])
        ordered_group = group.sort("i", "j")
        assert ordered_group.objects[0].indexes == {"i": 1, "j": 1}
        assert ordered_group.objects[1].indexes == {"i": 1, "j": 2}
        assert ordered_group.objects[2].indexes == {"i": 2, "j": 1}

        ordered_group_desc = group.sort("j", "i")
        assert ordered_group_desc.objects[0].indexes == {"i": 1, "j": 1}
        assert ordered_group_desc.objects[1].indexes == {"i": 2, "j": 1}
        assert ordered_group_desc.objects[2].indexes == {"i": 1, "j": 2}

    def test_sort_cast_str_indexes(self):
        """Test that sort can internally cast index values from strings to integers for sorting."""
        obj1 = IndexedObject(indexes={"i": "10"}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": "2"}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        ordered_group = group.sort("i")
        assert ordered_group.objects[0].indexes["i"] == "2"
        assert ordered_group.objects[1].indexes["i"] == "10"

    def test_sort_mixed_type_indexes(self):
        """Test that sort can handle mixed types (int and str) in index values."""
        obj1 = IndexedObject(indexes={"i": 1, "j": "2"}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": "2", "j": 1}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        ordered_group = group.sort("i", "j")
        assert ordered_group.objects[0].indexes == {"i": 1, "j": "2"}
        assert ordered_group.objects[1].indexes == {"i": "2", "j": 1}

    def test_sort_empty_group(self):
        """Test that sort on an empty ObjectGroup returns an empty ObjectGroup."""
        group = ObjectGroup()
        ordered_group = group.sort("i")
        assert isinstance(ordered_group, ObjectGroup)
        assert len(ordered_group) == 0

    def test_sort_str_indexes(self):
        """Test that sort can handle actual string indexes."""
        obj1 = IndexedObject(indexes={"i": "one"}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": "two"}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        ordered_group = group.sort("i")
        assert ordered_group.objects[0].indexes["i"] == "one"
        assert ordered_group.objects[1].indexes["i"] == "two"

    def test_sort_with_nonexistent_key(self):
        """Test that sort silently ignores keys that do not exist in the indexes."""
        obj1 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 2}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        ordered_group = group.sort("nonexistent_key")
        assert ordered_group == group  # Should return the same group if the key is not found

    def test_sort_with_duplicate_indexes_raises(self):
        """Test that sort raises an error if there are duplicate index values for the sorting keys."""
        obj1 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 1}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        with pytest.raises(ValueError, match="Multiple objects found"):
            group.sort("i")

    def test_get_index_lists(self):
        """Test the get_index_lists method of ObjectGroup."""
        obj1 = IndexedObject(indexes={"i": 1, "j": 2}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 2, "j": 3}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        # No names should return all index names
        index_lists = group.get_index_lists()
        assert index_lists == {"i": [1, 2], "j": [2, 3]}

        # Specifying names should return only those index names
        index_lists_i = group.get_index_lists("i")
        assert index_lists_i == {"i": [1, 2]}

    def test_get_index_lists_with_nonexistent_name_raises(self, object_group):
        """Test that get_index_lists raises a KeyError for a nonexistent index name."""
        with pytest.raises(KeyError):
            object_group.get_index_lists("nonexistent")

    def test_get_index_lists_preserves_order(self):
        """Test that get_index_lists preserves the order of index values as they appear in the group."""
        obj1 = IndexedObject(indexes={"i": 2, "j": 3}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 1, "j": 4}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        index_lists = group.get_index_lists()
        assert index_lists["i"] == [2, 1]
        assert index_lists["j"] == [3, 4]

    def test_get_index_lists_with_mixed_types(self):
        """Test that get_index_lists can handle mixed types in index values."""
        obj1 = IndexedObject(indexes={"i": 1, "j": "a"}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": "2", "j": "b"}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2])
        index_lists = group.get_index_lists()
        assert index_lists["i"] == [1, "2"]
        assert index_lists["j"] == ["a", "b"]

    def test_get_index_lists_reindex(self, object_group):
        """Test that get_index_lists can reindex the group based on specified index values."""
        index_lists = object_group.get_index_lists(j=["i"])
        assert index_lists == {"i": [1, 2], "j": [0, 0]}

    def test_get_index_lists_reindex_with_incorrect_key_raises(self, object_group):
        """Test that get_index_lists raises a KeyError when reindexing with or for an incorrect key."""
        with pytest.raises(KeyError):
            object_group.get_index_lists(j=["value"]) # Unknown index name "value" should raise KeyError
        with pytest.raises(KeyError):
            object_group.get_index_lists(nonexistent=["i"]) # Unknown index name "nonexistent" should raise KeyError

    def test_get_index_lists_reindex_with_coumpound_keys(self):
        """Test that get_index_lists can reindex an index based on compound keys."""
        obj1 = IndexedObject(indexes={"i": 1, "j": 2, "k": 6}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 4, "j": 5, "k": 4}, data=make_lsstype_object())
        obj3 = IndexedObject(indexes={"i": 1, "j": 2, "k": 6}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2, obj3])
        index_lists = group.get_index_lists(i=["j", "k"])
        assert index_lists == {"i": [0, 0, 0], "j": [2, 5, 2], "k": [6, 4, 6]}

        index_lists = group.get_index_lists(i=["k", "j"])
        assert index_lists == {"i": [0, 0, 0], "j": [2, 5, 2], "k": [6, 4, 6]}

    def test_get_index_lists_reindex_chained(self):
        """Test that get_index_lists can be chained to reindex multiple times."""
        obj1 = IndexedObject(indexes={"i": 1, "j": 2, "k": 6}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 4, "j": 5, "k": 4}, data=make_lsstype_object())
        obj3 = IndexedObject(indexes={"i": 1, "j": 2, "k": 6}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1, obj2, obj3])
        index_lists = group.get_index_lists(i=["j", "k"], j=["k"])
        assert index_lists == {"i": [0, 0, 0], "j": [0, 0, 0], "k": [6, 4, 6]}

        # NOTE: this reindexing makes no sense but tests the limits of the functionality
        index_lists = group.get_index_lists(i=["j", "k"], j=["k"], k=["i", "j"])
        assert index_lists == {"i": [0, 0, 0], "j": [0, 0, 0], "k": [0, 1, 0]}


@patch("lsstypes.read", return_value=make_lsstype_object())
class TestCompressor:
    """Test the Compressor class for compressing estimator measurements."""

    @pytest.fixture
    def make_files(self, tmp_path):
        """Create a set of mock files in the temporary directory for testing."""
        # Create directories and files based on the pattern
        for i, j, k in itertools.product(range(2), range(2), range(1, 4)):
            dir_path = tmp_path / f"I{i}/J{j}_K{k}"
            dir_path.mkdir(parents=True, exist_ok=True)
            file_path = dir_path / "file.h5"
            file_path.touch()  # Create an empty file
            other_path = dir_path / "other_file.txt"
            other_path.touch()  # Create a non-matching file

    @pytest.fixture
    def compressor(self, make_files, tmp_path):
        """Fixture to create a Compressor instance with the mock files."""
        return Compressor(root=tmp_path, pattern="I{i}/J{j}_K{k}/file.h5")

    def test_init_no_params(self, reader):
        """Test that the Compressor initializes with default root (cwd) and pattern (*.h5)."""
        compressor = Compressor()
        assert compressor._pattern.root == Path.cwd()
        assert compressor._pattern.pattern == "*.h5"

    def test_init_with_params(self, reader):
        """Test that the Compressor initializes with specified root and pattern."""
        root_path = Path("/data")
        pattern = "I{i}/J{j}/file.h5"
        compressor = Compressor(root=root_path, pattern=pattern)
        assert compressor._pattern.root == root_path
        assert compressor._pattern.pattern == pattern

    def test_init_no_macthing_files(self, reader, tmp_path):
        """Test that the Compressor raises a ValueError when no files match the pattern."""
        compressor = Compressor(root=tmp_path, pattern="I{i}/J{j}/file.h5")
        assert compressor._files == []  # No files should match

    def test_init_sorted_files(self, reader, compressor, tmp_path):
        """Test that the Compressor sorts the files based on the pattern."""
        expected_files = sorted(tmp_path.glob("I*/J*_K*/file.h5"))
        assert compressor._files == expected_files
        assert len(compressor._files) == 12  # 2*2*3 combinations of i, j, k

    def test_read(self, reader, compressor):
        result = compressor.read(reader=reader, arg='somearg')
        # Should call lsstypes.read for each file
        assert reader.call_count == len(compressor._files)
        assert len(result) == len(compressor._files)

    def test_read_forward_kwargs(self, reader, compressor):
        """Test that read forwards kwargs to the reader function."""
        _ = compressor.read(reader=reader, arg='somearg')
        for call in reader.call_args_list:
            assert call.kwargs.get('arg') == 'somearg'  # Check that 'arg' is forwarded

    def test_read_does_not_handle_duplicates(self, reader, compressor):
        """Test that read does not handle duplicate files."""
        # add a duplicate file to the compressor's file list
        duplicate_file = compressor._files[0]
        compressor._files.append(duplicate_file)
        result = compressor.read(reader=reader)
        # Should call lsstypes.read for each file, including the duplicate
        assert reader.call_count == len(compressor._files)
        assert len(result) == len(compressor._files)
        assert result[0] == result[-1]  # The first and last results should be the same due to the duplicate file
        assert len(result.get(i="0", j="0", k="1")) == 2  # There should be two objects for the duplicate file indices

    def test_read_raises_on_reader_error(self, reader, compressor):
        """Test that read raises an error if the reader function raises an error."""
        patch_read = patch("lsstypes.read", side_effect=ValueError("Test error"))
        with patch_read as _reader, pytest.raises(ValueError, match="Test error"):
            _ = compressor.read(reader=_reader)
        assert _reader.call_count == 1  # Should only call the reader once before raising the error

    def test_read_empty_file_list(self, reader):
        """Test that read raises an error if the compressor has no files to read."""
        compressor = Compressor(root=Path.cwd(), pattern="nonexistent_pattern")
        result = compressor.read(reader=reader)
        assert len(result) == 0  # Should return an empty list since there are no files to read

    def test_read_ignore_index(self, reader, compressor):
        """Test that read ignores specified index names when reading files."""
        result = compressor.read(reader=reader, ignore_index=["j"])
        # Should call lsstypes.read for each file
        assert reader.call_count == len(compressor._files)
        assert "j" not in result.names

    def test_read_logs_on_unmatching_file(self, reader, compressor, caplog):
        """Test that logs a warning if a file does not match the pattern."""
        # Add a non-matching file to the compressor's file list
        non_matching_file = compressor._pattern.root / "I0/J0_K0/other_file.txt"
        compressor._files.append(non_matching_file)
        with caplog.at_level(logging.WARNING):
            _ = compressor.read(reader=reader)
        # Check that a warning was logged for the non-matching file
        n_warn = sum(1 for record in caplog.records if "does not match the pattern" in record.message)
        assert n_warn == 1  # One file only should trigger one warning in this test

    def test_compress(self, reader, compressor):
        """Test that compress returns a DataArray with the correct number of objects."""
        data = compressor.read(reader=reader)
        result = compressor.compress(data=data)
        assert isinstance(result, xarray.DataArray)
        assert result.shape == (2, 2, 3, 3, 51, 101)
        assert result.dims == ("i", "j", "k", "pairs", "s", "mu")
        assert result.coords["k"].values.tolist() == [1, 2, 3] # This one should be as provided
        assert result.attrs["sample"] == ["i", "j", "k"]
        assert result.attrs["features"] == ["pairs", "s", "mu"]

        # Check that the index types are correctly downcasted - in this case to int64
        assert all(result.coords[dim].dtype == np.int64 for dim in ["i", "j", "k"])

        # Check that the first object matches the compressed data
        compressed_vals = result.sel(i=0, j=0, k=1).values
        original_vals = np.stack(list(data.get(i="0", j="0", k="1")[0].flatten()))
        assert np.array_equal(compressed_vals, original_vals)

    def test_compress_raise_on_incorrect_reshaping(self, reader, compressor):
        """Test that compress raises a ValueError if the data cannot be reshaped correctly."""
        # Add a duplicate file to the compressor's file list
        # this should give the data a bigger size than inferred from the coordinates
        duplicate_file = compressor._files[0]
        compressor._files.append(duplicate_file)
        data = compressor.read(reader=reader)
        with pytest.raises(ValueError, match="Cannot reshape array of size"):
            _ = compressor.compress(data=data)

    def test_compress_and_full_order(self, reader, compressor):
        """Test that compress with full ordering returns a DataArray with ordered dimensions."""
        data = compressor.read(reader=reader)
        result = compressor.compress(data=data, order=["k", "j", "i"])
        assert isinstance(result, xarray.DataArray)
        assert result.shape == (3, 2, 2, 3, 51, 101)
        assert result.dims == ("k", "j", "i", "pairs", "s", "mu")
        assert result.attrs["sample"] == ["k", "j", "i"]

    def test_compress_and_partial_order(self, reader, compressor):
        """Test that compress with partial ordering returns a DataArray with no reordered dimensions."""
        data = compressor.read(reader=reader)
        result = compressor.compress(data=data, order=["k", "i"])
        assert isinstance(result, xarray.DataArray)
        assert result.shape == (2, 2, 3, 3, 51, 101)
        assert result.dims == ("i", "j", "k", "pairs", "s", "mu")
        assert result.attrs["sample"] == ["i", "j", "k"]

    def test_compress_and_reindex(self, reader, compressor):
        """Test that compress with reindexing returns a DataArray with the specified index values."""
        data = compressor.read(reader=reader)
        result = compressor.compress(data=data, reindex={"k": ["i", "j"]})
        assert isinstance(result, xarray.DataArray)
        assert result.shape == (2, 2, 3, 3, 51, 101)
        assert result.dims == ("i", "j", "k", "pairs", "s", "mu")
        assert result.attrs["sample"] == ["i", "j", "k"]
        assert result.coords["k"].values.tolist() == [0, 1, 2]  # Reindexed to the order of ["i", "j"]
        # FIXME: maybe there is a test for a more exotic reindexing case?

    def test_compress_forwards_order_and_reindex(self, reader, compressor):
        """Test that compress forwards arguments to the underlying data object."""
        data = compressor.read(reader=reader)
        p1 = patch.object(data, "sort", wraps=data.sort)
        p2 = patch.object(data, "get_index_lists", wraps=data.get_index_lists)

        # Separate 2 cases because the p2 patch won't work as data is reassigned when ordering is applied.
        with p1 as mock_sort, p2 as mock_get_index_lists:
            _ = compressor.compress(data=data, order=["k", "j", "i"])
            mock_sort.assert_called_once_with("k", "j", "i")
            mock_sort.reset_mock()  # Reset mock to avoid interference with the next test

            _ = compressor.compress(data=data, reindex={"k": ["i", "j"]})
            mock_sort.assert_called_once_with()  # Should return self - allowing the next patch to work
            mock_get_index_lists.assert_called_once_with(k=["i", "j"])  # Ensure reindexing was called with the correct arguments

    def test_compress_and_order_and_reindex(self, reader, compressor):
        """Test that compress with ordering and reindexing returns a DataArray with the specified order and index values."""
        data = compressor.read(reader=reader)
        result = compressor.compress(data=data, order=["k", "j", "i"], reindex={"k": ["i", "j"]})
        assert isinstance(result, xarray.DataArray)
        assert result.shape == (3, 2, 2, 3, 51, 101)
        assert result.dims == ("k", "j", "i", "pairs", "s", "mu")
        assert result.attrs["sample"] == ["k", "j", "i"]
        assert result.coords["k"].values.tolist() == [0, 1, 2]  # Reindexed to the order of ["i", "j"]

    @pytest.fixture
    def dummy_data(self):
        """Create a dummy ObjectGroup with mixed index types for testing."""
        obj1 = IndexedObject(indexes={"i": 1, "j": "a"}, data=make_lsstype_object())
        obj2 = IndexedObject(indexes={"i": 1, "j": "b"}, data=make_lsstype_object())
        obj3 = IndexedObject(indexes={"i": 1, "j": "c"}, data=make_lsstype_object())
        return ObjectGroup(objects=[obj1, obj2, obj3])

    def test_compress_mixed_index_types(self, reader, dummy_data):
        """Test that compress can handle mixed index types (int and str) and correctly downcast."""
        result = Compressor.compress(data=dummy_data, drop_single=False)
        assert isinstance(result, xarray.DataArray)
        assert result.shape == (1, 3, 3, 51, 101)  # 1 unique 'i' value and 3 unique 'j' values
        assert result.dims == ("i", "j", "pairs", "s", "mu")
        assert result.attrs["sample"] == ["i", "j"]
        assert result.attrs["features"] == ["pairs", "s", "mu"]
        # Check that the index types are correctly downcasted
        assert result.coords["i"].dtype == np.int64
        assert result.coords["j"].dtype.type is np.str_


    def test_compress_drop_single(self, reader, dummy_data):
        """Test that compress with drop_single=True drops dimensions with a single unique value."""
        result = Compressor.compress(data=dummy_data, drop_single=True)
        assert isinstance(result, xarray.DataArray)
        assert result.shape == (3, 3, 51, 101)  # 'i' dimension dropped
        assert "i" not in result.dims
        assert "i" not in result.attrs["sample"]

        result_no_drop = Compressor.compress(data=dummy_data, drop_single=False)
        assert result_no_drop.shape == (1, 3, 3, 51, 101)  # 'i' dimension retained
        assert "i" in result_no_drop.dims
        assert "i" in result_no_drop.attrs["sample"]

    def test_compress_drop_single_for_one_object(self, reader):
        """Test that compress with drop_single=True works correctly for a single object in the group."""
        obj = IndexedObject(indexes={"i": 1, "j": "a"}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj])
        result = Compressor.compress(data=group, drop_single=True)
        assert isinstance(result, xarray.DataArray)
        assert result.shape == (3, 51, 101)  # Both 'i' and 'j' dimensions dropped
        assert "i" not in result.dims
        assert "j" not in result.dims
        assert "i" not in result.attrs["sample"]
        assert "j" not in result.attrs["sample"]

    def test_compress_clashing_coords_raises(self, reader):
        """Test that compress raises a ValueError when there are clashing coordinates (e.g. features and sample have a similar coordinate name)."""
        obj1 = IndexedObject(indexes={"i": 1, "s": "a"}, data=make_lsstype_object())
        group = ObjectGroup(objects=[obj1])
        with pytest.raises(ValueError, match="Sample coordinates and feature coordinates have overlapping names"):
            _ = Compressor.compress(data=group)


class TestDowncast:
    """Test the downcast function for converting numpy arrays to lower precision types."""

    def test_str_to_int(self):
        arr = np.array(["1", "2", "3"])
        downcasted = downcast(arr)
        assert downcasted.dtype == np.int64

    def test_str_to_float(self):
        arr = np.array(["1.0", "2.5", "3.14"])
        downcasted = downcast(arr)
        assert downcasted.dtype == np.float64

    def test_pure_strings_unchanged(self):
        arr = np.array(["a", "b", "c"])
        downcasted = downcast(arr)
        assert np.array_equal(downcasted, arr)  # Should remain unchanged

    def test_float_unchanged(self):
        arr = np.array([1.0, 2.5, 3.14])
        downcasted = downcast(arr)
        assert downcasted.dtype == np.float64  # Should remain unchanged
        assert np.array_equal(downcasted, arr)  # Values should remain the same

    def test_int_unchanged(self):
        arr = np.array([1, 2, 3])
        downcasted = downcast(arr)
        assert downcasted.dtype == np.int64  # Should remain unchanged
        assert np.array_equal(downcasted, arr)  # Values should remain the same

    def test_mixed_types(self):
        arr = np.array([1, 2.5, 3])
        downcasted = downcast(arr)
        assert downcasted.dtype == np.float64  # Should be upcast to float

    def test_mixed_types_with_str_unchanged(self):
        arr = np.array([1, 2.5, "a"])
        downcasted = downcast(arr)
        assert np.array_equal(downcasted, arr)  # Should remain unchanged

    def test_empty_array(self):
        arr = np.array([])
        downcasted = downcast(arr)
        assert np.array_equal(downcasted, arr)  # Should remain unchanged


@pytest.fixture
def simple_dataset():
    """2D dataset with 2 dimensions."""
    rng = np.random.RandomState(seed=42)
    x = xarray.DataArray(
        rng.rand(3, 4),
        dims=["i", "j"],
        coords={"i": [0, 1, 2], "j": [0, 1, 2, 3]},
        name="x",
    )
    y = xarray.DataArray(
        rng.rand(3, 4),
        dims=["i", "j"],
        coords={"i": [0, 1, 2], "j": [0, 1, 2, 3]},
        name="y",
    )
    return xarray.Dataset({"x": x, "y": y})

class TestSplitTestSet:
    """Tests for the split_test_set function."""

    def test_adds_test_train_variables(self, simple_dataset):
        """Test that the output contains x_test, x_train, y_test, y_train."""
        result = split_test_set(simple_dataset, filters={"i": [0, 1]})
        assert "x_test" in result
        assert "x_train" in result
        assert "y_test" in result
        assert "y_train" in result

    def test_nan_dims_attr(self, simple_dataset):
        """Test that the 'nan_dims' attribute is set correctly on the test and train variables."""
        filters = {"i": [0, 1]}
        result = split_test_set(simple_dataset, filters=filters)
        assert result["x_test"].attrs["nan_dims"] == list(filters)
        assert result["x_train"].attrs["nan_dims"] == list(filters)

    def test_missing_variable_raises(self, simple_dataset):
        """Test that if a variable in to_split is missing from the dataset, a ValueError is raised."""
        with pytest.raises(ValueError, match="not found"):
            split_test_set(simple_dataset, filters={"i": [0]}, to_split=["z"])

    def test_custom_to_split(self, simple_dataset):
        """Test that only variables specified in to_split are split, and others are left unchanged."""
        result = split_test_set(simple_dataset, filters={"i": [0, 1]}, to_split=["x"])
        assert "x_test" in result
        assert "x_train" in result
        assert "y_test" not in result
        assert "y_train" not in result

    def test_preserves_original_variables(self, simple_dataset):
        """Test that the original variables are still present in the output dataset."""
        result = split_test_set(simple_dataset, filters={"i": [0, 1]})
        assert "x" in result
        assert "y" in result

    def test_complementary_splits(self, simple_dataset):
        """Test that the test and train splits are complementary and cover all data."""
        filters = {"i": [0, 1]}
        result = split_test_set(simple_dataset, filters=filters)
        x_test = result["x_test"].dropna(dim="i", how="all")
        x_train = result["x_train"].dropna(dim="i", how="all")
        # Check that the union of test and train covers all original data
        combined = xarray.concat([x_test, x_train], dim="i")
        assert np.array_equal(combined.values, simple_dataset["x"].values)


#%% NOTE: Full black-box test to move somewhere else eventually ?
@pytest.fixture
def make_files(tmp_path):
    """Create a set of mock files in the temporary directory for testing."""
    # Create directories and files based on the pattern
    for i, j, k in itertools.product(range(2), range(2), range(1, 4)):
        dir_path = tmp_path / f"I{i}/J{j}_K{k}_LL"
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / "file_M.h5"
        file_path.touch()  # Create an empty file

@patch("lsstypes.read", return_value=make_lsstype_object())
def test_full_chain(reader, tmp_path, make_files):  # noqa: ARG001
    """Test the full chain of reading, merging, selecting, and compressing data using the Compressor class."""
    pattern = "I{i}/J{j}_K{k}_L{l}/file_{m}.h5"

    compressor = Compressor(tmp_path, pattern)
    group = compressor.read(reader=reader, ignore_index=["m"])
    group = group.merge(method=lsstypes.mean)
    group = group.select(s=(0, 50))
    result = Compressor.compress(
        data=group,
        order=["i", "j", "k"],
        reindex={"j": ["i"], "k": ["i", "j"]},
        drop_single=True,
    )

    assert isinstance(result, xarray.DataArray)
    assert result.dims == ("i", "j", "k", "pairs", "s", "mu")
    assert result.shape == (2, 2, 3, 3, 13, 101) # droped 'l' dimension + selected 13 s values
    assert result.attrs["sample"] == ["i", "j", "k"]
    assert result.attrs["features"] == ["pairs", "s", "mu"]
    assert result.coords["j"].values.tolist() == [0, 1]  # Reindexed to the order of ["i"]
    assert result.coords["k"].values.tolist() == [0, 1, 2]  # Reindexed to the order of ["i", "j"]
