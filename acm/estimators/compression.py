"""
Compression module for estimator results.

This module provides classes and functions to compress estimator results stored as lsstypes objects.

Example
-------
>>> compressor = Compressor(
        root="path/to/data",
        pattern="I{indice_i}/J{indice_j}_K{indice_k}/L{indice_l}/",
    )
>>> group = compressor.read(reader=lsstypes.read, ignore_index=['indice_l'])
>>> group = group.merge(method=lsstypes.mean, **merge_kwargs)
>>> group = group.select(**rebin).select(**select).get(**get)
>>> result = Compressor.compress(
        data=group,
        order=['indice_i', 'indice_j', 'indice_k'],
        reindex={'indice_j': ['indice_i'], 'indice_k': ['indice_i', 'indice_j']},
        drop_single=True,
    )
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import lsstypes
import numpy as np
import xarray
from pandas import to_numeric

from acm.typing import LsstypeObject
from acm.utils.xarray import split_vars

logger = logging.getLogger(__name__)

PATTERNS = {
    "cpsh": r"c{cosmo_idx}_p{phase_idx}/seed{seed}/hod{hod_idx}/",
}


@dataclass
class Pattern:
    """
    Store a pattern with its root directory and provide methods to convert it to glob or regex.

    Attributes
    ----------
    root : Path
        The root directory where the files are located.
    pattern : str
        A pattern to match the file paths (file name not included) from the root directory.
        Expects a string with placeholders for indexes, extracted as indice names.

    Example
    -------
    >>> pattern = Pattern(root=Path("/data"), pattern="I{indice_i}/J{indice_j}_K{indice_k}/myfile.h5")
    >>> pattern.to_glob()
    'I*/J*_K*/myfile.h5'
    >>> pattern.to_regex()
    re.compile('I(?P<indice_i>[^/]+)/J(?P<indice_j>[^/]+)_K(?P<indice_k>[^/]+)/myfile.h5')
    """

    root: Path
    pattern: str

    @property
    def names(self) -> list[str]:
        """Extract index names from the string pattern."""
        return re.findall(r"\{(\w+)\}", self.pattern)

    def to_glob(self) -> str:
        """Convert the pattern to a glob pattern."""
        glob_pattern = self.pattern
        index_names = self.names
        for index in index_names:
            glob_pattern = glob_pattern.replace(f"{{{index}}}", "*")
        # Avoid reccursive glob patterns for adjacent braces
        glob_pattern = glob_pattern.replace("**", "*")
        return glob_pattern

    def to_regex(self, ignore_index: list[str] | None = None) -> re.Pattern:
        """Convert the pattern to a regex pattern."""
        index_names = self.names
        ignore_index = ignore_index or []  # Ensure list behavior
        re_pattern = re.escape(str(self.root / self.pattern))
        for index in index_names:
            placeholder = re.escape(f"{{{index}}}")
            if index in ignore_index:
                # Non-capturing group for ignored indexes
                re_pattern = re_pattern.replace(placeholder, r"[^/]+", 1)
            else:
                # Named capture group for known indexes
                re_pattern = re_pattern.replace(placeholder, f"(?P<{index}>[^/]+)", 1)
        re_pattern = re_pattern.replace(r"\*", r"[^/]+")  # Allow for wildcard matching
        return re.compile(re_pattern)


@dataclass
class IndexedObject:
    """Index a LsstypeObject with arbitrary indices, stored as key-value pairs."""

    indexes: dict
    data: LsstypeObject

    @property
    def names(self) -> list[str]:
        """Return the names of the indexes."""
        return list(self.indexes)


@dataclass
class ObjectGroup:
    """Store IndexedObject element with common names."""

    objects: list[IndexedObject] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        """Get the name of the objects indices."""
        if len(self.objects) == 0:
            raise ValueError("No object stored, can't infer names.")
        return self.objects[0].names

    def __repr__(self) -> str:
        """Return a string representation of the ObjectGroup."""
        r = "ObjectGroup("
        if len(self.objects) > 0:
            r += f"n_objects={len(self.objects)}"
            r += f", names={self.names}"
        r += ")"
        return r

    def _match_names(self, other: IndexedObject) -> None:
        if len(self.objects) != 0 and self.names != other.names:
            raise ValueError(
                f"Expected matching indexes, got {other.names} not {self.names}."
            )

    def __post_init__(self) -> None:
        """Post-initialization check on objects."""
        for o in self.objects:
            self._match_names(o)

    def __len__(self) -> int:
        """Return the number of objects in the group."""
        return len(self.objects)

    def __getitem__(self, index: int) -> IndexedObject:
        """Get the IndexedObject at the specified index."""
        return self.objects[index]

    def append(self, other: IndexedObject) -> None:
        """Append an instance of IndexedObject to the objects property if the index keys match the existing objects ones."""
        self._match_names(other)
        self.objects.append(other)

    def get_idx(self, **indexes) -> list[LsstypeObject]:
        """Get the data of objects matching the required indexes."""

        def _match_indexes(obj: IndexedObject, **idx) -> bool:
            return all(obj.indexes.get(k) == v for k, v in idx.items())

        return [o.data for o in self.objects if _match_indexes(o, **indexes)]

    def merge(
        self,
        method: Callable[[list[LsstypeObject]], LsstypeObject],
        **kwargs,
    ) -> "ObjectGroup":
        """
        Merge objects having the same indices.

        Parameters
        ----------
        method: Callable[[list[LsstypeObject]], LsstypeObject]
            Method to use to merge the different LsstypeObject with the same indices.
        **kwargs
            Extra arguments to pass to `method`

        Returns
        -------
        ObjectGroup:
            New instance of the class, with unique indices and merged data lsstypes objects.
        """
        new = ObjectGroup()
        for o in self.objects:
            if len(new.get_idx(**o.indexes)) == 0:  # Indexes not already used
                match = self.get_idx(**o.indexes)
                logger.debug(f"Found {len(match)} objects matching {o.indexes}")
                new.append(IndexedObject(o.indexes, method(match, **kwargs)))
        return new

    def __getattr__(self, name: str) -> Callable[..., "ObjectGroup"]:
        """
        Proxy attribute access to data objects, returning a callable that applies the named method to all objects in the group.

        Returns
        -------
        Callable
            A function that accepts arbitrary arguments, applies
            ``data.<name>(*args, **kwargs)`` on the first object,
            then propagates the result shape to all objects via :func:`lsstypes.match`.
            Returns a new ObjectGroup.

        Raises
        ------
        AttributeError
            If the ObjectGroup is empty or if the named attribute does not exist in the data objects.

        Example
        -------
        >>> new_group = group.select(k_min=0.01, k_max=0.5)  # calls .select() on all data objects
        """
        if name == "objects":  # Safeguard for calls before __init__ is complete
            raise AttributeError(name)

        if not self.objects:
            raise AttributeError(
                f"'ObjectGroup' has no attribute '{name}' because it is empty."
            )

        if not hasattr(self.objects[0].data, name):
            raise AttributeError(f"'{name}' not found in data objects.")

        method = getattr(self.objects[0].data, name)
        if not callable(method):
            raise TypeError(f"Attribute '{name}' of data object is not callable.")

        def _apply(*args, **kwargs) -> "ObjectGroup":
            # Call the method on the first data object to check for errors
            d0 = method(*args, **kwargs)
            new = ObjectGroup()
            for obj in self.objects:
                # Match the output of the first data object
                new.objects.append(IndexedObject(obj.indexes, obj.data.match(d0)))
            return new

        return _apply

    def sort(self, *names: str) -> "ObjectGroup":
        """
        Order the objects in the group based on the specified index names.

        Parameters
        ----------
        *names : str
            The names of the indexes to order by.
            If no names are provided, the original order is preserved.
            Names not present in the indexes of the objects will be ignored.

        Returns
        -------
        ObjectGroup
            A new ObjectGroup instance with the objects ordered based on the specified index names.

        Raises
        ------
        ValueError
            If multiple objects are found for the same index values, indicating non-unique indexes.

        Example
        -------
        >>> group = ObjectGroup(objects=[IndexedObject(indexes={'i': 1, 'j': 2}, data=data1),
        ...                              IndexedObject(indexes={'i': 0, 'j': 1}, data=data2)])
        >>> ordered_group = group.sort('i', 'j')  # Orders by 'i' first, then 'j'
        >>> print(ordered_group.objects)
        [IndexedObject(indexes={'i': 0, 'j': 1}, data=data2), IndexedObject(indexes={'i': 1, 'j': 2}, data=data1)]
        """
        if not names:
            return self  # No ordering specified, return the original group

        def _sort_key(obj: IndexedObject) -> tuple:
            """Generate a sorting key based on the specified index names."""
            _names = [name for name in names if name in obj.indexes]
            if len(self.get_idx(**obj.indexes)) > 1:
                raise ValueError(
                    f"Multiple objects found for indexes {obj.indexes}."
                    "Cannot order by non-unique indexes."
                )
            # Downcast to ensure consistent sorting behavior across different data types
            out = downcast(np.array([obj.indexes[name] for name in _names]))
            return tuple(out)

        ordered_objects = sorted(self.objects, key=_sort_key)
        return ObjectGroup(objects=ordered_objects)

    def get_index_lists(
        self,
        *names: str,
        **reindex: list[str],
    ) -> dict[str, list]:
        """
        Get the index values as lists for each index name.

        Parameters
        ----------
        *names : str
            The names of the indexes to retrieve values for.
            Defaults to all index names if not provided.
        **reindex : list[str]]
            Key-value pairs of index names to reindex. The key is the name of the index
            to reindex, and the value is a list of index names to group by for reindexing.

        Returns
        -------
        dict[str, list]
            A dictionary where keys are index names and values are lists of index values
            corresponding to each object in the group.

        Raises
        ------
        KeyError
            If reindexing is specified for an unknown index name.

        Example
        -------
        >>> group = ObjectGroup(objects=[IndexedObject(indexes={'i': 0, 'j': 5}, data=data1),
        ...                              IndexedObject(indexes={'i': 1, 'j': 3}, data=data2)])
        >>> group.get_index_lists()
        {'i': [0, 1], 'j': [5, 3]}
        >>> group.get_index_lists('i', 'j', reindex={'j': ['i']})
        {'i': [0, 1], 'j': [0, 0]}  # 'j' is reindexed based on 'i'
        """
        names = names or tuple(self.names)
        index_lists = {name: [o.indexes[name] for o in self.objects] for name in names}

        unknown = set(reindex) - set(index_lists)
        if unknown:
            raise KeyError(f"Reindexing specified for unknown index names: {unknown}")

        n = len(self.objects)
        for name in names:
            if name not in reindex:
                continue

            group_names = reindex[name]  # Which indexes to group by for reindexing
            group_map: dict[tuple, dict] = {}
            new_values = []

            for row in range(n):
                group = tuple(index_lists[prev_name][row] for prev_name in group_names)
                local = group_map.setdefault(group, {})
                # Assign a new integer to each unique raw value within this group
                new_values.append(local.setdefault(index_lists[name][row], len(local)))
            index_lists[name] = new_values

        return index_lists


class Compressor:
    """Compression class for estimator results."""

    extension: str = ".h5"  # The expected file extension for the input files.

    def __init__(
        self,
        root: str | Path | None = None,
        pattern: str | None = None,
    ) -> None:
        """
        Initialize the Compressor class.

        Parameters
        ----------
        root : str | Path | None
            The root directory where the files are located.
            If None, the current working directory is used.
        pattern : str | None
            A pattern to match the file paths (file name not included) from the root.
            Expects a string with placeholders for indexes, extracted as indice names.
        """
        if root is None:
            root = Path.cwd()
        root = Path(root)
        pattern = pattern or f"*{self.extension}"

        logger.info(f"Initializing Compressor with root: {root} and pattern: {pattern}")

        self._pattern = Pattern(root=root, pattern=pattern)

        # Register files matching the pattern in the root directory.
        self._files = sorted(root.glob(self._pattern.to_glob()))

        logger.info(f"Found {len(self._files)} files matching the pattern.")

    def read(
        self,
        reader: Callable[[Path], LsstypeObject],
        ignore_index: list[str] | None = None,
        **kwargs,
    ) -> ObjectGroup:
        """
        Read objects from files, and store them with the recovered indices.

        Parameters
        ----------
        reader: Callable[[Path], LsstypeObject]
            A function that reads a file and returns an LsstypeObject.
        ignore_index: list[str] | None
            A list of index names to ignore when matching files.
            If None, all indexes are considered.
        **kwargs : dict
            Additional keyword arguments to pass to the reader function.

        Returns
        -------
        ObjectGroup
            An ObjectGroup containing IndexedObjects.
            The IndexedObjects will all have the same indexes for a given pattern.
        """
        cout = ObjectGroup()
        re_pattern = self._pattern.to_regex(ignore_index=ignore_index)
        for f in self._files:
            match = re_pattern.match(str(f))
            if match:
                index_values = match.groupdict()
                logger.debug(f"Reading file: {f} with index values: {index_values}")
                data = reader(f, **kwargs)
                cout.append(IndexedObject(index_values, data))
            else:
                logger.warning(
                    f"File {f} does not match the pattern and will be skipped."
                )
        return cout

    @staticmethod
    def compress(
        data: ObjectGroup,
        order: list[str] | None = None,
        reindex: dict[str, list[str]] | None = None,
        attrs: list[str] | None = None,
        drop_single: bool = True,
    ) -> xarray.DataArray:
        """
        Compress an ObjectGroup instance in a xarray DataArray.

        If attrs is provided, it will extract the attributes from the data objects instead of the data itself.
        The resulting feature dimension will be named "parameters" and will contain the provided attribute names.

        Parameters
        ----------
        data: ObjectGroup
            The ObjectGroup instance to compress.
        order: list[str], optional
            The order of the indexes the objects should be sorted by.
            Changes the order of the dimensions in the resulting DataArray if all indexes are provided
            (otherwise, it is not possible to know the correct order of the dimensions).
            If None, the original order of the objects is preserved.
        reindex: dict[str, list[str]], optional
            A dictionary specifying how to reindex the indexes.
            See :meth:`ObjectGroup.get_index_lists` for details.
        attrs: list[str], optional
            A list of attribute names to extract from the data objects.
            The resulting DataArray will have a single feature dimension named "parameters"
            containing these attributes in the provided order.
        drop_single: bool, optional
            Whether to drop singleton dimensions in the resulting DataArray.

        Returns
        -------
        xarray.DataArray
            A DataArray containing the compressed data from the ObjectGroup.
            The dimensions correspond to the unique values of the indexes and the features of the data objects.
            The attributes "sample" and "features" are added to indicate which dimensions correspond to sample indexes and feature coordinates, respectively.

        Raises
        ------
        ValueError
            If sample coordinates and feature coordinates have overlapping names.
            If the resulting array cannot be reshaped to the expected shape based on the provided coordinates.
        """
        order = order or []
        reindex = reindex or {}

        data = data.sort(*order)  # Ensure ordering and uniqueness of indexes
        index_lists = data.get_index_lists(**reindex)  # ordered + reindexed
        if set(order) == set(data.names):
            index_lists = {k: index_lists[k] for k in order}  # Order indexes
            logger.info(f"Ordering indexes as specified: {order}")
        sample_coords = {idx: np.unique(values) for idx, values in index_lists.items()}

        if attrs is not None:
            features_coords = {"parameters": np.array(attrs)}
            result = np.asarray(
                [o.data.attrs.get(attr) for o in data.objects for attr in attrs]
            )
        else:
            # Unflattened labels from the first data object, assuming all objects have the same structure.
            object_data = data.objects[0].data
            if isinstance(object_data, lsstypes.ObservableTree):
                _tmp = {
                    **object_data.labels(return_type="unflatten", level=None),
                    **object_data.flatten(level=None)[0].coords(),
                }  # extract labels and coordinates
            else:  # lsstypes.ObservableLeaf
                _tmp = object_data.coords()  # access coordinates only
            features_coords = {k: np.unique(v) for k, v in _tmp.items()}
            result = np.asarray([o.data for o in data.objects])

        if set(sample_coords) & set(features_coords):
            raise ValueError(
                "Sample coordinates and feature coordinates have overlapping names."
                f" Sample: {list(sample_coords)}, Features: {list(features_coords)}"
            )

        coords = {**sample_coords, **features_coords}
        coords = {k: downcast(v) for k, v in coords.items()}
        logger.debug(f"Coordinates for xarray DataArray: {coords}")

        shape = [len(v) for v in coords.values()]
        if result.size != np.prod(shape):
            raise ValueError(
                f"Cannot reshape array of size {result.size} to shape {shape} based on provided coordinates."
            )
        logger.debug(f"Reshaping result array of size {result.size} to shape {shape}")

        cout = xarray.DataArray(
            data=result.reshape(shape),
            coords=coords,
        )

        if drop_single:
            singleton_dims = [dim for dim in cout.dims if cout.sizes[dim] == 1]
            logger.debug(f"Dropping singleton dimensions: {singleton_dims}")
            cout = cout.squeeze(drop=True)

        cout.attrs = {
            "sample": [s for s in sample_coords if s in cout.dims],
            "features": [s for s in features_coords if s in cout.dims],
        }  # Assign attrs here to avoid singleton dimension in attrs

        return cout


def downcast(array: np.ndarray) -> np.ndarray:
    """
    Downcast a numpy array to the smallest possible dtype without losing information.

    Converts strings arrays and returns the original array if
    downcasting is not possible (ValueError raised).
    """
    try:
        array = to_numeric(array, errors="raise")
    except ValueError:
        pass
    return array


def split_test_set(
    ds: xarray.Dataset,
    filters: dict,
    to_split: list[str] | None = None,
) -> xarray.Dataset:
    """
    Split DataArrays into test/train sets based on filters and merge into a single Dataset.

    Based on split_vars. "in" matches the filters and is suffixed with "_test",
    "out" is the complementary subset suffixed with "_train".

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset containing the variables to split.
        Must contain all variables listed in to_split (or defaulting to "x" and "y").
    filters : dict
        Dictionary of dimension names and values to filter the DataArrays
        (see "in" variables in split_vars).
    to_split : list[str], optional
        List of variable names in the dataset to apply the split to. If None,
        defaults to ``x`` and ``y`` if they exist in the dataset.

    Returns
    -------
    xarray.Dataset
        Split dataset with filtered variables. New data variables have a ``nan_dims``
        attribute listing the dimensions that were filtered out and filled with NaNs
        in the complementary variable.
    """
    to_split = to_split or ["x", "y"]
    for v in to_split:
        if v not in ds:
            raise ValueError(
                f"Variable '{v}' not found in dataset. Available variables: {list(ds.data_vars)}"
            )

    logger.debug(f"Splitting variables: {to_split} with filters: {filters}")

    data_vars = [ds[v] for v in to_split]
    for v_in, v_out in split_vars(*data_vars, **filters):
        # NOTE: cast to str to avoid type issues
        v_in.name = str(v_in.name) + "_test"
        v_out.name = str(v_out.name) + "_train"

        # Mark filtered dimensions that will be filled with NaNs
        v_in.attrs["nan_dims"] = list(filters)
        v_out.attrs["nan_dims"] = list(filters)

        ds = xarray.merge([ds, v_in, v_out], join="outer")
    return ds
