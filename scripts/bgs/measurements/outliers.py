import argparse  # noqa: INP001
import itertools
from collections.abc import Callable
from pathlib import Path

import lsstypes
import numpy as np
import yaml
from astropy.stats import sigma_clip
from compress_files import select
from measure_box import get_estimator

from acm.estimators.compression import Compressor, ObjectGroup, downcast
from acm.utils.logging import get_logger_for_script, setup_logging
from acm.utils.scripts import NumpyLoader

logger = get_logger_for_script(__name__)

def check_corrupted_files(
    compressor: Compressor,
    reader: Callable,
    **kwargs,
) -> np.ndarray:
    """Check for corrupted files in all the compressed data files."""
    missing_idx = []
    re_pattern = compressor._pattern.to_regex()
    for f in compressor._files:
        match = re_pattern.match(str(f))
        if match:
            index_values = match.groupdict()
            try:
                reader(f, **kwargs)
            except OSError as e:
                logger.warning(f"Error occurred while reading {f}: {e}")
                missing_idx.append(list(index_values.values()))
    return downcast(np.array(missing_idx))


def check_outliers(group: ObjectGroup, *names: str, **kwargs) -> np.ndarray:
    """Check for outliers in the compressed data."""
    data = np.asarray([obj.data for obj in group]) # flattened data array
    logger.debug(f"Data shape for outlier detection: {data.shape}")
    clipped_data = sigma_clip(data, masked=True, return_bounds=False, **kwargs)
    mask = clipped_data.mask.any(axis=1) # True for outlier samples  # ty:ignore[unresolved-attribute]
    logger.info(f"Detected {mask.sum()} outliers out of {len(group)} samples.")
    index_lists = group.get_index_lists(*names) # str types
    index_values = [downcast(np.array(v))[mask] for v in index_lists.values()]
    return np.array(list(zip(*index_values, strict=True)))

# NOTE: impossible to find missing files in folders with sparse indexing.
# Previously, we could check for unexpected empty folders, which is no longer possible as empty folders can now exist.
# This method returns re-indexed missing indexes which is not the same as the original missing indexes in the file system.
# We can't use this to know which files are missing :/
def check_missing_files(
    group: ObjectGroup,
    n_expected: int | None = None,
    reindex: dict[str, list[str]] | None = None,
    **expected: list,
) -> np.ndarray:
    """Check for missing files in the compressed data."""
    reindex = reindex or {}

    nfiles = len(group)
    names = list(expected.keys())
    if n_expected is not None and n_expected != nfiles:
        logger.warning(f"Expected {n_expected} files, but found only {nfiles}.")

    index_lists = group.get_index_lists(*names, **reindex) # str types
    index_values = [downcast(np.array(v)) for v in index_lists.values()]
    values = list(zip(*index_values, strict=True))
    missing_idx = [list(idx) for idx in itertools.product(*expected.values()) if idx not in values]
    logger.info(f"Found {len(missing_idx)} missing files out of {n_expected if n_expected is not None else nfiles}.")
    return np.array(missing_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outlier detection for BGS measurements")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing the files to compress")
    parser.add_argument("--measurement", type=str, required=True, help="Measurement to process")
    parser.add_argument("--n_expected", type=int, required=True, help="Expected number of files")
    parser.add_argument("--estimator_config", type=str, required=True, help="YAML file containing estimator parameters.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the output files (corrupted and outlier indices)")
    parser.add_argument("--raw", action="store_true", help="If set, will use the raw data instead of the merged/selected data")
    parser.add_argument("--sigma", type=float, default=3.0, help="Sigma threshold for outlier detection")
    parser.add_argument("--log_level", type=str, default='warning', help="Set logging level (e.g., DEBUG, INFO)")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    with Path(args.estimator_config).open() as f:
        estimator_config = yaml.load(f, Loader=NumpyLoader)  # noqa: S506

    stat_name = args.measurement
    confargs = estimator_config.get(stat_name, {})
    load_args = confargs.get("load", {})
    reader = get_estimator(stat_name).load

    # NOTE: using hardcoded pattern/index structure for those files, as they handle outputs of measure_box.py
    pattern = r"c{cosmo_idx}_ph{phase_idx}/seed{seed}/hod{hod_idx}/" + stat_name + r"_los-{los}.h5"
    ignore_index = ["los"] if not args.raw else None

    compressor = Compressor(root=Path(args.root), pattern=pattern)
    corrupted_idx = check_corrupted_files(
        compressor,
        reader=reader,
        **load_args,
    )

    nfiles = len(compressor._files)
    if nfiles != args.n_expected:
        logger.warning(f"Expected {args.n_expected} files, but found {nfiles}.")

    group = compressor.read(reader=reader, ignore_index=ignore_index, **load_args)
    if not args.raw:
        group = select(stat_name, group)
        group = group.merge(method=lsstypes.mean)  # Merge identical indices
    outlier_idx = check_outliers(group, sigma=args.sigma)

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(args.save_dir) / f"{stat_name}_corrupted_idx.npy", corrupted_idx)
        np.save(Path(args.save_dir) / f"{stat_name}_outlier_idx.npy", outlier_idx)
