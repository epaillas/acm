import argparse  # noqa: INP001
from pathlib import Path

import lsstypes
import numpy as np
import yaml
from lsstypes import ObservableTree
from measure_box import get_estimator
from sample_hods import order

from acm.estimators.compression import Compressor, ObjectGroup
from acm.utils.logging import get_logger_for_script, setup_logging
from acm.utils.scripts import NumpyLoader

logger = get_logger_for_script(__file__)

K_MIN = 2 * np.pi / 500  # lower limit fixed by small boxsize
K_MAX = np.pi * 512 / 2200  # Higher limit fixed by Nyquist freq. of the largest boxsize


def select(group: ObjectGroup, **kwargs) -> ObjectGroup:
    """Select the relevant data from the ObjectGroup based on the statistic name."""
    get = kwargs.get("get", {})
    _rebin = kwargs.get("rebin", {})
    if not all(isinstance(v, int) and v > 0 for v in _rebin.values()):
        raise ValueError(f"Rebin values must be positive integers, got {_rebin}")
    rebin = {k: slice(0, None, v) for k, v in _rebin.items()}
    select = kwargs.get("select", {})
    logger.debug(f"Selecting data with {get=}, {rebin=}, {select=}")
    return group.get(**get).select(**rebin).select(**select)


def split_tree(tree: ObservableTree, **labels) -> tuple[ObservableTree, ObservableTree]:
    """Split an ObservableTree into two trees based on label filters."""
    if len(labels) > 1:  # FIXME
        raise NotImplementedError("split_tree currently only supports single label.")
    label_name = next(iter(labels.keys()))
    tree_label = tree.labels(return_type="unflatten", level=None)[label_name]
    label_in = labels[label_name]
    label_out = list(set(tree_label) - set(label_in))
    logger.debug(f"Splitting tree on {label_name}: {label_in=} and {label_out=}")
    tree_in = tree.get(**{label_name: label_in})
    tree_out = tree.get(**{label_name: label_out})
    return tree_in, tree_out


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root directory containing the files to compress")
    parser.add_argument("--measurement", type=str, required=True, help="Measurement to process")
    parser.add_argument("--estimator_config", type=str, required=True, help="YAML file containing estimator parameters.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the compressed files")
    parser.add_argument("--test_cosmos", type=int, nargs="+", default=[], help="List of cosmo indices to use as test set")
    parser.add_argument("--log_level", type=str, default="info", help="Set logging level (e.g., DEBUG, INFO)")
    args = parser.parse_args()
    # fmt: on

    setup_logging(level=args.log_level)

    with Path(args.estimator_config).open() as f:
        estimator_config = yaml.load(f, Loader=NumpyLoader)  # noqa: S506

    test_filter = {}
    if args.test_cosmos:
        logger.info(f"Using test cosmologies: {args.test_cosmos}")
        test_filter["cosmo_idx"] = args.test_cosmos

    stat_name = args.measurement
    confargs = estimator_config.get(stat_name, {})
    load_args = confargs.get("load", {})
    compress_args = confargs.get("compress", {})
    reader = get_estimator(stat_name).load

    # NOTE: using hardcoded pattern/index structure for those files, as they handle outputs of measure_box.py
    pattern = r"c{cosmo_idx}_ph{phase_idx}/seed{seed}/hod{hod_idx}/" + stat_name + r"_los-{los}.h5"  # fmt: skip
    ignore_index = ["los"]
    reindex = None  # {"hod_idx": ["cosmo_idx", "phase_idx"]}

    compressor = Compressor(root=Path(args.root) / "base", pattern=pattern)
    group = compressor.read(reader=reader, ignore_index=ignore_index, **load_args)
    group = select(group, **compress_args)
    group = group.merge(method=lsstypes.mean)  # Merge identical indices
    y = group.to_lsstypes(reindex=reindex)
    x = group.to_lsstypes(reindex=reindex, attrs=order)

    compressor = Compressor(root=Path(args.root) / "small", pattern=pattern)
    group = compressor.read(reader=reader, ignore_index=ignore_index, **load_args)
    group = select(group, **compress_args)
    group = group.merge(method=lsstypes.mean)  # Merge identical indices
    cov_y = group.to_lsstypes(reindex=reindex)

    data = lsstypes.ObservableTree(
        branches=[x, y, cov_y],
        name=["x", "y", "cov_y"],
    )

    if test_filter:  # Only split if test_filter is not empty
        for name in ["x", "y"]:
            logger.info(f"Splitting {name} into test/train using filter: {test_filter}")
            data = data.insert(
                split_tree(data.get(name=name), **test_filter),
                name=[f"{name}_test", f"{name}_train"],
            )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(args.save_dir) / f"{stat_name}.h5"
    data.write(save_fn)  # NOTE: no overwrite protection or atomic write
    logger.info(f"Saving compressed data to {save_fn}")
