import argparse  # noqa: INP001
from pathlib import Path

import lsstypes
import numpy as np
import xarray
import yaml
from measure_box import get_estimator

from acm.estimators.compression import Compressor, ObjectGroup, split_test_set
from acm.utils.logging import get_logger_for_script, setup_logging
from acm.utils.scripts import NumpyLoader
from acm.utils.xarray import dataset_to_dict

logger = get_logger_for_script(__file__)

K_MIN = 2 * np.pi / 500  # lower limit fixed by small boxsize
K_MAX = np.pi * 512 / 2200 # Higher limit fixed by Nyquist frequency of the largest boxsize*

# Order of the parameters to select in the attributes of the read objects.
parameters = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'nrun', 'N_ur', 'w0_fld', 'wa_fld', 'logM_cut', 'logM_1', 'sigma', 'alpha', 'kappa', 'alpha_c', 'alpha_s', 's', 'A_cen', 'A_sat', 'B_cen', 'B_sat']

def select(stat_name: str, group: ObjectGroup) -> ObjectGroup:
    """Select the relevant data from the ObjectGroup based on the statistic name."""
    # FIXME (later): Replace hardcoded values ?
    _get, _rebin, _select = {}, {}, {} # Empty by default
    if stat_name == "tpcf" or "xi" in stat_name:
        _get.update({"ells": [0, 2]})  # Get only monopole and quadrupole
        _rebin.update({"s": slice(0, None, 3)})  # Rebin s by a factor of 3
    if stat_name == "spectrum" or "pk" in stat_name:
        _get.update({"ells": [0, 2]})
        _rebin.update({"k": slice(0, None, 3)})
        _select.update({"k": (K_MIN, K_MAX)})
    if stat_name.startswith("ds"):
        _get.update({"quantiles": [0, 1, 3, 4]})
    group = group.get(**_get).select(**_rebin).select(**_select)
    return group

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root directory containing the files to compress")
    parser.add_argument("--measurement", type=str, required=True, help="Measurement to process")
    parser.add_argument("--estimator_config", type=str, required=True, help="YAML file containing estimator parameters.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the compressed files")
    parser.add_argument("--n_hod", type=int, default=None, help="Number of HODs to keep (default: all)")
    parser.add_argument("--test_cosmos", type=int, nargs="+", default=[], help="List of cosmo indices to use as test set")
    parser.add_argument("--log_level", type=str, default='warning', help="Set logging level (e.g., DEBUG, INFO)")
    args = parser.parse_args()

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
    reader = get_estimator(stat_name).load

    # NOTE: using hardcoded pattern/index structure for those files, as they handle outputs of measure_box.py
    pattern = r"c{cosmo_idx}_ph{phase_idx}/seed{seed}/hod{hod_idx}/" + stat_name + r"_los-{los}.h5"
    ignore_index = ["los"]
    reindex = {"hod_idx": ["cosmo_idx", "phase_idx"]}

    compressor = Compressor(root=Path(args.root) / "base", pattern=pattern)
    group = compressor.read(reader=reader, ignore_index=ignore_index, **load_args)
    group = select(stat_name, group)
    group = group.merge(method=lsstypes.mean)  # Merge identical indices
    y = Compressor.compress(data=group, reindex=reindex)
    x = Compressor.compress(data=group, reindex=reindex, attrs=parameters)

    compressor = Compressor(root=Path(args.root) / "small", pattern=pattern)
    group = compressor.read(reader=reader, ignore_index=ignore_index, **load_args)
    group = select(stat_name, group)
    group = group.merge(method=lsstypes.mean)  # Merge identical indices
    cov_y = Compressor.compress(data=group, reindex=reindex)

    ds = xarray.Dataset({"x": x, "y": y, "cov_y": cov_y})
    ds = split_test_set(ds, filters=test_filter)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(args.save_dir) / f"{stat_name}.npy"
    payload = np.array(dataset_to_dict(ds), dtype=object)
    np.save(save_fn, payload)
    logger.info(f"Saving compressed data to {save_fn}")
