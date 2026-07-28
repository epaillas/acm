"""
Script to measure clustering statistics from HOD catalogs generated with AbacusHOD.

Usage:
    python measure_box.py -h
"""  # noqa: INP001
import argparse
import itertools
import logging
from gc import collect
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from cosmoprimo.fiducial import AbacusSummit
from jax import clear_caches

from acm.catalogs.backends.abacus import AbacusHODBackend  # noqa: F401
from acm.catalogs.dataclasses import Tracer
from acm.catalogs.factories import SnapshotCatalogFactory
from acm.catalogs.products.snapshot import SnapshotCatalog, boundary_check
from acm.utils.logging import get_logger_for_script, setup_logging
from acm.utils.paths import lookup_registry_path
from acm.utils.scripts import (
    NumpyLoader,
    apply_parser_default,
    detect_gpu,
    dump_config,
    get_nthreads,
    load_parser_default,
    retry,
)

from _estimators import get_estimator

logger = get_logger_for_script(__file__)

def get_hod_params(
    tracer_names: list[str],
    hod_dir: str | Path,
    pattern: str,
) -> dict[str, list[dict]]:
    """
    Determine the combination of HOD parameters for each tracer.

    Provides only the first N parameters, where N is the smallest number of parameters found across all files.

    Parameters
    ----------
    tracer_names: list[str]
        List of tracer names to get the HOD parameters for.
    hod_dir: str | Path
        Dicrectory where to find the HOD parameter CSV files.
    pattern: str
        Formatted string pattern to find the HOD parameter files in hod_dir.
        Will try to format it with a parameter `tracer` corresponding to the tracer name.

    Returns
    -------
    dict[str, list[dict]]
        A dictionnary associating each tracer name with a list of dictionnaries,
        each dictionary containing HOD parameters.

    Raises
    ------
    FileNotFoundError
        If any of the constructed filenames do not exist.
    ValueError
        If the number of parameters do not match across the HOD parameter files.

    Examples
    --------
    >>> get_hod_params(['BGS'], '/some_dir/', pattern='{tracer}/myfile.csv')
    {'BGS': [
        {'p0': 0, 'p1': 1},
        {'p0': 2, 'p1': 3},
        ...
    ]}
    """
    fns = [Path(hod_dir)/pattern.format(tracer=t) for t in tracer_names]
    fnf = [fn for fn in fns if not fn.exists()] # Files Not Found
    if any(fnf):
        raise FileNotFoundError(f"Some files were not found: {fnf}")

    _params = [pd.read_csv(f) for f in fns]

    # Check that the number of columns is consistent
    shapes = np.array([df.shape for df in _params])
    Nhod = np.unique(shapes[:, 0])
    Nparams = np.unique(shapes[:, 1])
    if len(Nparams) != 1:
        raise ValueError(f"Found inconsitent number of parameters across HOD parameter files: {Nparams}")

    _d = dict(zip(tracer_names, shapes, strict=True))
    logger.debug(f"Found HOD parameter files of shapes: {_d}.")

    if len(np.unique(Nhod)) != 1:
        n_min = min(Nhod)
        logger.warning(f"Found different lengths for HOD parameter files. Keeping only first {n_min} parameter combinations.")
        _params = [df[:n_min] for df in _params]

    return {k: v.to_dict('records') for k, v in zip(tracer_names, _params, strict=True)}

def update_dict_with_keys(*d: dict, **kwargs) -> None:
    """Update dictionaries in place if the parameter names are present in their respective keys."""
    for _d in d:
        update_keys = {k: v for k, v in kwargs.items() if k in _d}
        _d.update(update_keys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate snapshot mocks and compute measurements on several statistics.")
    parser.add_argument("--config", type=str, help="Path to a YAML file to set default parameters. Command line arguments override config file settings.")
    parser.add_argument("--dump_config", action="store_true", help="If set, dumps the current configuration in the console and exits.")

    config_args = load_parser_default(parser)

    parser.add_argument("-c", "--cosmologies", type=int, nargs="+", required=True, help="List of cosmology indices to process.")
    parser.add_argument("-p", "--phases", type=int, nargs="+", required=True, help="List of phase indices to process.")
    parser.add_argument("-s", "--seeds", type=int, nargs="+", required=True, help="List of seeds to process.")
    parser.add_argument("--hod_dir", type=str, required=True, help="Directory containing the HOD parameter files.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the measurements.")
    parser.add_argument("--estimator_config", type=str, required=True, help="YAML file containing estimator parameters.")
    parser.add_argument("--hods", type=int, nargs="+", help="List of HOD indices to process. Disables start_hod and n_hod.")
    parser.add_argument("--n_hod", type=int, default=100, help="Number of HODs to run per cosmology, phase and seed.")
    parser.add_argument("--start_hod", type=int, default=0, help="Starting index for HODs to process.")
    parser.add_argument("--max_hod", type=int, default=1000, help="Maximum number of HOD processed per cosmology, phase and seed.")
    parser.add_argument("--sim_type", type=str, default="base", help="Simulation type (e.g., base, small).")
    parser.add_argument("--redshift", type=float, default=0.2, help="Redshift of the simulations to load.")
    parser.add_argument("--add_rsd", action="store_true", help="Add RSD distorsions.")
    parser.add_argument("--add_ap", action="store_true", help="Add AP distorsions.")
    parser.add_argument("--target_density", type=float, help="Only compute measurements on mocks reaching this density if set.")
    parser.add_argument("--process_underdense", action="store_true", help="Compute underdense measurements, if target_density is set.")
    parser.add_argument("--save_galaxies", action="store_true", help="Save galaxy catalogs.")
    parser.add_argument("--measurements", type=str, nargs="+", default=[], help="List of statistics to measure on mocks.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--parameters_override", type=str, help="File containing an array of parameters overriding cosmologies, phases, seeds and hod parameter values.")
    parser.add_argument("--failures", type=int, default=3, help="Number of tries for each etsimator computation before skipping (solving memory issues).")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--log_file", type=str, help="File to save logs. If None, logs are printed to console.")

    apply_parser_default(parser, config_args)
    dump_config(parser)
    args = parser.parse_args()
    target_density = args.target_density

    setup_logging(level=args.log_level, filename=args.log_file)
    logging.getLogger("numba").setLevel(logging.INFO) # Remove noisy DEBUG levels
    logging.getLogger("jax").setLevel(logging.INFO)
    logging.getLogger("h5py").setLevel(logging.INFO)

    with Path(args.estimator_config).open() as f:
        estimator_config = yaml.load(f, Loader=NumpyLoader)  # noqa: S506

    is_gpu = detect_gpu()
    nthreads = get_nthreads()

    # NOTE: Hardcoded single BGS tracer for this script
    abacus_paths = lookup_registry_path("Abacus.yaml", "BGS", "box", args.sim_type)
    tracer_names = ['BGS']

    # Precompute indices to avoid loop nesting & make indice overload easier
    hods = args.hods or range(args.start_hod, args.start_hod + args.max_hod)
    indices = itertools.product(args.cosmologies, args.phases, args.seeds, hods)
    if args.parameters_override:
        _po = np.load(args.parameters_override)
        indices = _po[np.lexsort((_po[:, 1], _po[:, 0]))] # sort by (cosmo, phase)
        logger.info(f"Overriding parameters from {args.parameters_override}.")
    grouped = itertools.groupby(indices, key=lambda x: (x[0], x[1]))

    for (cosmo_idx, phase_idx), group in grouped:
        hod_count = 0 # Number of computed HODs per cosmo/phase pair
        factory = SnapshotCatalogFactory(
            backend = "AbacusHOD",
            catalog_class = SnapshotCatalog,
            cosmo = AbacusSummit(cosmo_idx), # NOTE: cosmo_fid=DESI()
            cosmo_idx = cosmo_idx, # From here, backend arguments are passed as kwargs
            phase_idx = phase_idx,
            sim_type = args.sim_type,
            sim_dir = abacus_paths["sim_dir"],
            subsample_dir = abacus_paths["subsample_dir"],
        )
        logger.info(f'Loaded factory for c{cosmo_idx:03d}_ph{phase_idx:03d}')

        factory.backend.load_dark_matter_catalog(
            redshift = args.redshift,
            tracers = [Tracer(name=k) for k in tracer_names] # Name only = default
        )

        hod_fn = f"Bouchard25_c{cosmo_idx:03d}.csv" # NOTE: Hardcoded pattern
        hod_params = get_hod_params(tracer_names, args.hod_dir, pattern=hod_fn)

        for _, _, seed, hod_idx in group:
            tracers = [Tracer(name=k, params=v[hod_idx]) for k, v in hod_params.items()]
            factory.make_catalogs(
                redshifts = [args.redshift],
                tracers = tracers,
                use_logsigma = True,
                seed = seed,
            )
            catalog = factory.get_catalog(args.redshift)
            mock_dir = Path(args.save_dir) / args.sim_type / f'c{cosmo_idx:03d}_ph{phase_idx:03d}/seed{seed}/hod{hod_idx:03d}'

            if args.save_galaxies:
                catalog.save(mock_dir / 'catalog.h5')

            for los in ['x', 'y', 'z']:
                logger.info(f'Computing measurements for HOD {hod_idx:03d}, seed {seed}, los {los}')
                catalog.clear_transforms()
                if args.add_rsd:
                    offset = catalog.boxsize[catalog.pos_columns.index(los)] / 2
                    catalog.rsd(los=los, wrap=True, offset=offset)
                if args.add_ap:
                    catalog.ap(los=los)

                nbar = catalog.nbar

                if los =='x':
                    logger.info(f"Density for hod {hod_idx:03d}: {nbar:.4e} h^3 Mpc^-3")
                    density_file = mock_dir / 'density.npy'
                    if not density_file.exists() or args.overwrite:
                        mock_dir.mkdir(exist_ok=True, parents=True)
                        np.save(density_file, nbar)

                if target_density is not None:
                    if nbar < target_density and not args.process_underdense:
                        logger.info(f"Density below target ({nbar:.4e}<{target_density:.4e}). Skipping...")
                        break # In theory, same density for all los on boxes
                    for tracer in tracers: # FIXME (later): target density selection wrt tracers ?
                        catalog.downsample(tracer.name, nbar=target_density, seed=42)

                positions = catalog.positions().to_numpy()
                boundary_check(positions, catalog.boxsize, center_at_zero=True)
                logger.debug(f'Positions shape: {positions.shape}')
                logger.info(f'Box size: {catalog.boxsize}')

                for stat_name in args.measurements:
                    fn = mock_dir / f"{stat_name}_los-{los}.h5"
                    if fn.exists() and args.overwrite is False:
                        logger.info(f'File {fn} exists and {args.overwrite=}. Skipping...')
                        continue

                    estimator_kwargs = estimator_config.get(stat_name, {}).copy()
                    update_dict_with_keys(
                        estimator_kwargs,
                        boxsize = catalog.boxsize,
                        boxcenter = 0,
                        los = los,
                        gpu = is_gpu,
                        nthreads = nthreads,
                    )
                    # FIXME: use estimator classes when implemented
                    func = get_estimator(stat_name)
                    retry(args.failures, func, positions, fn, **estimator_kwargs)
            else: # Only run if target_density does not break los loop
                hod_count += 1
                logger.debug(f"c{cosmo_idx:03d}_ph{phase_idx:03d}: Computed {hod_count}/{args.n_hod} mocks.")
            if hod_count >= args.n_hod:
                break # break inner loop
        del factory
        clear_caches()
        collect()
