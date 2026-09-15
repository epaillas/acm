import argparse  # noqa: INP001
from pathlib import Path

import pandas as pd
from sunbird.inference.priors import Bouchard25

from acm.utils.abacus import load_cosmologies
from acm.utils.default import cosmo_list
from acm.utils.logging import get_logger_for_script, setup_logging
from acm.utils.sampler import LatinHyperCubeSampler

logger = get_logger_for_script(__file__)

# Default parameters
abacus_fn = "/pscratch/sd/s/sbouchar/acm/bgs/parameters/cosmo/AbacusSummit.csv"
parameters = [
    "omega_b",
    "omega_cdm",
    "sigma8_m",
    "n_s",
    "alpha_s",
    "N_ur",
    "w0_fld",
    "wa_fld",
]
order = [
    "omega_b",
    "omega_cdm",
    "sigma8_m",
    "n_s",
    "nrun",
    "N_ur",
    "w0_fld",
    "wa_fld",
    "logM_cut",
    "logM_1",
    "sigma",
    "alpha",
    "kappa",
    "alpha_c",
    "alpha_s",
    "s",
    "A_cen",
    "A_sat",
    "B_cen",
    "B_sat",
]

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, required=True, help="Total number of combinations to sample")
    parser.add_argument("-s", "--save_dir", type=str, required=True, help="Directory to save the sampled parameters")
    parser.add_argument("-f", "--filename", type=str, default=abacus_fn, help="Path to the AbacusSummit cosmology parameters CSV file")
    parser.add_argument("-p", "--parameters", type=str, nargs="+", default=parameters, help="List of cosmological parameters to include")
    parser.add_argument("-c", "--cosmologies", type=int, nargs="+", default=cosmo_list, help="List of cosmology indices to sample HODs for")
    args = parser.parse_args()
    # fmt: on

    setup_logging()
    logger.info(
        f"Sampling {args.number} HODs for {len(args.cosmologies)} cosmologies from {args.filename} with parameters {args.parameters}"
    )

    cosmologies = load_cosmologies(
        filename=args.filename,
        cosmologies=args.cosmologies,
        parameters=parameters,
        mapping={"alpha_s": "nrun"},  # map alpha_s to nrun as it exists in HODs
    )

    ranges = Bouchard25().ranges
    lhc = LatinHyperCubeSampler(ranges=ranges)  # ty: ignore[invalid-argument-type]
    sample = lhc.sample(n=args.number)  # All HODs sampled in the Latin Hypercube
    splits = lhc.split(sample, keys=list(cosmologies))  # Split the HODs by cosmology

    finals = {}
    for k, d in splits.items():  # Add cosmology parameters to the HODs
        new_cols = pd.DataFrame(cosmologies[k], index=[0]) # Only one row to duplicate
        finals[k] = lhc.add_columns(d, cosmologies[k])

    fn = Path(args.save_dir) / "hod/Bouchard25_{key}.csv"
    lhc.save(splits, save_fn=fn , order=order[len(parameters):])
    fn = Path(args.save_dir) / "cosmo+hod/AbacusSummit_{key}.csv"
    lhc.save(finals, save_fn=fn, order=order)
