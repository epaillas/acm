from pathlib import Path
import argparse
from astropy.io import fits
import numpy as np
from pycorr import TwoPointCorrelationFunction
from pyrecon.utils import sky_to_cartesian
import fitsio
from pathlib import Path
from cosmoprimo.fiducial import DESI
from acm.estimators.galaxy_clustering.density_split import DensitySplit
from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform
from acm.estimators.galaxy_clustering.minkowski import MinkowskiFunctionals 
from acm.utils import setup_logging
import logging

logger = logging.getLogger("fibers")
setup_logging()

cosmo = DESI()
distance = cosmo.comoving_radial_distance


def read_desi_data(mock_id=0, region="NGC", fiber_asign=None):
    if fiber_asign is None:
        data_fn = data_dir / f"mock{mock_id}/LRG_complete_{region}_clustering.dat.fits"
    elif fiber_asign == "altmtl":
        data_fn = (
            data_dir
            / f"altmtl{mock_id}/mock{mock_id}/LSScats/LRG_{region}_clustering.dat.fits"
        )
    elif fiber_asign == "ffa":
        data_fn = data_dir / f"mock{mock_id}/LRG_ffa_{region}_clustering.dat.fits"
    return fitsio.read(data_fn)


def read_desi_randoms(mock_id=0, irand=0, region="NGC", fiber_asign=None):
    if fiber_asign is None:
        data_fn = (
            data_dir
            / f"mock{mock_id}/LRG_complete_{region}_{irand}_clustering.ran.fits"
        )
    elif fiber_asign == "altmtl":
        data_fn = (
            data_dir
            / f"altmtl{mock_id}/mock{mock_id}/LSScats/LRG_{region}_{irand}_clustering.ran.fits"
        )
    elif fiber_asign == "ffa":
        data_fn = (
            data_dir / f"mock{mock_id}/LRG_ffa_{region}_{irand}_clustering.ran.fits"
        )
    return fitsio.read(data_fn)


def get_clustering_positions_weights(data, zmin=None, zmax=None, keep_ratio=None):
    if zmin is not None and zmax is not None:
        mask = (data["Z"] > zmin) & (data["Z"] < zmax)
    else:
        mask = np.ones(len(data), dtype=bool)
    if keep_ratio is not None:
        downsampling_mask = np.zeros(len(data), dtype=bool)
        n_points_to_keep = int(len(data) * keep_ratio)
        indices_to_keep = np.random.choice(
            len(data), size=n_points_to_keep, replace=False
        )
        downsampling_mask[indices_to_keep] = True
        mask = mask & downsampling_mask
    ra = data[mask]["RA"]
    dec = data[mask]["DEC"]
    dist = distance(data[mask]["Z"])
    pos = sky_to_cartesian(ra=ra, dec=dec, dist=dist)
    weights = data[mask]["WEIGHT"]
    return pos, weights, mask


def get_data_and_randoms(
    mock_id=0,
    region="NGC",
    n_random_files=5,
    fiber_asign=None,
    zmin=None,
    zmax=None,
    keep_ratio=None,
):
    data = read_desi_data(
        mock_id,
        region,
        fiber_asign=fiber_asign,
    )
    pos, weights, mask = get_clustering_positions_weights(
        data, zmin=zmin, zmax=zmax, keep_ratio=keep_ratio
    )
    randoms_positions, randoms_weights = [], []
    for irand in range(n_random_files):
        randoms = read_desi_randoms(
            mock_id=mock_id, irand=irand, region=region, fiber_asign=fiber_asign
        )
        rpos, rweights, rmask = get_clustering_positions_weights(
            randoms,
            zmin=zmin,
            zmax=zmax,
        )
        randoms_positions.append(rpos)
        randoms_weights.append(rweights)
    randoms_positions = np.concatenate(randoms_positions)
    randoms_weights = np.concatenate(randoms_weights)
    return pos, weights, randoms_positions, randoms_weights


def get_tpcf(
    data_positions,
    data_weights,
    randoms_positions,
    randoms_weights,
    nthreads=128,
):
    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (sedges, muedges)
    return TwoPointCorrelationFunction(
        "smu",
        edges,
        data_positions1=data_positions.T,
        data_weights1=data_weights,
        randoms_positions1=randoms_positions.T,
        randoms_weights1=randoms_weights,
        nthreads=nthreads,
        los="midpoint",
        engine="corrfunc",
    )


def compute_and_store_tpcf(
    pos, weights, randoms_pos, randoms_weights, mock_id, fiber_asign=None
):
    tpcf = get_tpcf(pos, weights, randoms_pos, randoms_weights)
    filename = f"tpcf_mock{mock_id}"
    if fiber_asign is not None:
        filename += f"_{fiber_asign}"
    tpcf.save(output_dir / f"{filename}.npy")



def compute_and_store_dsc(
    pos, weights, randoms_pos, randoms_weights, mock_id, fiber_asign=None,
):
    smoothing_radius = 10
    cellsize = 10.0
    nquantiles = 5
    ds = DensitySplit(positions=randoms_pos, cellsize=cellsize)
    ds.assign_data(
        positions=pos,
        weights=weights,
    )
    ds.assign_randoms(
        positions=randoms_pos,
        weights=randoms_weights,
    )
    ds.set_density_contrast(smoothing_radius=smoothing_radius, save_wisdom=False)
    ds.set_quantiles(
        query_positions=randoms_pos, nquantiles=nquantiles, query_method="randoms"
    )
    sedges = np.arange(0, 201, 1)
    muedges = np.linspace(-1, 1, 241)
    ccf = ds.quantile_data_correlation(
        data_positions=pos, randoms_positions=randoms_pos,
        edges=(sedges, muedges), los="midpoint", nthreads=128, gpu=False
    )
    acf = ds.quantile_correlation(
        randoms_positions=randoms_pos,
        edges=(sedges, muedges), los="midpoint", nthreads=128, gpu=False
    )
    filename = f"dsc_ccf_mock{mock_id}"
    if fiber_asign is not None:
        filename += f"_{fiber_asign}"
    for quantile, ccf_quantile in enumerate(ccf):
        ccf_quantile.save(output_dir / f"{filename}_q{quantile}.npy")
    filename = f"dsc_acf_mock{mock_id}"
    if fiber_asign is not None:
        filename += f"_{fiber_asign}"
    for quantile, acf_quantile in enumerate(acf):
        acf_quantile.save(output_dir / f"{filename}_q{quantile}.npy")



def compute_and_store_wst(
    pos, weights, randoms_pos, randoms_weights, mock_id, fiber_asign=None
):
    smoothing_radius = 10
    cellsize = 5.0
    wst = WaveletScatteringTransform(positions=randoms_pos, cellsize=cellsize)
    wst.assign_data(
        positions=pos,
        weights=weights,
    )
    wst.assign_randoms(
        positions=randoms_pos,
        weights=randoms_weights,
    )
    wst.set_density_contrast(smoothing_radius=smoothing_radius, save_wisdom=True)
    smatavg = wst.run()
    filename = f"wst_mock{mock_id}"
    if fiber_asign is not None:
        filename += f"_{fiber_asign}"
    np.save(output_dir / f"{filename}.npy", smatavg)


def compute_and_store_mf(
    pos, weights, randoms_pos, randoms_weights, mock_id, fiber_asign=None
):
    smoothing_radius = 10
    cellsize = 5.0
    mf = MinkowskiFunctionals(positions=randoms_pos, cellsize=cellsize)
    mf.assign_data(
        positions=pos,
        weights=weights,
    )
    mf.assign_randoms(
        positions=randoms_pos,
        weights=randoms_weights,
    )
    mf.set_density_contrast(smoothing_radius=smoothing_radius, save_wisdom=True)
    mf.run()
    filename = f"mf_mock{mock_id}"
    if fiber_asign is not None:
        filename += f"_{fiber_asign}"
    np.save(output_dir / f"{filename}.npy", mf.MFs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_statistics", nargs="+", default=["tpcf", "wst"])
    args = parser.parse_args()
    summary_statistics = args.summary_statistics

    data_dir = Path(
        "/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/"
    )
    output_dir = Path("/pscratch/sd/c/cuesta/fibers/")
    test_mode = True
    mock_ids = list(range(25))
    fiber_asignments = [None, "altmtl", "ffa"]
    # turn summary statistics into command line arguments
    keep_ratio = 0.6
    zmin, zmax = 0.4, 0.6
    for mock_id in mock_ids:
        for fiber_asign in fiber_asignments:
            print(f"Analysing mock {mock_id} with fiber assignment {fiber_asign}")
            pos, weights, randoms_pos, randoms_weights = get_data_and_randoms(
                mock_id=mock_id,
                fiber_asign=fiber_asign,
                zmin=zmin,
                zmax=zmax,
                keep_ratio=keep_ratio if fiber_asign is None else None,
            )
            if test_mode:
                pos = pos[::]
                weights = weights[::]
                randoms_pos = randoms_pos[::10]
                randoms_weights = randoms_weights[::10]
            print(f"Number of galaxies: {len(pos)}")
            print(f"Number of randoms: {len(randoms_pos)}")

            for statistic in summary_statistics:
                print(f"Computing {statistic}")
                if statistic == "tpcf":
                    compute_and_store_tpcf(
                        pos,
                        weights,
                        randoms_pos,
                        randoms_weights,
                        mock_id=mock_id,
                        fiber_asign=fiber_asign,
                    )
                elif statistic == "dsc":
                    compute_and_store_dsc(
                        pos,
                        weights,
                        randoms_pos,
                        randoms_weights,
                        mock_id=mock_id,
                        fiber_asign=fiber_asign,
                    )
                elif statistic == "wst":
                    compute_and_store_wst(
                        pos,
                        weights,
                        randoms_pos,
                        randoms_weights,
                        mock_id=mock_id,
                        fiber_asign=fiber_asign,
                    )
                elif statistic == "mf":
                    compute_and_store_mf(
                        pos,
                        weights,
                        randoms_pos,
                        randoms_weights,
                        mock_id=mock_id,
                        fiber_asign=fiber_asign,
                    )