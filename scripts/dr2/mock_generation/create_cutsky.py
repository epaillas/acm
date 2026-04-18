from acm.hod import CutskyHOD, CutskyRandoms
from acm import setup_logging
import numpy as np
from pathlib import Path
import pandas
from cosmoprimo.fiducial import AbacusSummit
from mpi4py import MPI
import argparse
import logging

mpicomm = MPI.COMM_WORLD

setup_logging()


def parse_ranges(pairs):
    """Parse a list of comma-separated pairs like 0.4,0.6 0.6,1.1."""
    ranges = []
    for pair in pairs:
        low, high = pair.split(',')
        ranges.append((float(low), float(high)))
    return ranges

def get_cli_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument('--start_cosmo', type=int, default=0)
    parser.add_argument('--n_cosmo', type=int, default=1) 
    parser.add_argument('--tracer', type=str, default='LRG')
    parser.add_argument('--region', type=str, default='NGC')
    parser.add_argument('--release', type=str, default='Y3')
    parser.add_argument(
        '--snapshots',
        type=float,
        nargs='+',
        required=True,
        help='Space-separated list of redshifts, e.g. "0.5 0.8".'
    )
    parser.add_argument(
        '--zranges',
        type=str,
        nargs='+',
        required=True,
        help='Space-separated z ranges as comma-separated pairs, e.g. "0.4,0.6 0.6,1.1".'
    )
    parser.add_argument('--make_randoms', action='store_true', default=False)
    parser.add_argument('--n_randoms', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='/pscratch/sd/a/acasella/acm/dr2/HOD/cutsky_mocks/')

    args = parser.parse_args()
    return args

def get_hod_params(nrows=None):
    """Some example HOD parameters."""
    hod_dir = Path(f'/pscratch/sd/a/acasella/acm/dr2/HOD/')
    hod_fn = hod_dir / f'test7_hod_params_c000.csv'
    df = pandas.read_csv(hod_fn, delimiter=',')
    df.columns = df.columns.str.strip()
    df.columns = list(df.columns.str.strip('# ').values)
    return df.to_dict('list')



if __name__ == '__main__':

    logger = logging.getLogger(__name__)

    args = get_cli_args()
    setup_logging()
    snapshots = args.snapshots
    zranges = parse_ranges(args.zranges)
    tracer = args.tracer
    region = args.region
    release = args.release
    cosmos = list(range(args.start_cosmo, args.start_cosmo + args.n_cosmo))
    phases = list(range(args.start_phase, args.start_phase + args.n_phase))
    hod_idx = 1  # TODO : allow varying hod_idx

    for cosmo_idx in cosmos:
        fid_cosmo = AbacusSummit(cosmo_idx)

        for phase_idx in phases:
            if mpicomm.rank == 0:
                logger.info(f'Generating cutsky mock for cosmo {cosmo_idx}, phase {phase_idx}.')

            save_dir = Path(args.save_dir) / f'c{cosmo_idx:03}_ph{phase_idx:03}'
            save_dir.mkdir(parents=True, exist_ok=True)

            # read example HOD parameters
            hod_params = get_hod_params()

            # initialize class
            cutsky = CutskyHOD(
                tracer=tracer,
                varied_params=hod_params.keys(),
                zranges=zranges, snapshots=snapshots,
                cosmo_idx=cosmo_idx, phase_idx=phase_idx,
                load_existing_hod=False, mpicomm=mpicomm
            )
            # you can set load_existing_hod=True to load a pre-made catalog rather
            # than actually sampling from AbacusSummit for a quick debugging

            # sample HOD parameters and build the cutsky mock
            # this does not have the angular or radial mask carved in yet
            nz_filename = (
                f'/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/'
                f'{tracer}_full_HPmapcut_nz.txt'
            )
            hod = {key: hod_params[key][hod_idx] for key in hod_params.keys()}
            cutsky.sample_hod(
                hod,
                nthreads=1,
                region=region,
                release=release,
                target_nz_filename=nz_filename,
            )

            # apply angular and radial masks
            cutsky.apply_angular_mask(
                region=region, release=release, npasses=None, program='dark'
            )
            cutsky.apply_radial_mask(nz_filename=nz_filename)

            cutsky.save(save_dir / f'{tracer}_{region}_hod{hod_idx:03}_new_constraints5.dat.fits')

            if args.make_randoms:
                for rnd_idx in range(args.n_randoms):
                # generate a random catalog with the same angular and radial masks
                    cutsky_randoms = CutskyRandoms(
                        rarange=(cutsky.catalog['RA'].min(), cutsky.catalog['RA'].max()),
                        decrange=(cutsky.catalog['DEC'].min(), cutsky.catalog['DEC'].max()),
                        zrange=(np.min(zranges), np.max(zranges)),
                        nbar=2000,  # this is *surface area* density, in (deg^2)^-1
                        # csize=10_000_000,  # alternatively, pass the desired number of randoms
                        seed=rnd_idx,
                    )
                    cutsky_randoms.apply_angular_mask(region=region, release=release, npasses=None, program='dark')
                    # we use `shape_only=True` to only match the n(z) shape, keeping the randoms amplitude
                    cutsky_randoms.apply_radial_mask(nz_filename=nz_filename, shape_only=True)
                    save_dir = Path(args.save_dir)
                    cutsky_randoms.save(save_dir / f'{tracer}_{region}_{rnd_idx}.ran.fits')
