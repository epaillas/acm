"""
Run this with MPI on <nproc> tasks as 
srun -n <nproc> python test_fa.py
"""

from acm.hod import CutskyHOD, CutskyRandoms
from acm import setup_logging
from pyrecon.utils import sky_to_cartesian
import numpy as np
from pathlib import Path
import pandas
from cosmoprimo.fiducial import AbacusSummit
from mockfactory.desi import build_tiles_for_fa, apply_fiber_assignment
from mpi4py import MPI
import fitsio
from mpytools import Catalog

mpicomm = MPI.COMM_WORLD
setup_logging(level=('info' if mpicomm.rank == 0 else 'warning'))


def get_hod_params(nrows=None):
    """Some example HOD parameters."""
    hod_dir = Path(f'/pscratch/sd/e/epaillas/emc/hod_params/yuan23/')
    hod_fn = hod_dir / f'hod_params_yuan23_c000.csv'
    df = pandas.read_csv(hod_fn, delimiter=',')
    df.columns = df.columns.str.strip()
    df.columns = list(df.columns.str.strip('# ').values)
    return df.to_dict('list')


if __name__ == '__main__':
    # redshifts of the snapshots that will be used to build the cutsky
    snapshots = [0.5]

    # redshift range (in the cutsky) that will be covered by each snapshot
    zranges = [(0.4, 0.6)]

    # fiducial cosmology for the redshift-distance relation and RSD
    cosmo = AbacusSummit(0)

    # read example HOD parameters
    hod_params = get_hod_params()

    # initialize class
    cutsky = CutskyHOD(varied_params=hod_params.keys(),
                    zranges=zranges, snapshots=snapshots,
                    cosmo_idx=0, phase_idx=0,
                    load_existing_hod=True)

    # sample HOD parameters and build the cutsky mock
    hod = {key: hod_params[key][30] for key in hod_params.keys()}
    cutsky.sample_hod(hod, nthreads=1, region='NGC', release='Y1', program='dark')

    # apply angular and radial masks
    cutsky.apply_angular_mask(region='NGC', release='Y1', npasses=None, program='dark')
    nz_filename='/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/LRG_NGC_nz.txt'
    cutsky.apply_radial_mask(nz_filename=nz_filename)

    # add columns that are needed by fiber assignment
    cutsky.add_columns_fiberassign()

    # save the cutsky catalog to disk (optional, for debugging)
    # cutsky.save('cutsky.npy')
    # cutsky = np.load('cutsky.npy', allow_pickle=True).item()

    data = Catalog(cutsky.catalog, mpicomm=mpicomm)

    # need to make the TARGETID unique across all processes
    cumsize = np.cumsum([0] + mpicomm.allgather(data.size))[mpicomm.rank]
    data['TARGETID'] = cumsize + np.arange(data.size)

    npasses = 1

    # Collect tiles from surveyops directory on which the fiber assignment will be applied
    tiles = build_tiles_for_fa(release_tile_path=f'/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/LSS/tiles-DARK.fits', program='dark', npasses=npasses)  
    # Get info from origin fiberassign file and setup options for F.A. (see fiberassign.scripts.assign.parse_assign to modify margins, number of sky fibers for each petal ect...)
    ts = str(tiles['TILEID'][0]).zfill(6)
    fht = fitsio.read_header(f'/global/cfs/cdirs/desi/target/fiberassign/tiles/trunk/{ts[:3]}/fiberassign-{ts}.fits.gz')
    opts_for_fa = ["--target", " ", "--rundate", fht['RUNDATE'], "--mask_column", "DESI_TARGET"]
    # columns needed to run the F.A. and collect the info (They will be exchange between processes during the F.A.)
    columns_for_fa = ['RA', 'DEC', 'TARGETID', 'DESI_TARGET', 'SUBPRIORITY', 'OBSCONDITIONS', 'NUMOBS_MORE']

    # run fiber assignment
    apply_fiber_assignment(data, tiles, npasses, opts_for_fa, columns_for_fa, mpicomm, use_sky_targets=True)





