"""
Fiber assignment utilities for DESI mock catalogs.

This module provides a high-level Python interface to run the DESI
`fiberassign` pipeline on pre-loaded mock catalogs. It is adapted 
from the DESI `fiberassign` codebase and the `mockfactory`, with
MPI-parallel execution, multi-pass assignment, and direct
in-memory target handling.

The main entry point is :func:`apply_fiber_assignment`, which applies
fiber assignment pass-by-pass and tile-by-tile, updating target
properties such as assigned fibers, observation counts, availability,
and completeness weights.



Main features
-------------
- Builds DESI tiles from surveyops and release catalogs
- Runs fiber assignment tile-by-tile to ensure correct hardware state
- Supports multi-pass DESI observing strategy
- Uses precomputed sky targets from fiberassign outputs (optional)
- MPI-parallelized over tiles within each pass
- Computes completeness weights for clustering analyses
- Designed for mock and random catalogs produced with `mockfactory`

MPI behavior
------------
Targets are expected to be distributed across MPI ranks, while the full
tile list is available on each rank. Particle exchange is used to
balance tiles across ranks during each pass.

Environment requirements
------------------------
This module assumes access to DESI software stacks and file systems,
including:
- DESI_TARGET
- DESIMODEL
- fiberassign
- desitarget
- desimodel

These environment variables are set explicitly at import time for
Perlmutter/DVS environments.

Notes
-----
- Fiber assignment is applied pass-by-pass; within a pass, each target
  can be observed at most once.
- Sky targets are read from precomputed fiberassign files rather than
  the full DR9 sky catalog, significantly improving I/O performance.
- The code closely follows DESI fiberassign internals and may require
  updates if upstream APIs change.

Authors
-------
Rocher Antoine
Code adapted from mockfactory/desi github repository
Primary contributors include Edmond Chaussidon, Anand Raichoor and 
DESI Collaboration members

"""

import os
import logging

import fitsio
import numpy as np
import pandas as pd
from mpytools import Catalog, setup_logging
from mpi4py import MPI
import mockfactory

logger = logging.getLogger('F.A.')
setup_logging()

def prepare_cat_for_FA_using_randoms(cutsky_mock):
    """
    Prepare a mock catalog for fiber assignment.
    Add required columns to run Fiber Assignment on catalog

    Parameters
    ----------
    cutsky_mock : mpytools.Catalog
        Mock catalog to prepare (modified in place).
    """



    types = ['ELG', 'LRG', 'QSO']
    priority = {'ELG':3200, 'LRG':3200, 'QSO':3400} # ELG 3000 if not LRG prio
    numobs = {'ELG':2, 'LRG':2, 'QSO':4}
    desitar = {'ELG':2**1, 'LRG':2**0, 'QSO':2**2}

    names_col = ['PRIORITY_INIT', 'PRIORITY', 'NUMOBS_MORE', 'NUMOBS_INIT', 'DESI_TARGET', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'MASKBITS']
    for name in names_col:
        cutsky_mock[name] = cutsky_mock.ones(dtype=int)
        
    for type_ in types: 
        mask= cutsky_mock['TRACER'] == type_
        cutsky_mock['PRIORITY_INIT'][mask] = priority[type_]
        cutsky_mock['PRIORITY'][mask] = priority[type_]
        cutsky_mock['NUMOBS_MORE'][mask] = numobs[type_]
        cutsky_mock['NUMOBS_INIT'][mask] = numobs[type_]
        cutsky_mock['DESI_TARGET'][mask] = desitar[type_]

    n = cutsky_mock.size

    cutsky_mock['BGS_TARGET'] = cutsky_mock.zeros(dtype='i8')
    cutsky_mock['MWS_TARGET'] = cutsky_mock.zeros(dtype='i8')
    cutsky_mock['SUBPRIORITY'] = np.random.uniform(0, 1, n)
    cutsky_mock['OBSCONDITIONS'] = cutsky_mock.ones(dtype='i8') #np.zeros(n, dtype='i8')+int(3) 
    cutsky_mock['SCND_TARGET'] = cutsky_mock.zeros(dtype='i8')
    cutsky_mock['ZWARN'] = cutsky_mock.zeros(dtype='i8')
    cutsky_mock['TARGETID'] = np.random.permutation(np.arange(1,n+1)) + cutsky_mock.mpicomm.rank * (10**6)

    maskcols = ['NOBS_G','NOBS_R','NOBS_Z','MASKBITS']
    for col in maskcols:
        cutsky_mock[col] = cutsky_mock.ones(dtype='i8')

def _run_mask(args, cat):
    fn, nob_fn, idx = args[0], args[1], args[2]
    bitmask_img, header = fitsio.read(fn, header=True)
    coadd_x, coadd_y = np.round(wcs.WCS(header).wcs_world2pix(cat['RA'][idx], cat['DEC'][idx], 0)).astype(int)
    bitmask = bitmask_img[coadd_y, coadd_x]
    nobs_g, nobs_r, nobs_z = [fitsio.read(nob_fn.format(band))[coadd_y, coadd_x] for band in 'grz']
    return bitmask, nobs_g, nobs_r, nobs_z


def get_maskbit_nobs(cat, nproc = 256, path_to_bricks='/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/randoms/survey-bricks-dr9-randoms-0.48.0.fits', 
    bitmask_dir = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/'):

    """
    Assign imaging maskbits and number-of-observations to targets.

    Uses Legacy Survey coadds and multiprocessing for performance.

    Parameters
    ----------
    cat : mpytools.Catalog
        Input catalog.
    nproc : int, default=256
        Number of processes.
    path_to_bricks : str
        Path to survey bricks file.
    bitmask_dir : str
        Directory containing maskbit and nobs files.

    Returns
    -------
    res : list
        Per-brick maskbit and nobs results.
    """

    from desiutil import brick
    import time
    from multiprocessing import Pool
    import fitsio
    from astropy import wcs
    from functools import partial

    bricks = Catalog.read(path_to_bricks)

    tmp = brick.Bricks(bricksize=0.25)
    cat['BRICKID'] = tmp.brickid(cat['RA'], cat['DEC'])
    cat['BRICKNAME']  = tmp.brickname(cat['RA'], cat['DEC'])
    brickid, bidcnts = np.unique(cat['BRICKID'], return_counts=True)
    bidcnts = np.insert(bidcnts, 0, 0)
    bidcnts = np.cumsum(bidcnts)
    bidorder = np.argsort(cat['BRICKID'])
    single_brick_id = np.in1d(bricks['BRICKID'], brickid)
    bricknames = bricks['BRICKNAME'][single_brick_id]
    regions = bricks['PHOTSYS'][single_brick_id]
    files = [bitmask_dir+'{}/coadd/{}/{}/legacysurvey-{}-maskbits.fits.fz'.format('south' if reg == 'S' else 'north', brickname[:3], brickname, brickname) for brickname, reg in zip(bricknames, regions)]
    nobs_fns = [bitmask_dir+'{}/coadd/{}/{}/legacysurvey-{}-nexp-{{}}.fits.fz'.format('south' if reg == 'S' else 'north', brickname[:3], brickname, brickname, None) for brickname, reg in zip(bricknames, regions)]
    idxs = [bidorder[bidcnts[bid_index]: bidcnts[bid_index+1]] for bid_index in np.arange(len(brickid))]

    print('Initiate Pool with {} processes'.format(nproc), flush=True)
    st=time.time()
    pool = Pool(processes=nproc)
    print('Pool initiated in {} sec'.format(time.time()-st), flush=True)
    
    mask = regions != ''
    print('Run maskbit and nobs assignement', flush=True)
    st=time.time()
    cat['MASKBITS'], cat['NOBS_G'], cat['NOBS_R'], cat['NOBS_Z'] = [np.zeros(len(cat['RA']))]*4
    res = pool.map(partial(_run_mask, cat=cat), zip(np.array(files)[mask], np.array(nobs_fns)[mask], [a for a,b in zip(idxs,mask) if b]))
    # cat['MASKBITS'][mask], cat['NOBS_G'][mask], cat['NOBS_R'][mask], cat['NOBS_Z'][mask] = np.hstack([np.vstack(rr) for rr in res])
    # cat['MASKBITS'][mask], cat['NOBS_G'][mask], cat['NOBS_R'][mask], cat['NOBS_Z'][mask] = np.hstack([np.vstack(rr) for rr in res])
    pool.close()
    print('Done in {} sec'.format(time.time()-st), flush=True)
    return res


def run_FA(cutsky, release='Y3', program='dark', npasses=1, use_sky_targets=False, add_random_tracers=False, tracer='LRG', seed=None,
           preload_sky_targets=False, plot_output=True, path_to_save=None):

    """
    Run a full fiber assignment workflow on a mock cutsky catalog.

    Optionally adds other tracers randomly, applies multi-pass fiber assignment,
    computes completeness weights, and produces diagnostic plots.

    Parameters
    ----------
    cutsky : mpytools.Catalog
        Input mock catalog.
    release : str, default='Y3'
        DESI data release.
    program : str, default='dark'
        Observing program.
    npasses : int, default=1
        Number of observing passes.
    use_sky_targets : bool, default=False
        Include sky targets.
    add_random_tracers : bool, default=False
        Add other tracers randomly.
    tracer : str, default='LRG'
        Input tracer in the catalog.
    seed : int or None
        RNG seed.
    preload_sky_targets : bool, default=False
        Preload sky targets.
    plot_output : bool, default=True
        Produce diagnostic plots.
    path_to_save : str or None
        Output catalog path.

    Returns
    -------
    cutsky_for_fa : mpytools.Catalog
        Catalog with fiber assignment results.
    """


    
    # from mockfactory.desi import is_in_desi_footprint
    from mockfactory.desi import is_in_desi_footprint, apply_fiber_assignment, compute_completeness_weight, build_tiles_for_fa, read_sky_targets

    
    if not isinstance(cutsky, Catalog):
        raise ValueError('cutsky should be either a path to a fits file or a mockfactory Catalog object')
    
    mpicomm = cutsky.mpicomm
    if 'TRACER' not in  cutsky.columns():
        if mpicomm.rank == 0:logger.info('Add tracer column for LRGs.')
        cutsky['TRACER'] = ['LRG']*cutsky.size

        
    if mpicomm.rank == 0: 
        logger.info('Run simple example to illustrate how to run fiber assignment.')
        logger.info(f'Add random ELGs and QSOs objects.')

    if add_random_tracers:
        # This is should be done better 
        tr_toadd = ['LRG','ELG', 'QSO']
        tr_toadd.remove(tracer)
        nbar1 = 240 if tr_toadd[0] == 'ELG' else 310 if tr_toadd[0] == 'QSO' else 610
        nbar2 = 240 if tr_toadd[1] == 'ELG' else 310 if tr_toadd[1] == 'QSO' else 610
        np.random.seed(seed)
        seed1, seed2 = np.random.randint(0,2**32-1,size=2)
        if mpicomm.rank == 0: 
            logger.info('Create random catalogues.')
        cutsky_1 = mockfactory.RandomCutskyCatalog(rarange=(cutsky['RA'].min(), cutsky['RA'].max()), decrange=(cutsky['DEC'].min(), cutsky['DEC'].max()), drange=(cutsky['Distance'].min(), cutsky['Distance'].max()), nbar= nbar1, seed=seed1, mpicomm=mpicomm)
        cutsky_2 = mockfactory.RandomCutskyCatalog(rarange=(cutsky['RA'].min(), cutsky['RA'].max()), decrange=(cutsky['DEC'].min(), cutsky['DEC'].max()), drange=(cutsky['Distance'].min(), cutsky['Distance'].max()), nbar=nbar2, seed=seed2, mpicomm=mpicomm)
        if mpicomm.rank == 0: 
            logger.info('Randon catalogues generated.')
        cutsky_1['TRACER'] = [tr_toadd[0]]*cutsky_1.size
        cutsky_2['TRACER'] = [tr_toadd[1]]*cutsky_2.size
        cutsky_2 = cutsky_2[is_in_desi_footprint(cutsky_2['RA'], cutsky_2['DEC'], release=release, program=program, npasses=npasses)]
        cutsky_1 = cutsky_1[is_in_desi_footprint(cutsky_1['RA'], cutsky_1['DEC'], release=release, program=program, npasses=npasses)]
        if mpicomm.rank == 0: 
            logger.info('Cut desi FP')
        cutsky_for_fa = Catalog.concatenate([cutsky[cutsky_1.columns()], cutsky_1, cutsky_2])
        if mpicomm.rank == 0: 
            logger.info('Concatenate catalogues')
    else:
        cutsky_for_fa = cutsky
    
    if mpicomm.rank == 0: logger.info(f'Prepare catalog for fiber assignment')
    prepare_cat_for_FA_using_randoms(cutsky_for_fa)

    # Collect tiles from surveyops directory on which the fiber assignment will be applied
    
    tiles = build_tiles_for_fa(release_tile_path=f'/global/cfs/cdirs/desi/survey/catalogs/{release}/LSS/tiles-{program.upper()}.fits', program=program, npasses=npasses)

    if use_sky_targets and preload_sky_targets:
        # tiles is not restricted here, we will load sky_targets for all the Y1 footprint
        sky_targets = read_sky_targets(dirname='/global/cfs/cdirs/desi/users/edmondc/desi_targets/sky_targets/', tiles=tiles, program=program, mpicomm=mpicomm)

    # Get info from origin fiberassign file and setup options for F.A.
    ts = str(tiles['TILEID'][0]).zfill(6)
    fht = fitsio.read_header(f'/global/cfs/cdirs/desi/target/fiberassign/tiles/trunk/{ts[:3]}/fiberassign-{ts}.fits.gz')
    rundate = fht['RUNDATE']
    # see fiberassign.scripts.assign.parse_assign (Can modify margins, number of sky fibers for each petal etc.)
    opts_for_fa = ["--target", " ", "--rundate", rundate, "--mask_column", "DESI_TARGET"]
    
    nbr_targets = cutsky_for_fa.csize
    if mpicomm.rank == 0: logger.info(f'Keep only objects which is in a tile. Working with {nbr_targets} targets')

    columns_for_fa = ['RA', 'DEC', 'TARGETID', 'DESI_TARGET', 'SUBPRIORITY', 'OBSCONDITIONS', 'NUMOBS_MORE']

    # Let's do the F.A.:
    if mpicomm.rank == 0: logger.info(f'Running fiber assignment')

    apply_fiber_assignment(cutsky_for_fa, tiles, npasses, opts_for_fa, columns_for_fa, mpicomm, use_sky_targets=use_sky_targets)
    # Compute the completeness weight: if multi-tracer, apply completeness weight once for each tracer independently
    compute_completeness_weight(cutsky_for_fa, tiles, npasses, mpicomm)
    # Summarize and plot
    ra, dec = cutsky_for_fa.cget('RA', mpiroot=0), cutsky_for_fa.cget('DEC', mpiroot=0)
    numobs, available = cutsky_for_fa.cget('NUMOBS', mpiroot=0), cutsky_for_fa.cget('AVAILABLE', mpiroot=0)
    obs_pass, comp_weight = cutsky_for_fa.cget('OBS_PASS', mpiroot=0), cutsky_for_fa.cget('COMP_WEIGHT', mpiroot=0)

    if mpicomm.rank == 0:logger.info('FA done')

    if add_random_tracers:
        mask_tr = cutsky_for_fa['TRACER']== tracer
        cutsky_for_fa = cutsky_for_fa[mask_tr]
        for col in cutsky.columns():
            if col not in cutsky_for_fa.columns():
                cutsky_for_fa[col] = cutsky[col]
                
    
    if path_to_save is not None:
        if mpicomm.rank == 0:logger.info(f'Save catalog with FA information to {path_to_save}')
        cutsky_for_fa.write(path_to_save)
    
    if plot_output & (mpicomm.rank == 0):

        logger.info(f"Nbr of targets observed: {(numobs >= 1).sum()} -- per pass: {obs_pass.sum(axis=0)} -- Nbr of targets available: {available.sum()} -- Nbr of targets: {ra.size}")
        logger.info(f"In percentage: Observed: {(numobs >= 1).sum()/ra.size:2.2%} -- Available: {available.sum()/ra.size:2.2%}")
        values, counts = np.unique(comp_weight, return_counts=True)
        logger.info(f'Sanity check for completeness weight: {available.sum() - (numobs >= 1).sum()} avialable unobserved targets and {np.nansum([(val - 1) * count for val, count in zip(values, counts)])} from completeness counts')
        logger.info(f'Completeness counts: {values} -- {counts}')

        import skyproj
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 12))
        # sp.draw_des(label='DES', edgecolor='k')
        sp = skyproj.DESSkyproj(ax=ax, extent=[145-5,175+5,30-5, 45+5], fontsize=8)


        sp.scatter(ra[available], dec[available], s=0.0001, c='k')
        sp.scatter(ra[numobs>0], dec[numobs>0], s=0.001,c='r')

        fig.tight_layout()
        fig.savefig(f'fiberasignment_{npasses}npasses.png', facecolor='w', bbox_inches='tight', pad_inches=0.2, dpi=400)
        logger.info(f'Plot save in fiberasignment_{npasses}npasses.png')
    mpicomm.Barrier()

    return cutsky_for_fa
