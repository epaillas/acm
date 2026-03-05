import numpy as np
from acm.hod import BoxHOD
from acm.utils import setup_logging
import logging
from pathlib import Path
import pandas
import argparse

setup_logging()

parser = argparse.ArgumentParser(
                    prog='generate_hods',
                    description='Generate HOD mocks for training set')
parser.add_argument('--N_hod', type=float, default=500, help='Number of mocks to generate for each cosmology')
parser.add_argument('--redshift', type=float, default=0.5, help='Redshift of cubic box')
parser.add_argument('--nbar_min', type=float, default=4.985e-4, 
                    help='Minimum number density cut to apply (defaults to 6-sigma)')
parser.add_argument('--nbar_max', type=float, default=5e-4, 
                    help='Maximum number density threshold for downsampling')
parser.add_argument('--process_underdense', required=False, default=False, action='store_true',
                    help='Whether to process catalgoues that are below the minimum density threshold')
parser.add_argument('--chunk', nargs='+', type=int, required=False, default=None, 
                    help='Divide cosmologies for simultaneous batch jobs')
parser.add_argument('--save_dir', type=str, required=False, default='/pscratch/sd/n/ntbfin/emulator/hods/v1.4/',
                    help='Directory used to store mocks')
args = parser.parse_args()

redshift = args.redshift
tracer_density_mean = (args.nbar_min, args.nbar_max)  # lower and upper thresholds (lower defaults to 6-sigma)
process_underdense = args.process_underdense

N_hod = args.N_hod
phases = list(range(1))
seeds = list(range(1))
cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))

if args.chunk is not None: cosmos = cosmos[args.chunk[0]:args.chunk[1]]

# load hod parameters sampled from Yuan2022 prior (generated using acm/hod/parameters.py)
param_dir = "/pscratch/sd/n/ntbfin/emulator/hods/"
hod_params_all = np.load(f"{param_dir}hod_params.npy", allow_pickle=True)[()]

for cosmo_idx in cosmos:
    hod_params = hod_params_all[f'c{cosmo_idx:03d}']
    hods = range(len(hod_params['logM_cut']))
       
    n_gal = np.zeros(len(hods))
    for phase_idx in phases:
        # load abacusHOD class
        abacus = BoxHOD(varied_params=hod_params.keys(),
                        sim_type='base', redshift=redshift,
                        cosmo_idx=cosmo_idx, phase_idx=phase_idx)

        for seed in seeds:
            save_dir = Path(args.save_dir) / f'z{redshift:.1f}/yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed{seed}/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            N = 1
            for hod_idx in hods:
                if N > N_hod: continue
                hod = {key: hod_params[key][hod_idx] for key in hod_params.keys()}
    
                save_fn = Path(save_dir) / f'hod{hod_idx:03}.fits'
                # sample HODs and save to disk
                abacus.run(hod, nthreads=64, tracer_density=tracer_density, process_underdense=process_underdense,
                           add_ap=True, seed=seed, save_fn=save_fn, save_distortions=True)
                n_gal[hod_idx] = abacus.n_gal
                N += abacus.in_density

    np.save(f"{param_dir}number_density/n_gal_c{cosmo_idx:03d}.npy", n_gal)
                
