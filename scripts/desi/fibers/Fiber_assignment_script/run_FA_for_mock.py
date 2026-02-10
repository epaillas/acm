
from mockfactory import Catalog, setup_logging
from fiber_assignment import run_FA
from mpi4py import MPI
import logging
import numpy as np
import argparse


logger = logging.getLogger('F.A.')
setup_logging()

argparser = argparse.ArgumentParser(description='Run Fiber Assignment on mockfactory cutsky mock')
argparser.add_argument('--input_catalog', type=str, help='Path to input cutsky catalog')
argparser.add_argument('--tracer', type=str, help='Tracer used choices: ELG, LRG, QSO', choices=['LRG', 'ELG', 'QSO'], default='LRG')
argparser.add_argument('--output_catalog', type=str, help='Path to output cutsky catalog with FA results')
argparser.add_argument('--npasses', type=int, help='Number of passes for fiber assignment', default=7)
argparser.add_argument('--release', type=str, help='Survey release', default='Y3')
argparser.add_argument('--program', type=str, help='Survey program', default='dark')
argparser.add_argument('--plot_output', action='store_true', help='Whether to plot output')
argparser.add_argument('--add_random_tracers', action='store_true', help='Add random tracers (ELG and QSO) to mimic real target selection')

args = argparser.parse_args()

logger.info('Test run FA on mockfactory cutsky mock with random ELG and QSO targets')
mpicomm = MPI.COMM_WORLD
cutsky = Catalog.read(args.input_catalog, mpicomm=mpicomm)
cutsky_lrg_fa = run_FA(cutsky, release=args.release, program=args.program, npasses=args.npasses, tracer=args.tracer, plot_output=args.plot_output, add_random_tracers=args.add_random_tracers, path_to_save=args.output_catalog)

