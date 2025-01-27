#!/bin/bash
#SBATCH --account desi_g
#SBATCH --constraint gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --array=0-4,13-13,100-114

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module swap pyrecon/mpi pyrecon/main
export NUMEXPR_MAX_THREADS=4

N_HOD=50
START_HOD=50
START_COSMO=$((SLURM_ARRAY_TASK_ID))

python /global/u1/e/epaillas/code/acm/projects/emc/dsc_abacus.py --start_cosmo $START_COSMO --start_hod $START_HOD --n_hod $N_HOD
