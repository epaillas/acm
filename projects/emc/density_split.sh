#!/bin/bash
#SBATCH --account desi_g
#SBATCH --constraint gpu
#SBATCH -q regular
#SBATCH -t 10:00:00
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --array=1-9

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export NUMEXPR_MAX_THREADS=4

N_HOD=1000
START_HOD=$((SLURM_ARRAY_TASK_ID * N_HOD))

python /global/u1/e/epaillas/code/acm/projects/emc --start_hod $START_HOD --n_hod $N_HOD
