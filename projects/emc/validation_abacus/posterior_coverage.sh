#!/bin/bash
#SBATCH --account desi
#SBATCH --constraint cpu
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH --array=0-4,13

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

COSMO_IDX=$((SLURM_ARRAY_TASK_ID))

for HOD_IDX in {0..100}; do

    # HOD_IDX=$(($RANDOM % 100))

    python /global/u1/e/epaillas/code/acm/projects/emc/validation_abacus/sample_hmc.py \
    --cosmo_idx $COSMO_IDX --hod_idx $HOD_IDX

done