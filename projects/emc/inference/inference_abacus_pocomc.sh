#!/bin/bash
#SBATCH --account=desi_g
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus-per-task 1
#SBATCH -n 1
#SBATCH --array=100-120

leading_zero_fill ()
{
    # print the number as a string with a given number of leading zeros
    printf "%0$1d\\n" "$2"
}

ecosmodesi

COSMO_IDX=0
HOD_IDX=$(leading_zero_fill 3 $SLURM_ARRAY_TASK_ID)

python /global/u1/e/epaillas/code/acm/projects/emc/inference/inference_abacus_pocomc.py --cosmo_idx $COSMO_IDX --hod_idx $HOD_IDX