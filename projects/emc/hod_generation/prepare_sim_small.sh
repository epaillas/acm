#!/bin/bash
#SBATCH --account=desi
#SBATCH -q preempt
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -c 256
#SBATCH --array=0-99

# leading_zero_fill ()
# {
#     # print the number as a string with a given number of leading zeros
#     printf "%0$1d\\n" "$2"
# }

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
BASE_DIR=/pscratch/sd/e/epaillas/tmp/summit_subsamples/boxes/small/
PATH2CONFIG=~/code/emc/abacushod_cov_config.yaml
REDSHIFTS=( 1.100 )
COSMO=000

# PHASE=$(leading_zero_fill 3 $SLURM_ARRAY_TASK_ID)
# COSMO=$(leading_zero_fill 3 $SLURM_ARRAY_TASK_ID)

# SLURM_ARRAY_TASK_ID=0

N_PHASE=20
# START_PHASE=$((SLURM_ARRAY_TASK_ID * N_PHASE + 3000))
# END_PHASE=$((START_PHASE + N_PHASE))
START_PHASE=3000
END_PHASE=3001

# set -e
# for PHASE in {3000..5000}; do
for PHASE in $( seq $START_PHASE $END_PHASE ); do
    for ALT_Z in "${REDSHIFTS[@]}"; do
        ALT_SIMNAME=AbacusSummit_small_c"$COSMO"_ph"$PHASE"
        # HALOS=$BASE_DIR/$ALT_SIMNAME/z$ALT_Z/halos_xcom_0_seed600_abacushod_oldfenv_new.h5
        # PARTICLES=$BASE_DIR/$ALT_SIMNAME/z$ALT_Z/particles_xcom_0_seed600_abacushod_oldfenv_withranks_new.h5

        # if [[ -f $HALOS && -f $PARTICLES ]]; then
        #     echo "Skipping $ALT_SIMNAME $ALT_Z because it already exists"
        #     continue
        # fi
        echo "Preparing $ALT_SIMNAME $ALT_Z"

        python -m abacusnbody.hod.prepare_sim --path2config $PATH2CONFIG --alt_simname $ALT_SIMNAME --alt_z $ALT_Z
    done
done

