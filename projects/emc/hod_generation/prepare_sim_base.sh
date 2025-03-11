#!/bin/bash
#SBATCH --account=desi
#SBATCH -q preempt
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -c 256
#SBATCH --array=3-24

leading_zero_fill ()
{
    # print the number as a string with a given number of leading zeros
    printf "%0$1d\\n" "$2"
}

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
PATH2CONFIG=~/code/acm/projects/emc/hod_generation/abacushod_config_base.yaml
REDSHIFTS=( 0.800 )
# COSMO=000
# PHASE=002

PHASE=$(leading_zero_fill 3 $SLURM_ARRAY_TASK_ID)
# COSMO=$(leading_zero_fill 3 $SLURM_ARRAY_TASK_ID)
COSMO=000

# SLURM_ARRAY_TASK_ID=0

# N_PHASE=20
# START_PHASE=$((SLURM_ARRAY_TASK_ID * N_PHASE + 3000))
# END_PHASE=$((START_PHASE + N_PHASE))
# START_PHASE=$(leading_zero_fill 3 0)
# END_PHASE=$(leading_zero_fill 3 0)

# set -e
# for PHASE in {3000..5000}; do
# for PHASE in $( seq $START_PHASE $END_PHASE ); do
# for PHASE in {002..024}; do
for ALT_Z in "${REDSHIFTS[@]}"; do
    ALT_SIMNAME=AbacusSummit_base_c"$COSMO"_ph"$PHASE"
    # HALOS=$BASE_DIR/$ALT_SIMNAME/z$ALT_Z/halos_xcom_0_seed600_abacushod_oldfenv_new.h5
    # PARTICLES=$BASE_DIR/$ALT_SIMNAME/z$ALT_Z/particles_xcom_0_seed600_abacushod_oldfenv_withranks_new.h5

    # if [[ -f $HALOS && -f $PARTICLES ]]; then
    #     echo "Skipping $ALT_SIMNAME $ALT_Z because it already exists"
    #     continue
    # fi
    echo "Preparing $ALT_SIMNAME $ALT_Z"

    python -m abacusnbody.hod.prepare_sim --path2config $PATH2CONFIG --alt_simname $ALT_SIMNAME --alt_z $ALT_Z
done
# done

