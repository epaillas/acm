#!/bin/bash
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -c 256
#SBATCH --array=0-0

leading_zero_fill ()
{
    # print the number as a string with a given number of leading zeros
    printf "%0$1d\\n" "$2"
}

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
BASE_DIR=/pscratch/sd/e/epaillas/summit_subsamples/lightcones
PATH2CONFIG=~/code/acm/projects/desi/ds_abacus_lightcone.yaml
# REDSHIFTS=( 0.100 0.150 0.200 0.250 0.300 0.350 0.400 0.450 0.500 )  # LOWZ
# REDSHIFTS=( 0.575 0.650 0.725 0.800 0.875 0.950 1.025 1.100 )  # DESI LRG
REDSHIFTS=( 0.800 0.875 0.950 1.025 1.100 )  # DESI LRG
# PHASE=000

# PHASE=$(leading_zero_fill 3 $SLURM_ARRAY_TASK_ID)
PHASE=000
# COSMO=$(leading_zero_fill 3 $SLURM_ARRAY_TASK_ID)

set -e
for COSMO in {000..000}; do
    for ALT_Z in "${REDSHIFTS[@]}"; do
        ALT_SIMNAME=AbacusSummit_base_c"$COSMO"_ph"$PHASE"
        HALOS=$BASE_DIR/$ALT_SIMNAME/z$ALT_Z/halos_xcom_0_seed600_abacushod_oldfenv_new.h5
        PARTICLES=$BASE_DIR/$ALT_SIMNAME/z$ALT_Z/particles_xcom_0_seed600_abacushod_oldfenv_withranks_new.h5

        if [[ -f $HALOS && -f $PARTICLES ]]; then
            echo "Skipping $ALT_SIMNAME $ALT_Z because it already exists"
            continue
        fi
        echo "Preparing $ALT_SIMNAME $ALT_Z"

        python -m abacusnbody.hod.prepare_sim --path2config $PATH2CONFIG --alt_simname $ALT_SIMNAME --alt_z $ALT_Z
    done
done

