#!/bin/bash -l

#SBATCH --nodes 1
#SBATCH --constraint cpu
#SBATCH --qos regular
#SBATCH --account desi

#SBATCH --time 02:30:00

#SBATCH --job-name prepare_sim
#SBATCH --output /pscratch/sd/s/sbouchar/Output_jobs/prepare_sim/base/%A.%x_%a.out
#SBATCH --error /pscratch/sd/s/sbouchar/Output_jobs/prepare_sim/base/%A.%x_%a.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Warning on branch & code version
echo "Warning: This script is meant to run on the bgs_prep branch of https://github.com/SBouchard01/abacusutils installed locally."

# Setup
COSMO_LIST=(0 {1..4} 13 {100..126} {130..181}) # List of cosmologies to be used
PATH2CONFIG=/global/homes/s/sbouchar/acm/scripts/bgs/prepare_simulations/config.yaml
LOG_FOLDER=/pscratch/sd/s/sbouchar/summit_subsamples/logs/

# Get cosmo-dependent parameters
ID=$((SLURM_ARRAY_TASK_ID)) # ID of the cosmology to be used, starting from 0
COSMO=${COSMO_LIST[$ID]} # Cosmology to be used
SIMNAME=$(printf "AbacusSummit_base_c%03d_ph000" ${COSMO}) # Simulation name
PATH2LOG=${LOG_FOLDER}${SIMNAME}.log

cd '/global/homes/s/sbouchar/01- Packages/abacusutils/' # Needed to avoid ImportErrors due to relative imports
python -m abacusnbody.hod.prepare_sim --path2config $PATH2CONFIG --path2log $PATH2LOG --alt_simname $SIMNAME --overwrite 0

# Launch with : sbatch --array=0-84 ...


# Tests
# PATH2CONFIG=/global/homes/s/sbouchar/acm/scripts/bgs/prepare_simulations/config.yaml
# PATH2LOG=/pscratch/sd/s/sbouchar/summit_subsamples_v2/logs/AbacusSummit_base_c000_ph000.log
# cd '/global/homes/s/sbouchar/01- Packages/abacusutils/'
# python -m abacusnbody.hod.prepare_sim --path2config $PATH2CONFIG --path2log $PATH2LOG