#!/bin/bash -l

#SBATCH --nodes 1
#SBATCH --constraint cpu
#SBATCH --qos regular
#SBATCH --account desi

#SBATCH --time 02:00:00

#SBATCH --job-name prep_sim
#SBATCH --output /global/homes/s/sbouchar/ACM_pipeline/prepare_simulations/Out_jobs_tmp/%A.%x_%a.out
#SBATCH --error /global/homes/s/sbouchar/ACM_pipeline/prepare_simulations/Out_jobs_tmp/%A.%x_%a.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Parameters for the script (Can be given when launching the job !)
COSMO_LIST=(0 {1..4} 13 {100..126} {130..181}) # List of cosmologies to be used
PATH2CONFIG='/global/homes/s/sbouchar/ACM_pipeline/prepare_simulations/config_c000.yaml'
LOG_PATH='/global/homes/s/sbouchar/ACM_pipeline/prepare_simulations/logs_cosmos/'

ID=$((SLURM_ARRAY_TASK_ID)) # ID of the cosmology to be used, starting from 0
COSMO=${COSMO_LIST[$ID]} # Cosmology to be used
LOG=$(printf "log_c%03d.log" ${COSMO}) # Log file
SIMNAME=$(printf "AbacusSummit_base_c%03d_ph000" ${COSMO}) # Simulation name

PATH2LOG=${LOG_PATH}${LOG}

# Just to check, not necessary
echo "Launching ${SIMNAME} with config file ${PATH2CONFIG} and log file ${PATH2LOG}"

python /global/homes/s/sbouchar/ACM_pipeline/prepare_simulations/prepare_sim_logged_bgs.py --path2config $PATH2CONFIG --path2log $PATH2LOG --alt_simname $SIMNAME --overwrite 0 

# Launch with : sbatch --array=0-84 /global/homes/s/sbouchar/ACM_pipeline/prepare_simulations/prepare_sim_logged_bgs.sh