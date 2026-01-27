#!/bin/bash -l

#SBATCH --account desi_g
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --qos regular
#SBATCH --constraint gpu&hbm80g

#SBATCH --time 10:00:00

#SBATCH --job-name 10-bgs_base
#SBATCH --output /pscratch/sd/s/sbouchar/Output_jobs/bgs-21.35_measurements/%A.%x_%a.out
#SBATCH --error /pscratch/sd/s/sbouchar/Output_jobs/bgs-21.35_measurements/%A.%x_%a.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Load the old pyrecon module for densitysplit
module swap pyrecon/mpi pyrecon/main

# Get the cosmology index from the SLURM_ARRAY_TASK_ID
COSMO_LIST=(0 {1..4} 13 {100..126} {130..181}) # List of cosmologies to be used
ID=$((SLURM_ARRAY_TASK_ID)) # ID of the cosmology to be used, starting from 0
COSMO=${COSMO_LIST[ID]} # Cosmology to be used

OVERRIDE=$(printf "/pscratch/sd/s/sbouchar/acm/bgs-20/parameters/override/c%03d.npy" ${COSMO}) # Ensure the same HODs are used for bgs-21.35 as for bgs-20
LOGFILE=$(printf "/pscratch/sd/s/sbouchar/acm/bgs-21.35/measurements/logs/log_base_c%03d_ph000_seed0.log" ${COSMO})

cd /global/homes/s/sbouchar/acm-repo/scripts/bgs/measurements
python measure_box.py --config jobs/bgs-21.35/config.yaml --gpu --cosmologies ${COSMO} --log_file "${LOGFILE}" --parameters_override "${OVERRIDE}"

# Launch with : sbatch --array=0-84 ...