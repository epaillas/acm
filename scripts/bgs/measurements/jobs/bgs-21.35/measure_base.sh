#!/bin/bash -l

#SBATCH --account desi_g
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --qos regular
#SBATCH --constraint gpu&hbm80g

#SBATCH --time 12:00:00

#SBATCH --job-name 10-bgs_base
#SBATCH --output /pscratch/sd/s/sbouchar/Output_jobs/bgs-21.35_measurements/%A.%x_%a.out
#SBATCH --error /pscratch/sd/s/sbouchar/Output_jobs/bgs-21.35_measurements/%A.%x_%a.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Load the old pyrecon module for densitysplit
module swap pyrecon/mpi pyrecon/main

export XLA_PYTHON_CLIENT_ALLOCATOR=platform # JAX backend for GPU memory allocation

# Get the cosmology index from the SLURM_ARRAY_TASK_ID
COSMO_LIST=(0 {1..4} 13 {100..126} {130..181}) # List of cosmologies to be used
ID=$((SLURM_ARRAY_TASK_ID)) # ID of the cosmology to be used, starting from 0
COSMO=${COSMO_LIST[ID]} # Cosmology to be used

RUN=1
OVERRIDE=$(printf "/pscratch/sd/s/sbouchar/acm/bgs/parameters/override/idx/c%03d.csv" ${COSMO}) # Ensure the same HODs are used for bgs-21.35 as for bgs-20
LOGFILE=$(printf "/pscratch/sd/s/sbouchar/acm/bgs/mr-21.35/logs/v2.0/measurements/abacus/base/run%d/log_c%03d_ph000_seed0.log" ${RUN} ${COSMO})

cd /global/homes/s/sbouchar/acm/scripts/bgs/measurements
python measure_box.py --config jobs/bgs-21.35/config.yaml --cosmologies ${COSMO} --log_file "${LOGFILE}" --parameters_override "${OVERRIDE}"

# Launch with : sbatch --array=0-84 ...