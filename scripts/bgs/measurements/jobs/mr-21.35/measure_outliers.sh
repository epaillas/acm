#!/bin/bash -l

#SBATCH --account desi_g
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --qos debug
#SBATCH --constraint gpu&hbm80g

#SBATCH --time 00:30:00

#SBATCH --job-name outliers-21
#SBATCH --output /pscratch/sd/s/sbouchar/Output_jobs/acm/bgs/mr-21.35/measurements/%A.%x_%a.out
#SBATCH --error  /pscratch/sd/s/sbouchar/Output_jobs/acm/bgs/mr-21.35/measurements/%A.%x_%a.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Load the old pyrecon module for densitysplit
module swap pyrecon/mpi pyrecon/main

export XLA_PYTHON_CLIENT_ALLOCATOR=platform # JAX backend for GPU memory allocation

SIMTYPE=base
TYPE=outlier # outlier/corrupted

RUN=2 # New run!
LOGFILE=$(printf "/pscratch/sd/s/sbouchar/acm/bgs/mr-21.35/logs/v2.0/measurements/abacus/${SIMTYPE}/run%d/log_outliers_${TYPE}.log" ${RUN})
OVERRIDE="/pscratch/sd/s/sbouchar/acm/bgs/parameters/override/${SIMTYPE}/${TYPE}_idx.csv" # Needs to be build by hand

cd /global/homes/s/sbouchar/acm/scripts/bgs/measurements
python measure_box.py --config jobs/mr-21.35/config.yaml --sim_type "${SIMTYPE}" --log_file "${LOGFILE}" --parameters_override "${OVERRIDE}" --overwrite
