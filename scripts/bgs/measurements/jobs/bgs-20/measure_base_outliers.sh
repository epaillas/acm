#!/bin/bash -l

#SBATCH --account desi_g
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --qos debug
#SBATCH --constraint gpu&hbm80g

#SBATCH --time 00:30:00

#SBATCH --job-name bgs-20_measure_base_outliers
#SBATCH --output /pscratch/sd/s/sbouchar/Output_jobs/bgs-20_measurements/%A.%x_%a.out
#SBATCH --error /pscratch/sd/s/sbouchar/Output_jobs/bgs-20_measurements/%A.%x_%a.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Load the old pyrecon module for densitysplit
module swap pyrecon/mpi pyrecon/main

cd /global/homes/s/sbouchar/acm/scripts/bgs/measurements

SIMTYPE=base
METHOD=corrupted_h5

LOGFILE="/pscratch/sd/s/sbouchar/acm/bgs-20/measurements/logs/outliers/log_${SIMTYPE}_${METHOD}.log"
OUTLIERS="jobs/bgs-20/outliers/${METHOD}-simtype_${SIMTYPE}-all_measurements.npy"

python measure_box.py --config jobs/bgs-20/config.yaml --gpu --log_file "${LOGFILE}" --overwrite --parameters_override "${OUTLIERS}" #--measurements power_spectrum density_split_power

# Launch with : sbatch ...