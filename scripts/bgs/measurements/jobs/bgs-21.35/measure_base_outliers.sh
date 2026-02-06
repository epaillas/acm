#!/bin/bash -l

#SBATCH --account desi_g
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --qos regular
#SBATCH --constraint gpu&hbm80g

#SBATCH --time 10:00:00

#SBATCH --job-name bgs_base_outliers
#SBATCH --output /pscratch/sd/s/sbouchar/Output_jobs/bgs-21.35_measurements/%A.%x_%a.out
#SBATCH --error /pscratch/sd/s/sbouchar/Output_jobs/bgs-21.35_measurements/%A.%x_%a.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Load the old pyrecon module for densitysplit
module swap pyrecon/mpi pyrecon/main

cd /global/homes/s/sbouchar/acm-repo/scripts/bgs/measurements

LOGFILE="/pscratch/sd/s/sbouchar/acm/bgs-21.35/measurements/logs/outliers/log_base_cxxx_ph000_seed0.log"
OUTLIERS='jobs/bgs-21.35/outliers/all_outliers_simtype-base_ells-02_sigma-20.0.npy'

python measure_box.py --config jobs/bgs-21.35/config.yaml --gpu --log_file "${LOGFILE}" --overwrite --parameters_override "${OUTLIERS}"

# Launch with : sbatch ...