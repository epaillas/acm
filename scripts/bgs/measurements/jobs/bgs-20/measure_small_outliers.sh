#!/bin/bash -l

#SBATCH --account desi_g
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --qos debug
#SBATCH --constraint gpu&hbm80g

#SBATCH --time 00:10:00

#SBATCH --job-name bgs_small_outliers
#SBATCH --output /pscratch/sd/s/sbouchar/Output_jobs/bgs-20_measurements/%A.%x_%a.out
#SBATCH --error /pscratch/sd/s/sbouchar/Output_jobs/bgs-20_measurements/%A.%x_%a.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Load the old pyrecon module for densitysplit
module swap pyrecon/mpi pyrecon/main

cd /global/homes/s/sbouchar/acm/scripts/bgs/measurements

LOGFILE="/pscratch/sd/s/sbouchar/acm/bgs-20/measurements/logs/small/outliers/log_small_c000_phxxx_seed0.log"
OUTLIERS='jobs/bgs-20/outliers/all_outliers_simtype-small_ells-02_sigma-6.0.npy'
# OUTLIERS='jobs/bgs-20/outliers/all_corrupted_fourier_small.npy'

python measure_box.py --config jobs/bgs-20/config.yaml --gpu --sim_type small --log_file "${LOGFILE}" --overwrite --parameters_override "${OUTLIERS}" #--measurements power_spectrum density_split_power

# Launch with : sbatch ... 