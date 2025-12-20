#!/bin/bash -l

#SBATCH --account desi
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --qos regular
#SBATCH --constraint cpu

#SBATCH --time 48:00:00

#SBATCH --job-name tpcf_train
#SBATCH --output /pscratch/sd/s/sbouchar/Output_jobs/bgs_training/%A.%x_%a.out
#SBATCH --error /pscratch/sd/s/sbouchar/Output_jobs/bgs_training/%A.%x_%a.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

python /global/homes/s/sbouchar/acm/scripts/bgs/training/optimize_model.py \
    --compressed_dir /pscratch/sd/s/sbouchar/acm/bgs/input_data \
    --study_dir /pscratch/sd/s/sbouchar/acm/bgs/trained_models/sigma_clipped/study \
    --save_dir /pscratch/sd/s/sbouchar/acm/bgs/trained_models/sigma_clipped \
    --n_trials 100 \
    --transform arcsinh \
    --same_n_hidden \
    --log_level info \
    --statistics tpcf
