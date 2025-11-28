#!/bin/bash -l

#SBATCH --account desi
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --qos regular
#SBATCH --constraint cpu

#SBATCH --time 48:00:00

#SBATCH --job-name ds_xigg_train
#SBATCH --output /global/homes/s/sbouchar/Output_jobs/bgs_training/%A.%x_%a.out
#SBATCH --error /global/homes/s/sbouchar/Output_jobs/bgs_training/%A.%x_%a.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test

# Load the old pyrecon module for densitysplit
module swap pyrecon/mpi pyrecon/main

python /global/homes/s/sbouchar/acm/scripts/bgs/training/optimize_model.py \
    --compressed_dir /pscratch/sd/s/sbouchar/acm/bgs/input_data \
    --study_dir /pscratch/sd/s/sbouchar/acm/bgs/trained_models/study \
    --save_dir /pscratch/sd/s/sbouchar/acm/bgs/trained_models \
    --n_trials 100 \
    --transform arcsinh \
    --same_n_hidden \
    --log_level info \
    --statistics ds_xigg
