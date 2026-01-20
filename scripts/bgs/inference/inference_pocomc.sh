#!/bin/bash -l

#SBATCH --account desi
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --qos regular
#SBATCH --constraint cpu

#SBATCH --time 10:00:00

#SBATCH --job-name s-basew0wa
#SBATCH --output /pscratch/sd/s/sbouchar/Output_jobs/bgs_inference/secondgen/%A.%x_%a.out
#SBATCH --error /pscratch/sd/s/sbouchar/Output_jobs/bgs_inference/secondgen/%A.%x_%a.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh 2025_05

cd /global/homes/s/sbouchar/acm-repo/scripts/bgs/inference/

COSMO_MODEL=base-w0wa
CONFIG=config_secondgen.yaml

HOD_MODEL=None
python inference_pocomc.py --config $CONFIG --statistics tpcf --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL  
python inference_pocomc.py --config $CONFIG --statistics ds_xiqg ds_xiqq --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL
python inference_pocomc.py --config $CONFIG --statistics tpcf ds_xiqg ds_xiqq --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL

HOD_MODEL=base
python inference_pocomc.py --config $CONFIG --statistics tpcf --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL  
python inference_pocomc.py --config $CONFIG --statistics ds_xiqg ds_xiqq --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL
python inference_pocomc.py --config $CONFIG --statistics tpcf ds_xiqg ds_xiqq --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL

HOD_MODEL=base-AB-CB-VB-s
python inference_pocomc.py --config $CONFIG --statistics tpcf --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL  
python inference_pocomc.py --config $CONFIG --statistics ds_xiqg ds_xiqq --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL
python inference_pocomc.py --config $CONFIG --statistics tpcf ds_xiqg ds_xiqq --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL