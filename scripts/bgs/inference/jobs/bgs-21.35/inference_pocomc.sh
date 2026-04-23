#!/bin/bash -l

#SBATCH --account desi
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --qos regular
#SBATCH --constraint cpu

#SBATCH --time 10:00:00

#SBATCH --job-name val21-inference
#SBATCH --output /pscratch/sd/s/sbouchar/Output_jobs/bgs-21.35_inference/validation/%A.%x_%a.out
#SBATCH --error /pscratch/sd/s/sbouchar/Output_jobs/bgs-21.35_inference/validation/%A.%x_%a.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

cd /global/homes/s/sbouchar/acm/scripts/bgs/inference/

CONFIG=jobs/bgs-21.35/config_validation.yaml
HOD_MODEL=base-AB-CB-VB-s

SLICE_MAP="{'tpcf': {'s': [30, 150]}, 'ds_xiqg': {'s': [1, 30]}, 'ds_xiqq': {'s': [1, 30]}}"
# OBS_KWARGS="{'ds_xiqg': {'stat_name': 'ds_xiqg_r3'}, 'ds_xiqq': {'stat_name': 'ds_xiqq_r3'}}" # For rebinned DS (1-150 Mpc/h)

COSMO_MODEL_LIST=(
    # base-w0wa-fixed-omega_b
    base-fixed-omega_b
    base-w0wa
    base
    base-nrun-Nur-w0wa
    # base-nrun-Nur-w0wa-fixed-omega_b
)

for COSMO_MODEL in "${COSMO_MODEL_LIST[@]}"; do
    echo "Running inference for COSMO_MODEL: $COSMO_MODEL"

    python inference_pocomc.py --config $CONFIG --statistics tpcf --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL
    python inference_pocomc.py --config $CONFIG --statistics ds_xiqg ds_xiqq --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL
    python inference_pocomc.py --config $CONFIG --statistics tpcf ds_xiqg ds_xiqq --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL
    
    # python inference_pocomc.py --config $CONFIG --statistics tpcf --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL --slice_map "$SLICE_MAP"
    # python inference_pocomc.py --config $CONFIG --statistics ds_xiqg ds_xiqq --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL --slice_map "$SLICE_MAP" --obs_kwargs "$OBS_KWARGS"
    # python inference_pocomc.py --config $CONFIG --statistics tpcf ds_xiqg ds_xiqq --cosmo_model $COSMO_MODEL --hod_model $HOD_MODEL --slice_map "$SLICE_MAP" --obs_kwargs "$OBS_KWARGS"
done

