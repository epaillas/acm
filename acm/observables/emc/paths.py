from acm.data.default import cosmo_list # List of cosmologies in AbacusSummit

# Reference paths for the EMC project
emc_paths = {
    # dir of the input data and covariance
    'lhc_dir': '/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod',
    'diffsky_dir': '/pscratch/sd/e/epaillas/emc/v1.1/diffsky/data_vectors/',
    'covariance_dir': '/pscratch/sd/e/epaillas/emc/v1.1/abacus/covariance_sets/small_box',
    # dir of the errors
    'emulator_error_dir': '/pscratch/sd/e/epaillas/emc/v1.1/emulator_error',
    'emulator_covariance_dir': '/pscratch/sd/s/sbouchar/acm/emc/emulator_error/',
    # dir of the trained models
    'study_dir': '/pscratch/sd/s/sbouchar/acm/emc/trained_models/optuna',
    'model_dir': '/pscratch/sd/s/sbouchar/acm/emc/trained_models/',
    'checkpoint_name': 'last.ckpt',
    # dir of the inference
    'chain_dir': '',
}