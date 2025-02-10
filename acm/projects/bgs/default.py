from acm.data.default import cosmo_list # List of cosmologies in AbacusSummit

# Reference paths for the BGS project
bgs_paths = {
    # dir of the input data and covariance
    'lhc_dir': '/pscratch/sd/s/sbouchar/acm/bgs/input_data/',
    'covariance_dir': '/pscratch/sd/s/sbouchar/acm/bgs/input_data/',
    # dir of the errors
    'error_dir': '/pscratch/sd/s/sbouchar/acm/bgs/emulator_error/',
    'emulator_covariance_dir': '/pscratch/sd/s/sbouchar/acm/bgs/emulator_error/',
    # dir of the trained models
    'study_dir': '/pscratch/sd/s/sbouchar/acm/bgs/trained_models/optuna/',
    'model_dir': '/pscratch/sd/s/sbouchar/acm/bgs/trained_models/',
    'checkpoint_name': 'last.ckpt',
    # dir of the inference
    'chain_dir': '/pscratch/sd/s/sbouchar/acm/bgs/chains/',
}

# Statistics coordinates, defining the array shape of the summary statistics
bgs_summary_coords_dict = {
    'cosmo_idx': cosmo_list,# List of cosmologies index in AbacusSummit
    'hod_number': 100,      # Number of HODs sampled by cosmology
    'param_number': 17,     # Number of parameters in lhc_x used to generate the simulations
    'phase_number': 1629,   # Number of phases in the small box simulations
    'statistics': {         # Statistic organization in lhc_y 
        'tpcf': {
            'multipoles': [0, 2],
        },
        'dsc_conf': {
            'statistics': ['quantile_data_correlation', 'quantile_correlation'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
        },
    },
}
