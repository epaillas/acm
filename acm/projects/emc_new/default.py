from acm.data.default import cosmo_list # List of cosmologies in AbacusSummit

# Reference paths for the EMC project
emc_paths = {
    # dir of the input data and covariance
    'lhc_dir': '/pscratch/sd/s/sbouchar/acm/emc/input_data/',
    'covariance_dir': '/pscratch/sd/s/sbouchar/acm/emc/input_data/',
    # dir of the errors
    'error_dir': '/pscratch/sd/s/sbouchar/acm/emc/emulator_error/',
    'emulator_covariance_dir': '/pscratch/sd/s/sbouchar/acm/emc/emulator_error/',
    'save_dir': '/pscratch/sd/s/sbouchar/acm/emc/emulator_error/',
    # dir of the trained models
    'study_dir': '/pscratch/sd/s/sbouchar/acm/emc/trained_models/optuna',
    'model_dir': '/pscratch/sd/s/sbouchar/acm/emc/trained_models/',
    'checkpoint_name': 'last.ckpt',
    # dir of the inference
    'chain_dir': '',
}

# Statistics coordinates, defining the array shape of the summary statistics
emc_summary_coords_dict = {
    'cosmo_idx': cosmo_list,# List of cosmologies index in AbacusSummit
    'hod_number': 100,      # Number of HODs sampled by cosmology
    'param_number': 20,     # Number of parameters in lhc_x used to generate the simulations
    'phase_number': 1786,   # Number of phases in the small box simulations
    'statistics': {         # Statistic organization in lhc_y 
        'tpcf': {
            'multipoles': [0, 2],
        },
        'pk': {
            'multipoles': [0, 2],
        },
        'dsc_conf': {
            'statistics': ['quantile_data_correlation', 'quantile_correlation'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
        },
        'dsc_pk': {
            'statistics': ['quantile_data_power', 'quantile_power'],
            'quantiles': [0, 1, 3, 4],
            'multipoles': [0, 2],
        },
        'bispectrum': {
            'multipoles': [0, 2],
        },
        'number_density': {},
        'cgf_r10': {},
        'pdf': {},
        'voxel_voids': {
            'multipoles': [0, 2],
        },
        'wp' :{},
        'knn': {},
        'wst': {},
        'minkowski': {},
        'mst': {},
    },
}
