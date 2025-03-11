# This file contains the default paths for the ACM package
# Any path in the pipeline should be defined here and called from acm.data.paths


# Dark matter catalog paths
BGS_Abacus_DM = {
    'box': {
        'small': {
            'sim_dir': '/global/cfs/cdirs/desi/cosmosim/Abacus/small/',
            'subsample_dir': '/pscratch/sd/s/sbouchar/summit_subsamples/boxes/small/',
        },
        
        'base': {
            'sim_dir': '/global/cfs/cdirs/desi/cosmosim/Abacus/',
            'subsample_dir': '/pscratch/sd/s/sbouchar/summit_subsamples/boxes/base/',
        },
    },
}

LRG_Abacus_DM = {
    'box': {
        'small': {
            'sim_dir': '/global/cfs/cdirs/desi/cosmosim/Abacus/small/',
            'subsample_dir': '/pscratch/sd/e/epaillas/summit_subsamples/boxes/small/',
        },
        
        'base': {
            'sim_dir': '/global/cfs/cdirs/desi/cosmosim/Abacus/',
            'subsample_dir': '/pscratch/sd/e/epaillas/summit_subsamples/boxes/base/',
        },
    },
    'lightcone': {
        'base': {
            'sim_dir': '/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/halo_light_cones/',
            'subsample_dir': '/pscratch/sd/e/epaillas/summit_subsamples/lightcones/',
        },
    },
}


#%% reference dictionary for the paths (not used)
tmp_paths = {
    # dir of the input data and covariance
    'lhc_dir': '',
    'covariance_dir': '',
    # dir of the errors
    'error_dir': '',
    'emulator_covariance_dir': '',
    # dir of the trained models
    'study_dir': '',
    'model_dir': '',
    'checkpoint_name': 'last.ckpt',
    # dir of the inference
    'chain_dir': '',
}