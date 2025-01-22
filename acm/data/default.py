# This file contains the default values for the ACM package
# Dictionaries containung specific values are defined here for clarity

# List of cosmologies in AbacusSummit
cosmo_list = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))

# Statistics coordinates, defining the array shape of the summary statistics
summary_coords_dict = {
    'cosmo_idx': cosmo_list,# List of cosmologies index in AbacusSummit
    'hod_number': 100,      # Number of HODs sampled by cosmology
    'param_number': 20,     # Number of parameters in lhc_x used to generate the simulations
    'phase_number': 1786,   # Number of phases inthe small box simulations
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

# Statistics labels
labels_stats = {
    'dsc_conf': 'Density-split',
    'dsc_pk': 'Density-split 'r'$P_\ell$',
    'dsc_conf_cross': 'Density-split (CCF)',
    'tpcf': 'Galaxy 2PCF',
    'tpcf+dsc_conf': 'DSC + Galaxy 2PCF',
    'number_density+tpcf': 'nbar + Galaxy 2PCF',
    'number_density+pk': 'nbar + P(k)',
    'pk': 'P(k)',
}

# ???
fourier_stats = ['pk', 'dsc_pk']
conf_stats = ['tpcf', 'dsc_conf']



#%% reference dictionary for the summary statistics (not used)
tmp = {
    'tpcf': {},
    'pk': {},
    'dsc_conf': {},
    'dsc_pk': {},
    'number_density': {},
    'cgf_r10': {},
    'pdf': {},
    'voxel_voids': {},
    'wp' :{},
    'knn': {},
    'wst': {},
    'minkowski': {},
    'mst': {},
}