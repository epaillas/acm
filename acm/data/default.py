
# List of cosmologies in AbacusSummit
cosmo_list = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))

summary_coords_stat = {
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
}

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