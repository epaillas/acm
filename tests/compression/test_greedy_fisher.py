import pytest
import numpy as np
from acm.compression.greedy_fisher import safe_inverse, get_fisher_information
import acm.observables.emc as emc

STAT_NAMES = [
    'tpcf', 'bk', 'pk', 'dsc_pk', 'wp', 'vg_voids', 
    'dt_voids', 'cumulant', 'minkowski', 'wst', 'mst', 'pdf'
]

@pytest.fixture
def statistics():
    """Fixture that creates and returns the stat_map object for use in multiple tests"""
    select_mocks = {'cosmo_idx': [0], 'hod_idx': [30,]}
    return {
        'tpcf': emc.GalaxyCorrelationFunctionMultipoles(
            select_mocks=select_mocks,
        ),
        'bk': emc.GalaxyBispectrumMultipoles(
            select_mocks=select_mocks,
        ),
        'pk': emc.GalaxyPowerSpectrumMultipoles(
            select_mocks=select_mocks,
        ),
        'dsc_pk': emc.DensitySplitPowerSpectrumMultipoles(
            select_mocks=select_mocks,
        ),
        'wp': emc.GalaxyProjectedCorrelationFunction(
            select_mocks=select_mocks,
        ),
        'vg_voids': emc.VoxelVoidGalaxyCorrelationFunctionMultipoles(
            select_mocks=select_mocks,
        ),
        'dt_voids': emc.DTVoidGalaxyCorrelationFunctionMultipoles(
            select_mocks=select_mocks,
        ),
        'cumulant': emc.CumulantGeneratingFunction(
            select_mocks=select_mocks,
        ),
        'minkowski': emc.MinkowskiFunctionals(
            select_mocks=select_mocks,
        ),
        'wst': emc.WaveletScatteringTransform(
            select_mocks=select_mocks,
        ),
        'mst': emc.MinimumSpanningTree(
            select_mocks=select_mocks,
        ),
        'pdf': emc.GalaxyOverdensityPDF(
            select_mocks=select_mocks,
        ),
    }



@pytest.mark.parametrize("stat_name", STAT_NAMES)
def test_well_behaved_covariance(statistics, stat_name, atol=1e-4, rtol=1e-4):
    """Test that covariance matrix and its inverse produce identity matrix."""
    stat = statistics[stat_name]
    small_box_y = stat.small_box_y
    covariance_matrix = np.cov(small_box_y.T)
    correction = stat.get_covariance_correction(
        n_s=len(small_box_y),
        n_d=len(covariance_matrix),
        n_theta=20,
        method='percival',
    )
    print(f"{stat_name}, correction = {correction}")
    precision_matrix = safe_inverse(covariance_matrix)
    identity_check = covariance_matrix @ precision_matrix 
    expected_identity = np.eye(covariance_matrix.shape[0])
    
    assert identity_check == pytest.approx(expected_identity, rel=rtol, abs=atol), \
        f"Covariance * inverse != identity for {stat_name}"

@pytest.mark.parametrize("stat_name", STAT_NAMES)
def test_well_behaved_fisher(statistics, stat_name):
    """Test that Fisher information is positive."""
    stat = statistics[stat_name]
    fisher_information = get_fisher_information(stat)
    assert fisher_information > 0., f"Fisher < 0 for {stat_name}"