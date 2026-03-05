"""
Tests for covariance matrix sanity checks.
"""
import numpy as np
import pytest
import warnings
from acm.utils.covariance import (
    check_symmetric,
    check_positive_definite,
    check_covariance_matrix,
    check_condition_number,
    correlation_from_covariance
)


class TestSymmetryCheck:
    """Test the check_symmetric function."""
    
    def test_symmetric_matrix(self):
        """Test with a symmetric matrix."""
        matrix = np.array([[1, 2, 3],
                           [2, 4, 5],
                           [3, 5, 6]])
        assert check_symmetric(matrix) is True
    
    def test_asymmetric_matrix(self):
        """Test with an asymmetric matrix."""
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
        assert check_symmetric(matrix) is False
    
    def test_non_square_matrix(self):
        """Test with a non-square matrix."""
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6]])
        assert check_symmetric(matrix) is False
    
    def test_nearly_symmetric_matrix(self):
        """Test with a nearly symmetric matrix within tolerance."""
        matrix = np.array([[1.0, 2.0, 3.0],
                           [2.0, 4.0, 5.0],
                           [3.0, 5.0, 6.0]])
        # Add small perturbation
        matrix[0, 1] += 1e-10
        assert check_symmetric(matrix, rtol=1e-5, atol=1e-8) is True


class TestPositiveDefiniteCheck:
    """Test the check_positive_definite function."""
    
    def test_positive_definite_matrix(self):
        """Test with a positive-definite matrix."""
        # Create a positive-definite matrix
        A = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])
        assert check_positive_definite(A) is True
    correlation_from_covariance
    def test_identity_matrix(self):
        """Test with identity matrix (always positive-definite)."""
        I = np.eye(5)
        assert check_positive_definite(I) is True
    
    def test_negative_definite_matrix(self):
        """Test with a negative-definite matrix."""
        # Create a negative-definite matrix
        A = np.array([[-2, 1, 0],
                      [1, -2, 1],
                      [0, 1, -2]])
        assert check_positive_definite(A) is False
    
    def test_singular_matrix(self):
        """Test with a singular (non-invertible) matrix."""
        A = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])
        assert check_positive_definite(A) is False
    
    def test_realistic_covariance(self):
        """Test with a realistic covariance matrix from random data."""
        # Generate random data
        np.random.seed(42)
        data = np.random.randn(100, 10)
        cov = np.cov(data, rowvar=False)
        assert check_positive_definite(cov) is True


class TestConditionNumberCheck:
    """Test the check_condition_number function."""
    
    def test_well_conditioned_matrix(self):
        """Test with a well-conditioned matrix."""
        A = np.array([[4, 1],
                      [1, 3]])
        status = check_condition_number(A, precision_threshold=10)
        assert status == 1  # Well-conditioned
    
    def test_ill_conditioned_matrix(self):
        """Test with an ill-conditioned matrix."""
        A = np.array([[1, 1],
                      [2, 4]]) # Should return 14 significant digits
        status = check_condition_number(A, precision_threshold=15) # Set threshold higher than 14 to trigger ill-conditioning
        assert status == 2  # Ill-conditioned
    
    def test_singular_matrix(self):
        """Test with a singular matrix."""
        A = np.array([[1, 2],
                      [2, 4]])
        status = check_condition_number(A, precision_threshold=10)
        assert status == 0  # Singular

class TestCovarianceMatrixCheck:
    """Test the check_covariance_matrix function."""
    
    def test_valid_covariance_matrix(self):
        """Test with a valid covariance matrix."""
        # Generate a valid covariance matrix
        np.random.seed(42)
        data = np.random.randn(100, 10)
        cov = np.cov(data, rowvar=False)
        
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            result = check_covariance_matrix(cov)
        
        assert result is True
    
    def test_non_square_matrix_warning(self):
        """Test that non-square matrix raises a warning."""
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6]])
        
        with pytest.warns(UserWarning, match="not square"):
            result = check_covariance_matrix(matrix)
        
        assert result is False
    
    def test_asymmetric_matrix_warning(self):
        """Test that asymmetric matrix raises a warning."""
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
        
        with pytest.warns(UserWarning, match="not symmetric"):
            result = check_covariance_matrix(matrix)
        
        assert result is False
    
    def test_non_positive_definite_warning(self):
        """Test that non-positive-definite matrix raises a warning."""
        # Create a symmetric but not positive-definite matrix
        matrix = np.array([[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        
        with pytest.warns(UserWarning, match="not positive-definite"):
            result = check_covariance_matrix(matrix)
        
        assert result is False
    
    def test_custom_name_in_warning(self):
        """Test that custom name appears in warning message."""
        matrix = np.array([[1, 2],
                           [4, 5]])
        
        with pytest.warns(UserWarning, match="test_matrix"):
            check_covariance_matrix(matrix, name="test_matrix")
    
    def test_non_2d_matrix_warning(self):
        """Test that non-2D matrix raises a warning."""
        matrix = np.array([1, 2, 3])
        
        with pytest.warns(UserWarning, match="not 2-dimensional"):
            result = check_covariance_matrix(matrix)
        
        assert result is False
    
    def test_ill_conditioned_warning(self):
        """Test that ill-conditioned matrix raises a warning."""
        matrix = np.array([[1, 1],
                           [2, 4]]) # Should return 14 significant digits
        
        with pytest.warns(UserWarning, match="ill-conditioned"):
            result = check_covariance_matrix(matrix, precision_threshold=15)

        assert result is False

    def test_multiple_failures(self):
        """Test that multiple issues are all reported."""
        # Non-square, asymmetric matrix
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6]])
        
        with pytest.warns(UserWarning) as record:
            result = check_covariance_matrix(matrix)
        
        # Should have at least one warning about non-square
        assert any("not square" in str(w.message) for w in record)
        assert result is False


class TestCorrelationFromCovariance:
    """Test the correlation_from_covariance function."""
    
    def test_identity_covariance(self):
        """Test with identity covariance matrix."""
        cov = np.eye(3)
        corr = correlation_from_covariance(cov)
        expected = np.eye(3)
        np.testing.assert_allclose(corr, expected, rtol=1e-10)
    
    def test_diagonal_covariance(self):
        """Test with diagonal covariance matrix."""
        cov = np.array([[4, 0, 0],
                        [0, 9, 0],
                        [0, 0, 16]])
        corr = correlation_from_covariance(cov)
        expected = np.eye(3)
        np.testing.assert_allclose(corr, expected, rtol=1e-10)
    
    def test_two_by_two_covariance(self):
        """Test with 2x2 covariance matrix."""
        cov = np.array([[4, 2],
                        [2, 3]])
        corr = correlation_from_covariance(cov)
        correl_value = 2 / (2 * np.sqrt(3))
        expected = np.array([[1, correl_value],
                                [correl_value, 1]])
        np.testing.assert_allclose(corr, expected, rtol=1e-10)
    
    def test_symmetric_output(self):
        """Test that output correlation matrix is symmetric."""
        np.random.seed(42)
        data = np.random.randn(50, 5)
        cov = np.cov(data, rowvar=False)
        corr = correlation_from_covariance(cov)
        np.testing.assert_allclose(corr, corr.T, rtol=1e-10)
    
    def test_unit_diagonal(self):
        """Test that diagonal elements are 1."""
        np.random.seed(42)
        data = np.random.randn(50, 5)
        cov = np.cov(data, rowvar=False)
        corr = correlation_from_covariance(cov)
        np.testing.assert_allclose(np.diag(corr), np.ones(5), rtol=1e-10)
    
    def test_bounds_correlation(self):
        """Test that all correlation values are in [-1, 1]."""
        np.random.seed(42)
        data = np.random.randn(50, 5)
        cov = np.cov(data, rowvar=False)
        corr = correlation_from_covariance(cov)
        assert np.all(corr >= -1 - 1e-10) and np.all(corr <= 1 + 1e-10)
    
    def test_zero_covariance_handling(self):
        """Test handling of zero covariance elements."""
        cov = np.array([[4, 0, 2],
                        [0, 9, 0],
                        [2, 0, 16]])
        corr = correlation_from_covariance(cov)
        # Zero covariance elements should result in zero correlation
        assert corr[0, 1] == 0
        assert corr[1, 0] == 0
        assert corr[1, 2] == 0
        assert corr[2, 1] == 0
    
    def test_perfect_correlation(self):
        """Test with perfectly correlated variables."""
        cov = np.array([[4, 4],
                        [4, 4]])
        corr = correlation_from_covariance(cov)
        expected = np.array([[1, 1],
                                [1, 1]])
        np.testing.assert_allclose(corr, expected, rtol=1e-10)
class TestCorrelationFromCovariance:
    """Test the correlation_from_covariance function."""
    
    def test_identity_covariance(self):
        """Test with identity covariance matrix."""
        cov = np.eye(3)
        corr = correlation_from_covariance(cov)
        expected = np.eye(3)
    