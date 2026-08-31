"""Tests for covariance matrix utility functions."""
import logging

import numpy as np
import pytest

from acm.utils.covariance import (
    ConditionStatus,
    check_condition_number,
    check_covariance_matrix,
    check_positive_definite,
    check_symmetric,
    correlation_from_covariance,
    get_covariance_correction,
    gk_mad_covariance,
    mad_1d,
    orthogonal_gk_mad_covariance,
)

# ruff: noqa: ANN001, ANN201, D101, D102, S101

#%% Fixtures
@pytest.fixture
def valid_cov():
    """Create a well-conditioned, symmetric, positive-definite covariance matrix."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 10))
    return np.cov(data, rowvar=False)

@pytest.fixture
def gaussian_residuals():
    """Well-behaved Gaussian residuals: (200 samples, 10 bins)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 10))


@pytest.fixture
def known_cov_residuals():
    """
    Residuals drawn from a known diagonal covariance diag(1, 4, 9, ...).

    Using a large n so sample estimates are close to truth.
    """
    rng = np.random.default_rng(0)
    scales = np.arange(1, 6, dtype=float)        # std = [1, 2, 3, 4, 5]
    return rng.standard_normal((50_000, 5)) * scales # Scale to match the known variances


#%% Tests
class TestGetCovarianceCorrection:

    def test_hartlap_value(self):
        """Test the Hartlap correction against its known closed-form expression."""
        # Known closed-form: (n_s - 1) / (n_s - n_d - 2)
        assert get_covariance_correction(100, 10, method="hartlap") == pytest.approx(99 / 88)

    def test_percival_fisher_value(self):
        """Test the Percival-Fisher correction against its known closed-form expression."""
        # Known closed-form: (n_s - 1) / (n_s - n_d + n_theta - 1)
        result = get_covariance_correction(100, 10, n_theta=3, method="percival-fisher")
        assert result == pytest.approx(99 / 92)

    def test_percival_value(self):
        """Test the Percival correction against its known closed-form expression for a specific case."""
        # Verify against manual calculation
        n_s, n_d, n_theta = 100, 10, 3
        B = (n_s - n_d - 2) / ((n_s - n_d - 1) * (n_s - n_d - 4))
        expected = (n_s - 1) * (1 + B * (n_d - n_theta)) / (n_s - n_d + n_theta - 1)
        result = get_covariance_correction(n_s, n_d, n_theta=n_theta, method="percival")
        assert result == pytest.approx(expected)

    def test_correction_is_positive(self):
        """All correction factors should be positive to avoid unphysical negative variances after correction."""
        for method, kwargs in [
            ("hartlap",         {}),
            ("percival-fisher", {"n_theta": 3}),
            ("percival",        {"n_theta": 3}),
        ]:
            assert get_covariance_correction(100, 10, method=method, **kwargs) > 0

    def test_unknown_method_raises(self):
        """Test that providing an unknown method name raises a ValueError with an appropriate message."""
        with pytest.raises(ValueError, match="Unknown method"):
            get_covariance_correction(100, 10, method="unknown")

    def test_percival_missing_n_theta_raises(self):
        """Test that providing the Percival method without n_theta raises a ValueError."""
        with pytest.raises(ValueError, match="requires n_theta"):
            get_covariance_correction(100, 10, method="percival")

    def test_percival_fisher_missing_n_theta_raises(self):
        """Test that providing the Percival-Fisher method without n_theta raises a ValueError."""
        with pytest.raises(ValueError, match="requires n_theta"):
            get_covariance_correction(100, 10, method="percival-fisher")

    def test_hartlap_ignores_n_theta(self):
        """Test that providing n_theta for the Hartlap method does not affect the result, since Hartlap does not depend on n_theta."""
        result = get_covariance_correction(100, 10, n_theta=5, method="hartlap")
        assert result == pytest.approx(get_covariance_correction(100, 10, method="hartlap"))


class TestMad1d:
    """Tests for the mad_1d function, which computes the median absolute deviation (MAD) as a robust estimator of variability."""

    def test_gaussian_consistency(self):
        """Test that for large samples from a standard normal distribution, the MAD estimator recovers a value close to 1 (the true standard deviation)."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100_000)
        assert mad_1d(x) == pytest.approx(1.0, rel=0.01)

    def test_scale_equivariance(self):
        """Test that scaling the data by a constant factor scales the MAD by the same factor, confirming that MAD is a scale-equivariant estimator of variability."""
        # MAD(c * x) == |c| * MAD(x)
        rng = np.random.default_rng(1)
        x = rng.standard_normal(1_000)
        assert mad_1d(3.0 * x) == pytest.approx(3.0 * mad_1d(x), rel=1e-10)

    def test_axis_columnwise(self):
        """Test that when computing MAD along axis=0 (column-wise), the result has the correct shape and each column of standard normal data yields a MAD close to 1."""
        # Along axis=0, result shape should match number of columns
        rng = np.random.default_rng(2)
        X = rng.standard_normal((500, 5))
        result = mad_1d(X, axis=0)
        assert result.shape == (5,)
        # Each column of N(0,1) data should be close to 1
        np.testing.assert_allclose(result, 1.0, atol=0.05)

    def test_keepdims(self):
        """Test that when keepdims=True, the output shape retains the reduced dimension as size 1, allowing for broadcasting in subsequent operations."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((100, 4))
        result = mad_1d(X, axis=0, keepdims=True)
        assert result.shape == (1, 4)

    def test_constant_array_is_zero(self):
        assert mad_1d(np.ones(50)) == pytest.approx(0.0)

    def test_outlier_robustness(self):
        """Test that the MAD is robust to outliers."""
        # MAD should be unaffected by a single extreme outlier
        rng = np.random.default_rng(4)
        x = rng.standard_normal(999)
        x_outlier = np.append(x, 1e6)
        assert mad_1d(x) == pytest.approx(mad_1d(x_outlier), rel=0.01)


class TestGkMadCovariance:

    def test_output_shape(self, gaussian_residuals):
        """Test that the output covariance matrix has the correct shape (n_bins, n_bins) given input residuals of shape (n_samples, n_bins)."""
        C = gk_mad_covariance(gaussian_residuals)
        n = gaussian_residuals.shape[1]
        assert C.shape == (n, n)

    def test_symmetric(self, gaussian_residuals):
        """Test that the covariance matrix returned is symmetric, as required for a valid covariance matrix."""
        C = gk_mad_covariance(gaussian_residuals)
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_diagonal_recovers_variance(self, known_cov_residuals):
        """Test that the diagonal entries of the covariance matrix returned approximate the true variances of the input residuals."""
        C = gk_mad_covariance(known_cov_residuals)
        expected_vars = np.arange(1, 6, dtype=float) ** 2
        np.testing.assert_allclose(np.diag(C), expected_vars, rtol=0.05)

    def test_uncorrelated_off_diagonal_near_zero(self, known_cov_residuals):
        """
        Test that GK covariance of independent columns yield near-zero off-diagonal entries.

        Checks that no pairwise covariance estimate exceeds 1.0 against true variances.
        """
        # Independent columns → off-diagonal entries should be near zero
        C = gk_mad_covariance(known_cov_residuals)
        off_diag = C[~np.eye(C.shape[0], dtype=bool)]
        assert np.abs(off_diag).max() < 1.0  # loose: relative to diag ~ [1..25]

    def test_single_bin(self):
        """Test that gk_mad_covariance can handle the case of a single bin (n_bins=1) without error."""
        rng = np.random.default_rng(5)
        x = rng.standard_normal((200, 1))
        C = gk_mad_covariance(x)
        assert C.shape == (1, 1)
        assert C[0, 0] > 0


class TestOrthogonalGkMadCovariance:

    def test_output_shape(self, gaussian_residuals):
        """Test that the output covariance matrix has the correct shape (n_bins, n_bins) given input residuals of shape (n_samples, n_bins)."""
        C = orthogonal_gk_mad_covariance(gaussian_residuals)
        n = gaussian_residuals.shape[1]
        assert C.shape == (n, n)

    def test_symmetric(self, gaussian_residuals):
        """Test that the covariance matrix returned is symmetric, as required for a valid covariance matrix."""
        C = orthogonal_gk_mad_covariance(gaussian_residuals)
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_positive_definite(self, gaussian_residuals):
        """
        Test that the covariance matrix returned is positive-definite, meaning all eigenvalues are positive.

        Which is a requirement for a valid covariance matrix and ensures it can be inverted for likelihood analysis.
        """
        C = orthogonal_gk_mad_covariance(gaussian_residuals)
        # All eigenvalues should be positive
        eigvals = np.linalg.eigvalsh(C)
        assert eigvals.min() > 0

    def test_diagonal_recovers_variance(self, known_cov_residuals):
        """Test that the diagonal entries of the covariance matrix returned approximate the true variances of the input residuals."""
        C = orthogonal_gk_mad_covariance(known_cov_residuals)
        expected_vars = np.arange(1, 6, dtype=float) ** 2
        np.testing.assert_allclose(np.diag(C), expected_vars, rtol=0.05)

    def test_better_conditioned_than_plain_gk(self, gaussian_residuals):
        """Test that the covariance matrix returned has a lower or equal condition number compared to the one returned by gk_mad_covariance."""
        # OGK should have a lower or equal condition number than plain GK
        C_gk  = gk_mad_covariance(gaussian_residuals)
        C_ogk = orthogonal_gk_mad_covariance(gaussian_residuals)
        assert np.linalg.cond(C_ogk) <= np.linalg.cond(C_gk) * 1.1  # 10 % margin

    def test_single_bin(self):
        """Test that orthogonal_gk_mad_covariance can handle the case of a single bin (n_bins=1) without error."""
        rng = np.random.default_rng(6)
        x = rng.standard_normal((200, 1))
        C = orthogonal_gk_mad_covariance(x)
        assert C.shape == (1, 1)
        assert C[0, 0] > 0


class TestCheckSymmetric:

    def test_symmetric(self):
        A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        assert check_symmetric(A) is True

    def test_asymmetric(self):
        A = np.array([[1, 2], [3, 4]])
        assert check_symmetric(A) is False

    def test_non_square(self):
        A = np.ones((2, 3))
        assert check_symmetric(A) is False

    def test_within_tolerance(self):
        """Small numerical asymmetry within tolerance should still be considered symmetric."""
        A = np.array([[1.0, 2.0], [2.0 + 1e-10, 4.0]])
        assert check_symmetric(A, rtol=1e-5, atol=1e-8) is True

    def test_outside_tolerance(self):
        """Small numerical asymmetry outside tolerance should be considered asymmetric."""
        A = np.array([[1.0, 2.0], [2.0 + 1e-4, 4.0]])
        assert check_symmetric(A, rtol=1e-5, atol=1e-8) is False


class TestCheckPositiveDefinite:

    def test_positive_definite(self, valid_cov):
        assert check_positive_definite(valid_cov) is True

    def test_negative_definite(self):
        A = np.array([[-2, 1], [1, -2]])
        assert check_positive_definite(A) is False

    def test_singular(self):
        """Singular matrix (zero eigenvalue) is not positive-definite."""
        A = np.ones((3, 3))
        assert check_positive_definite(A) is False

    def test_psd_not_pd(self):
        """Test that a positive semi-definite matrix (with zero eigenvalues) is not considered positive-definite."""
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert check_positive_definite(A) is False


class TestCheckConditionNumber:

    def test_well_conditioned(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        assert check_condition_number(A) == ConditionStatus.WELL_CONDITIONED

    def test_singular(self):
        """Singular matrix has infinite condition number."""
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        assert check_condition_number(A) == ConditionStatus.SINGULAR

    def test_ill_conditioned(self):
        """Matrix with very large condition number is considered ill-conditioned."""
        # [[1,1],[2,4]] yields ~14 significant digits; push threshold above that
        A = np.array([[1.0, 1.0], [2.0, 4.0]])
        assert check_condition_number(A, precision_threshold=15) == ConditionStatus.ILL_CONDITIONED

    def test_returns_condition_status_type(self, valid_cov):
        """Check that the return type is ConditionStatus enum."""
        result = check_condition_number(valid_cov)
        assert isinstance(result, ConditionStatus)


class TestCheckCovarianceMatrix:

    def test_valid_matrix_passes(self, valid_cov, caplog):
        """A valid covariance matrix should pass all checks without logging warnings."""
        with caplog.at_level(logging.WARNING):
            result = check_covariance_matrix(valid_cov)
        assert result is True
        assert caplog.records == []

    def test_non_2d_returns_false(self, caplog):
        """Input that is not 2-dimensional should return False and log a warning."""
        with caplog.at_level(logging.WARNING):
            result = check_covariance_matrix(np.array([1, 2, 3]))
        assert result is False
        assert "not 2-dimensional" in caplog.text

    def test_non_square_returns_false(self, caplog):
        """Input that is not square should return False and log a warning."""
        with caplog.at_level(logging.WARNING):
            result = check_covariance_matrix(np.ones((2, 3)))
        assert result is False
        assert "not square" in caplog.text

    def test_asymmetric_returns_false(self, caplog):
        """A non-symmetric matrix should return False and log a warning about symmetry."""
        A = np.array([[1.0, 9.0], [0.0, 1.0]])
        with caplog.at_level(logging.WARNING):
            result = check_covariance_matrix(A)
        assert result is False
        assert "not symmetric" in caplog.text

    def test_not_positive_definite_returns_false(self, caplog):
        """A matrix that is not positive-definite should return False and log a warning."""
        A = np.array([[-1.0, 0.0], [0.0, 1.0]])
        with caplog.at_level(logging.WARNING):
            result = check_covariance_matrix(A)
        assert result is False
        assert "not positive-definite" in caplog.text

    def test_singular_short_circuits(self, caplog):
        """A singular matrix should return False and log a warning about singularity, without also logging about positive-definiteness (since the singularity is the root cause)."""
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        with caplog.at_level(logging.WARNING):
            result = check_covariance_matrix(A)
        assert result is False
        assert "singular" in caplog.text
        assert "not positive-definite" not in caplog.text

    def test_ill_conditioned_returns_false(self, caplog):
        """An ill-conditioned matrix should return False and log a warning about being ill-conditioned."""
        A = np.array([[1.0, 1.0], [2.0, 4.0]])
        with caplog.at_level(logging.WARNING):
            result = check_covariance_matrix(A, precision_threshold=15)
        assert result is False
        assert "ill-conditioned" in caplog.text

    def test_custom_name_in_log(self, caplog):
        """The name parameter should be included in the warning messages for clarity."""
        A = np.array([1, 2, 3])
        with caplog.at_level(logging.WARNING):
            check_covariance_matrix(A, name="my_emulator_cov")
        assert "my_emulator_cov" in caplog.text

    def test_silent_at_debug_level(self, valid_cov, caplog):
        """When log level is set to DEBUG, no warnings should be logged for a valid covariance matrix."""
        with caplog.at_level(logging.WARNING):
            result = check_covariance_matrix(valid_cov, log_level=logging.DEBUG)
        assert result is True
        assert caplog.records == []


class TestCorrelationFromCovariance:

    def test_diagonal_is_ones(self, valid_cov):
        """The diagonal of the correlation matrix should be all ones, since it represents the correlation of each variable with itself."""
        corr = correlation_from_covariance(valid_cov)
        np.testing.assert_allclose(np.diag(corr), 1.0, rtol=1e-10)

    def test_symmetric(self, valid_cov):
        """The correlation matrix should be symmetric, since correlation is a symmetric relationship."""
        corr = correlation_from_covariance(valid_cov)
        np.testing.assert_allclose(corr, corr.T, rtol=1e-10)

    def test_values_in_bounds(self, valid_cov):
        """All off-diagonal values in the correlation matrix should be between -1 and 1, since they represent correlation coefficients."""
        corr = correlation_from_covariance(valid_cov)
        assert corr.min() >= -1 - 1e-10
        assert corr.max() <= 1 + 1e-10

    def test_known_two_by_two(self):
        """Test a known 2x2 covariance matrix where the correlation can be calculated by hand."""
        cov = np.array([[4.0, 2.0], [2.0, 9.0]])
        corr = correlation_from_covariance(cov)
        expected_off = 2.0 / (2.0 * 3.0)  # cov / (std_i * std_j)
        np.testing.assert_allclose(corr[0, 1], expected_off, rtol=1e-10)

    def test_zero_covariance_gives_zero_correlation(self):
        """Test that if the covariance between two variables is zero, the correlation is also zero (assuming non-zero variances)."""
        cov = np.array([[4.0, 0.0], [0.0, 9.0]])
        corr = correlation_from_covariance(cov)
        np.testing.assert_allclose(corr, np.eye(2), rtol=1e-10)
