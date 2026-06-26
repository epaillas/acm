import numpy as np

from acm.estimators.galaxy_clustering.backends.filters import (
    GaussianFilter,
    NoFilter,
    TopHatFilter,
)

# ruff: noqa: ANN201, D101, D102, INP001, S101

class TestFilters:
    K = (np.array([0.0, 0.1, 0.5]),) * 3
    V = np.array([1.0, 2.0, 3.0])

    def test_gaussian_zero_k_unchanged(self):
        """At k=0 the Gaussian kernel equals 1, so v is returned unchanged."""
        f = GaussianFilter(r=5.0)
        np.testing.assert_almost_equal(f((np.array([0.0]),) * 3, np.array([1.0])), [1.0])

    def test_gaussian_attenuates_high_k(self):
        """Gaussian filter should attenuate high-k modes more than low-k."""
        f = GaussianFilter(r=5.0)
        assert f((np.array([0.01]),) * 3, np.array([1.0])) > f((np.array([1.0]),) * 3, np.array([1.0]))

    def test_tophat_zero_k_unchanged(self):
        """At k=0 the top-hat kernel equals 1, so v is returned unchanged."""
        f = TopHatFilter(r=5.0)
        np.testing.assert_almost_equal(f((np.array([0.0]),) * 3, np.array([1.0])), [1.0])

    def test_nofilter_returns_v(self):
        np.testing.assert_array_equal(NoFilter(r=0.0)(self.K, self.V), self.V)

    def test_filter_radius_stored(self):
        for cls in [GaussianFilter, TopHatFilter, NoFilter]:
            assert cls(r=7.0).r == 7.0