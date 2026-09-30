"""Tests for acm.observables.model."""
from unittest.mock import patch

import numpy as np
import pytest

from acm.observables.model import ObservableModel

# ruff: noqa: ANN001, ANN201, D102, S101

class TestInit:
    """Tests for ObservableModel.__init__."""

    def test_default_transform_none(self, dummy_sunbird_model):
        model = ObservableModel(model=dummy_sunbird_model)
        assert model.transform is None
        assert model._model is dummy_sunbird_model


class TestLoad:
    """Tests for ObservableModel.load."""

    def test_forwards_to_checkpoint_loader(self, dummy_sunbird_model):
        """Checks load() calls load_model_from_checkpoint and forwards kwargs to __init__."""
        def fake_transform(pred):  # noqa: ANN202
            return pred

        with patch(
            "acm.observables.model.load_model_from_checkpoint",
            return_value=dummy_sunbird_model,
        ) as mock_load:
            model = ObservableModel.load("dummy.ckpt", model_cls=None, transform=fake_transform)

        mock_load.assert_called_once_with("dummy.ckpt", model_cls=None)
        assert model._model is dummy_sunbird_model
        assert model.transform is fake_transform


class TestGetPrediction:
    """Tests for ObservableModel.get_prediction."""

    def test_get_prediction_no_transform(self, make_dummy_model):
        model = make_dummy_model()
        x = np.zeros((3, 2))
        pred = model.get_prediction(x)
        assert pred.shape == (3, 4)
        np.testing.assert_allclose(pred[0], [1.0, 2.0, 3.0, 4.0])

    def test_get_prediction_applies_transform(self, make_dummy_model):
        model = make_dummy_model(transform=lambda p: p * 2)
        x = np.zeros((3, 2))
        pred = model.get_prediction(x)
        np.testing.assert_allclose(pred[0], [2.0, 4.0, 6.0, 8.0])


class TestGetError:
    """Tests for ObservableModel.get_error."""

    def test_median(self, make_dummy_model):
        model = make_dummy_model()
        x = np.zeros((2, 2))
        truth = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])
        error = model.get_error(x, truth, method="median")
        pred = model.get_prediction(x)
        expected = np.median(np.abs(truth - pred), axis=0)
        np.testing.assert_allclose(error, expected)

    def test_uses_make_covariance(self, make_dummy_model):
        """Checks the non-median fallback returns factor * diag(make_covariance(diff))."""
        model = make_dummy_model()
        x = np.zeros((5, 2))
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(5, 4)) + np.array([1.0, 2.0, 3.0, 4.0])
        factor = 2.0
        error = model.get_error(x, truth, method="stdev", factor=factor, diag=True)
        pred = model.get_prediction(x)
        diff = truth - pred
        expected = np.diag(factor * model.make_covariance(diff, "stdev", diag=True))
        np.testing.assert_allclose(error, expected)


class TestMakeCovariance:
    """Tests for ObservableModel.make_covariance."""

    @pytest.mark.parametrize("diag", [True, False])
    def test_make_covariance_mad(self, diag):
        rng = np.random.default_rng(1)
        y = rng.normal(size=(20, 3))
        cov = ObservableModel.make_covariance(y, method="mad", diag=diag)
        assert cov.shape == (3, 3)
        if diag:
            assert np.allclose(cov, np.diag(np.diagonal(cov)))

    def test_make_covariance_mean_diag(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=(20, 3))
        cov = ObservableModel.make_covariance(y, method="mean", diag=True)
        assert cov.shape == (3, 3)
        assert np.allclose(cov, np.diag(np.diagonal(cov)))

    def test_make_covariance_mean_full_raises(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=(20, 3))
        with pytest.raises(NotImplementedError):
            ObservableModel.make_covariance(y, method="mean", diag=False)

    @pytest.mark.parametrize("diag", [True, False])
    def test_make_covariance_stdev(self, diag):
        rng = np.random.default_rng(4)
        y = rng.normal(size=(20, 3))
        cov = ObservableModel.make_covariance(y, method="stdev", diag=diag)
        expected = np.diag(np.std(y, axis=0) ** 2) if diag else np.cov(y, rowvar=False)
        np.testing.assert_allclose(cov, expected)

    def test_make_covariance_unknown_method_raises(self):
        y = np.zeros((5, 2))
        with pytest.raises(ValueError, match="Unknown method"):
            ObservableModel.make_covariance(y, method="bogus", diag=False)
