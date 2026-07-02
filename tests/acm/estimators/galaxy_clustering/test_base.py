import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from acm.estimators.galaxy_clustering.base import BaseEstimator

MODULE = "acm.estimators.galaxy_clustering.base"

# ruff: noqa: ANN001, ANN201, ANN205, ARG004, D101, D102, D103, INP001, S101

class DummyEstimator(BaseEstimator):
    def compute(self):
        return MagicMock()

    @staticmethod
    def load(filename):
        return MagicMock()

    @staticmethod
    def plot(obj, fig=None, ax=None):
        return MagicMock(), MagicMock()

@pytest.fixture
def estimator(make_estimator):
    return make_estimator(DummyEstimator)

@pytest.fixture
def mock_obj():
    obj = MagicMock()
    obj.write.side_effect = lambda path, **kwargs: path.touch()  # noqa: ARG005
    return obj

class TestInit:

    def test_load_backend_called_with_correct_args(self, dummy_backend, data_positions, randoms_positions):
        with patch(f"{MODULE}.load_backend", return_value=dummy_backend) as mock_load:
            DummyEstimator(
                backend="dummy",
                data_positions=data_positions,
                randoms_positions=randoms_positions,
            )
        mock_load.assert_called_once_with(
            "dummy",
            data_positions=data_positions,
            randoms_positions=randoms_positions,
            data_weights=None,
            randoms_weights=None,
        )

    def test_logs_initialization(self, dummy_backend, data_positions, randoms_positions, caplog):
        with patch(f"{MODULE}.load_backend", return_value=dummy_backend), caplog.at_level(logging.INFO):
                DummyEstimator(
                    backend="dummy",
                    data_positions=data_positions,
                    randoms_positions=randoms_positions,
                )
        assert "DummyEstimator" in caplog.text

    def test_stores_positions_and_weights(self, estimator, data_positions, randoms_positions):
        assert np.array_equal(estimator.data_positions, data_positions)
        assert np.array_equal(estimator.randoms_positions, randoms_positions)
        assert estimator.data_weights is None
        assert estimator.randoms_weights is None

class TestSave:

    def test_raise_on_invalid_extension(self, estimator, tmp_path):
        with pytest.raises(ValueError, match="extensions"):
            estimator.save(MagicMock(), tmp_path / "result.txt")

    def test_skip_if_file_exists_and_no_overwrite(self, estimator, tmp_path, caplog):
        existing = tmp_path / "result.h5"
        existing.touch()
        with caplog.at_level(logging.INFO):
            estimator.save(MagicMock(), existing, overwrite=False)
        assert "Skipping" in caplog.text

    def test_overwrite_for_overwrite_true(self, estimator, tmp_path):
        """Test that the _atomic_write method is called when overwrite=True."""
        existing = tmp_path / "result.h5"
        existing.touch()
        mock_obj = MagicMock()
        with patch.object(estimator, "_atomic_write") as mock_write:
            estimator.save(mock_obj, existing, overwrite=True)
        mock_write.assert_called_once()

    def test_create_parent_directories(self, estimator, tmp_path):
        nested = tmp_path / "a" / "b" / "result.h5"
        mock_obj = MagicMock()
        with patch.object(estimator, "_atomic_write"):
            estimator.save(mock_obj, nested)
        assert nested.parent.exists()

    @pytest.mark.parametrize("ext", ["h5", "hdf5"])
    def test_accepted_extensions(self, estimator, tmp_path, ext):
        """Both .h5 and .hdf5 extensions should be accepted."""
        mock_obj = MagicMock()
        with patch.object(estimator, "_atomic_write"):
            estimator.save(mock_obj, tmp_path / f"result.{ext}")

class TestAtomicWrite:

    def test_call_obj_write(self, tmp_path, mock_obj):
        """Test that the object's write method is called."""
        filename = tmp_path / "result.h5"
        BaseEstimator._atomic_write(mock_obj, filename)
        mock_obj.write.assert_called_once()

    def test_write_to_tmp_file(self, tmp_path, mock_obj):
        """Test that the object is written to a temporary file."""
        filename = tmp_path / "result.h5"
        tmp_fn = filename.with_name(filename.stem + ".tmp" + filename.suffix)
        BaseEstimator._atomic_write(mock_obj, filename)
        mock_obj.write.assert_called_once_with(tmp_fn)

    def test_tmp_file_cleaned(self, tmp_path, mock_obj):
        """Temporary file should not remain after write."""
        filename = tmp_path / "result.h5"
        tmp_fn = filename.with_name(filename.stem + ".tmp" + filename.suffix)
        BaseEstimator._atomic_write(mock_obj, filename)
        assert not tmp_fn.exists()

    def test_final_file_exists(self, tmp_path, mock_obj):
        """Only the final file should exist after atomic write."""
        filename = tmp_path / "result.h5"
        BaseEstimator._atomic_write(mock_obj, filename)
        assert filename.exists()
