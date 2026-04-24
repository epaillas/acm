"""Tests for the EMC Sunbird training script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


def load_script_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "emc"
        / "training"
        / "train_sunbird.py"
    )
    spec = importlib.util.spec_from_file_location("train_sunbird", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def train_sunbird_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    fake_sunbird = ModuleType("sunbird")
    fake_emulators = ModuleType("sunbird.emulators")
    fake_data = ModuleType("sunbird.data")
    fake_transforms = ModuleType("sunbird.data.transforms_array")
    fake_acm = ModuleType("acm")
    fake_acm_observables = ModuleType("acm.observables")

    class PlaceholderTransform:
        def transform(self, value):
            return value

    fake_emulators.FCN = object
    fake_emulators.train = SimpleNamespace(FCNTrainer=object)
    fake_data.ArrayDataModule = object
    fake_transforms.LogTransform = PlaceholderTransform
    fake_transforms.ArcsinhTransform = PlaceholderTransform
    fake_sunbird.emulators = fake_emulators
    fake_sunbird.data = fake_data
    fake_acm.setup_logging = lambda *args, **kwargs: None
    fake_acm_observables.Observable = object

    monkeypatch.setitem(sys.modules, "sunbird", fake_sunbird)
    monkeypatch.setitem(sys.modules, "sunbird.emulators", fake_emulators)
    monkeypatch.setitem(sys.modules, "sunbird.data", fake_data)
    monkeypatch.setitem(sys.modules, "sunbird.data.transforms_array", fake_transforms)
    monkeypatch.setitem(sys.modules, "acm", fake_acm)
    monkeypatch.setitem(sys.modules, "acm.observables", fake_acm_observables)

    return load_script_module()


class FakeObservable:
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        covariance_y: np.ndarray,
        stat_name: str = "projected_tpcf",
    ):
        self.stat_name = stat_name
        self._dataset = SimpleNamespace(
            data_vars={name: object() for name in ("x_train", "y_train", "x_test", "y_test")}
        )
        self.x_train = x_train
        self.y_train = y_train
        self.covariance_y = covariance_y

    @property
    def x(self):
        raise AssertionError("TrainFCN should not read the full x array")

    @property
    def y(self):
        raise AssertionError("TrainFCN should not read the full y array")


def test_train_fcn_uses_dataset_split_and_training_only_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, train_sunbird_module: ModuleType
) -> None:
    full_x = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [100.0, 100.0],
            [200.0, 200.0],
        ]
    )
    full_y = np.array(
        [
            [10.0, 10.0],
            [20.0, 20.0],
            [300.0, 300.0],
            [400.0, 400.0],
        ]
    )
    covariance_y = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [4.0, 40.0],
            [8.0, 80.0],
        ]
    )
    observable = FakeObservable(full_x, full_y, covariance_y)
    captured: dict[str, object] = {}

    class FakeArrayDataModule:
        def __init__(self, x, y, val_fraction, batch_size, num_workers):
            x = np.asarray(x)
            y = np.asarray(y)
            captured["data_module_init"] = {
                "x": x,
                "y": y,
                "val_fraction": val_fraction,
                "batch_size": batch_size,
                "num_workers": num_workers,
            }
            self.n_input = x.shape[-1]
            self.n_output = y.shape[-1]
            self.ds_train = SimpleNamespace(
                tensors=(
                    torch.tensor(x[:2], dtype=torch.float32),
                    torch.tensor(y[:2], dtype=torch.float32),
                )
            )
            self._train_loader = object()
            self._val_loader = object()
            captured["train_loader"] = self._train_loader
            captured["val_loader"] = self._val_loader

        def setup(self):
            captured["setup_called"] = True

        def train_dataloader(self):
            return self._train_loader

        def val_dataloader(self):
            return self._val_loader

    class FakeFCN:
        def __init__(self, **kwargs):
            captured["model_kwargs"] = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer_init_kwargs"] = kwargs
            checkpoint_dir = kwargs["checkpoint_dir"]
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.best_checkpoint = checkpoint_dir / "epoch=02-step=3-val_loss=0.12300.ckpt"
            self.best_checkpoint.write_text("best-checkpoint")
            (checkpoint_dir / "last.ckpt").write_text("last-checkpoint")
            self.callbacks = [SimpleNamespace(best_model_path=str(self.best_checkpoint))]

        def fit(self, *, model, train_dataloaders, val_dataloaders):
            captured["fit_kwargs"] = {
                "model": model,
                "train_dataloaders": train_dataloaders,
                "val_dataloaders": val_dataloaders,
            }
            return 0.123

    monkeypatch.setattr(train_sunbird_module, "ArrayDataModule", FakeArrayDataModule)
    monkeypatch.setattr(train_sunbird_module, "FCN", FakeFCN)
    monkeypatch.setattr(
        train_sunbird_module,
        "seed_everything",
        lambda seed, workers: captured.setdefault("seed_everything", (seed, workers)),
    )
    monkeypatch.setattr(
        train_sunbird_module,
        "train",
        SimpleNamespace(FCNTrainer=FakeTrainer),
    )

    model_dir = tmp_path / "model"
    val_loss = train_sunbird_module.TrainFCN(
        observable=observable,
        learning_rate=1e-3,
        n_hidden=[32, 32],
        dropout_rate=0.0,
        weight_decay=0.0,
        model_dir=model_dir,
        val_fraction=0.25,
        seed=42,
    )

    assert val_loss == 0.123
    assert captured["setup_called"] is True
    assert captured["data_module_init"]["val_fraction"] == 0.25
    assert "val_idx" not in captured["data_module_init"]
    np.testing.assert_array_equal(captured["data_module_init"]["x"], full_x)
    np.testing.assert_array_equal(captured["data_module_init"]["y"], full_y)
    np.testing.assert_allclose(
        captured["model_kwargs"]["covariance_matrix"],
        np.cov(covariance_y, rowvar=False) / 64.0,
    )
    np.testing.assert_allclose(captured["model_kwargs"]["mean_input"], np.array([0.5, 0.5]))
    np.testing.assert_allclose(captured["model_kwargs"]["std_input"], np.array([0.5, 0.5]))
    np.testing.assert_allclose(captured["model_kwargs"]["mean_output"], np.array([15.0, 15.0]))
    np.testing.assert_allclose(captured["model_kwargs"]["std_output"], np.array([5.0, 5.0]))
    assert captured["seed_everything"] == (42, True)
    assert captured["trainer_init_kwargs"]["checkpoint_dir"] == model_dir / "checkpoints"
    assert captured["trainer_init_kwargs"]["deterministic"] is True
    assert captured["trainer_init_kwargs"]["log_dir"] == model_dir
    assert captured["trainer_init_kwargs"]["logger"] == "tensorboard"
    assert captured["trainer_init_kwargs"]["tensorboard_name"] == "tensorboard"
    assert captured["fit_kwargs"]["train_dataloaders"] is captured["train_loader"]
    assert captured["fit_kwargs"]["val_dataloaders"] is captured["val_loader"]
    assert (model_dir / "checkpoints" / "last.ckpt").exists()
    exported_checkpoint = model_dir / f"{observable.stat_name}.ckpt"
    assert exported_checkpoint.exists()
    assert exported_checkpoint.read_text() == "best-checkpoint"


def test_train_fcn_recomputes_covariance_in_transformed_output_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, train_sunbird_module: ModuleType
) -> None:
    x_train = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float)
    y_train = np.array([[1.0, 2.0], [2.0, 4.0], [4.0, 8.0]], dtype=float)
    covariance_y = np.array([[1.0, 3.0], [2.0, 9.0], [4.0, 27.0]], dtype=float)
    observable = FakeObservable(x_train, y_train, covariance_y)
    captured: dict[str, object] = {}

    class ScalingTransform:
        def transform(self, value):
            return np.asarray(value, dtype=float) * 10.0

    class FakeArrayDataModule:
        def __init__(self, x, y, val_fraction, batch_size, num_workers):
            captured["data_module_init"] = {
                "x": np.asarray(x),
                "y": np.asarray(y),
            }
            self.n_input = np.asarray(x).shape[-1]
            self.n_output = np.asarray(y).shape[-1]
            self.ds_train = SimpleNamespace(
                tensors=(
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32),
                )
            )
            self._train_loader = object()
            self._val_loader = object()

        def setup(self):
            return None

        def train_dataloader(self):
            return self._train_loader

        def val_dataloader(self):
            return self._val_loader

    class FakeFCN:
        def __init__(self, **kwargs):
            captured["model_kwargs"] = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            checkpoint_dir = kwargs["checkpoint_dir"]
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint = checkpoint_dir / "epoch=02-step=3-val_loss=0.12300.ckpt"
            best_checkpoint.write_text("best-checkpoint")
            self.callbacks = [SimpleNamespace(best_model_path=str(best_checkpoint))]

        def fit(self, *, model, train_dataloaders, val_dataloaders):
            return 0.123

    monkeypatch.setattr(train_sunbird_module, "ArrayDataModule", FakeArrayDataModule)
    monkeypatch.setattr(train_sunbird_module, "FCN", FakeFCN)
    monkeypatch.setattr(
        train_sunbird_module,
        "_build_transform",
        lambda transform_name: ScalingTransform() if transform_name == "log" else None,
    )
    monkeypatch.setattr(
        train_sunbird_module,
        "train",
        SimpleNamespace(FCNTrainer=FakeTrainer),
    )

    val_loss = train_sunbird_module.TrainFCN(
        observable=observable,
        learning_rate=1e-3,
        n_hidden=[32, 32],
        dropout_rate=0.0,
        weight_decay=0.0,
        model_dir=tmp_path / "model",
        transform_output="log",
        val_fraction=0.25,
        seed=42,
    )

    assert val_loss == 0.123
    np.testing.assert_allclose(captured["data_module_init"]["y"], y_train * 10.0)
    np.testing.assert_allclose(
        captured["model_kwargs"]["covariance_matrix"],
        np.cov(covariance_y * 10.0, rowvar=False) / 64.0,
    )


def test_train_fcn_builds_pruning_callbacks_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, train_sunbird_module: ModuleType
) -> None:
    observable = FakeObservable(
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        np.array([[10.0, 10.0], [20.0, 20.0]], dtype=float),
        np.array([[1.0, 2.0], [2.0, 4.0], [4.0, 8.0]], dtype=float),
    )
    captured: dict[str, object] = {}

    class FakeArrayDataModule:
        def __init__(self, x, y, val_fraction, batch_size, num_workers):
            self.n_input = np.asarray(x).shape[-1]
            self.n_output = np.asarray(y).shape[-1]
            self.ds_train = SimpleNamespace(
                tensors=(
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32),
                )
            )
            self._train_loader = object()
            self._val_loader = object()

        def setup(self):
            return None

        def train_dataloader(self):
            return self._train_loader

        def val_dataloader(self):
            return self._val_loader

    class FakeFCN:
        def __init__(self, **kwargs):
            captured["model_kwargs"] = kwargs

    class FakeLearningRateMonitor:
        def __init__(self, logging_interval):
            self.logging_interval = logging_interval

    class FakeRichProgressBar:
        pass

    class FakePruningCallback:
        def __init__(self, trial, monitor):
            self.trial = trial
            self.monitor = monitor

    class FakeTrainer:
        @staticmethod
        def early_stop_callback(**kwargs):
            return SimpleNamespace(kind="early_stop", kwargs=kwargs)

        @staticmethod
        def checkpoint_callback(**kwargs):
            return SimpleNamespace(kind="checkpoint", kwargs=kwargs, best_model_path=None)

        def __init__(self, **kwargs):
            captured["trainer_init_kwargs"] = kwargs
            checkpoint_dir = kwargs["checkpoint_dir"]
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.best_checkpoint = checkpoint_dir / "epoch=02-step=3-val_loss=0.12300.ckpt"
            self.best_checkpoint.write_text("best-checkpoint")
            for callback in kwargs["callbacks"]:
                if getattr(callback, "kind", None) == "checkpoint":
                    callback.best_model_path = str(self.best_checkpoint)
            self.callbacks = kwargs["callbacks"]

        def fit(self, *, model, train_dataloaders, val_dataloaders):
            captured["fit_kwargs"] = {
                "model": model,
                "train_dataloaders": train_dataloaders,
                "val_dataloaders": val_dataloaders,
            }
            return 0.123

    monkeypatch.setattr(train_sunbird_module, "ArrayDataModule", FakeArrayDataModule)
    monkeypatch.setattr(train_sunbird_module, "FCN", FakeFCN)
    monkeypatch.setattr(
        train_sunbird_module,
        "seed_everything",
        lambda seed, workers: captured.setdefault("seed_everything", (seed, workers)),
    )
    monkeypatch.setattr(
        train_sunbird_module,
        "_get_lightning_callback_classes",
        lambda: (FakeLearningRateMonitor, FakeRichProgressBar),
    )
    monkeypatch.setattr(
        train_sunbird_module,
        "_get_pruning_callback_cls",
        lambda: FakePruningCallback,
    )
    monkeypatch.setattr(
        train_sunbird_module,
        "train",
        SimpleNamespace(FCNTrainer=FakeTrainer),
    )

    fake_trial = object()
    model_dir = tmp_path / "model"
    val_loss = train_sunbird_module.TrainFCN(
        observable=observable,
        learning_rate=1e-3,
        n_hidden=[32, 32],
        dropout_rate=0.0,
        weight_decay=0.0,
        model_dir=model_dir,
        val_fraction=0.25,
        seed=42,
        trial=fake_trial,
        enable_pruning=True,
    )

    assert val_loss == 0.123
    assert captured["seed_everything"] == (42, True)
    callbacks = captured["trainer_init_kwargs"]["callbacks"]
    assert captured["trainer_init_kwargs"]["deterministic"] is True
    assert getattr(callbacks[0], "kind", None) == "early_stop"
    assert getattr(callbacks[1], "kind", None) == "checkpoint"
    assert isinstance(callbacks[2], FakeLearningRateMonitor)
    assert callbacks[2].logging_interval == "step"
    assert isinstance(callbacks[3], FakeRichProgressBar)
    assert isinstance(callbacks[4], FakePruningCallback)
    assert callbacks[4].trial is fake_trial
    assert callbacks[4].monitor == "val_loss"
    assert (model_dir / f"{observable.stat_name}.ckpt").read_text() == "best-checkpoint"


def test_get_pruning_callback_cls_uses_optuna_integration_when_available(
    train_sunbird_module: ModuleType,
) -> None:
    pytest.importorskip("optuna_integration.pytorch_lightning")

    pruning_callback_cls = train_sunbird_module._get_pruning_callback_cls()

    assert pruning_callback_cls.__name__ == "PyTorchLightningPruningCallback"


def test_train_fcn_requires_compressed_train_test_split(
    tmp_path: Path, train_sunbird_module: ModuleType
) -> None:
    observable = SimpleNamespace(
        stat_name="projected_tpcf",
        _dataset=SimpleNamespace(data_vars={"x": object(), "y": object()}),
    )

    with pytest.raises(ValueError, match="compress_files.py --statistic projected_tpcf"):
        train_sunbird_module.TrainFCN(
            observable=observable,
            learning_rate=1e-3,
            n_hidden=[32, 32],
            dropout_rate=0.0,
            weight_decay=0.0,
            model_dir=tmp_path / "model",
            val_fraction=0.25,
            seed=42,
        )
