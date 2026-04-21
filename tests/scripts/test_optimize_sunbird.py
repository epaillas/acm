"""Tests for the EMC Sunbird optimization script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
import warnings

import optuna
import pytest


def load_script_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "emc"
        / "training"
        / "optimize_sunbird.py"
    )
    spec = importlib.util.spec_from_file_location("optimize_sunbird", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def optimize_sunbird_module(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, dict[str, object]]:
    fake_train_sunbird = ModuleType("train_sunbird")
    fake_acm = ModuleType("acm")
    captured: dict[str, object] = {
        "calls": [],
        "prune_call_indices": set(),
    }

    class FakeObservable:
        def __init__(self, stat_name: str):
            self.stat_name = stat_name

    def fake_make_observable(*, root_dir, statistic):
        captured["observable_kwargs"] = {
            "root_dir": Path(root_dir),
            "statistic": statistic,
        }
        return FakeObservable(stat_name=statistic)

    def fake_train_fcn(
        *,
        observable,
        learning_rate,
        n_hidden,
        dropout_rate,
        weight_decay,
        model_dir,
        transform_input,
        transform_output,
        val_fraction,
        seed,
        trial=None,
        enable_pruning=False,
    ):
        call_index = len(captured["calls"])
        model_dir = Path(model_dir)
        (model_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (model_dir / "tensorboard" / "version_0").mkdir(parents=True, exist_ok=True)
        (model_dir / "tensorboard" / "version_0" / "events.out.tfevents").write_text(
            f"events-{call_index}"
        )
        captured["calls"].append(
            {
                "observable": observable,
                "learning_rate": learning_rate,
                "n_hidden": n_hidden,
                "dropout_rate": dropout_rate,
                "weight_decay": weight_decay,
                "model_dir": model_dir,
                "transform_input": transform_input,
                "transform_output": transform_output,
                "val_fraction": val_fraction,
                "seed": seed,
                "trial": trial,
                "enable_pruning": enable_pruning,
            }
        )
        if call_index in captured["prune_call_indices"]:
            raise optuna.TrialPruned(f"trial-{call_index}-pruned")

        checkpoint_path = model_dir / f"{observable.stat_name}.ckpt"
        checkpoint_path.write_text(f"checkpoint-{call_index}")
        (model_dir / "checkpoints" / "last.ckpt").write_text(f"last-{call_index}")
        losses = [0.4, 0.1, 0.3, 0.2]
        return losses[call_index]

    fake_train_sunbird.DEFAULT_ROOT_DIR = Path("/default/root")
    fake_train_sunbird.TrainFCN = fake_train_fcn
    fake_train_sunbird.get_default_study_dir = (
        lambda root_dir, statistic: Path(root_dir) / "emc/models/v1.3/optuna" / statistic
    )
    fake_train_sunbird.make_observable = fake_make_observable
    fake_acm.setup_logging = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "train_sunbird", fake_train_sunbird)
    monkeypatch.setitem(sys.modules, "acm", fake_acm)

    return load_script_module(), captured


def test_resolve_transform_output_uses_deprecated_alias(
    optimize_sunbird_module: tuple[ModuleType, dict[str, object]],
) -> None:
    module, _ = optimize_sunbird_module
    args = module.parse_args(["--transform", "log"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transform_output = module._resolve_transform_output(args)

    assert transform_output == "log"
    assert len(caught) == 1
    assert "--transform" in str(caught[0].message)


def test_parse_args_defaults_to_tpe_sampler(
    optimize_sunbird_module: tuple[ModuleType, dict[str, object]],
) -> None:
    module, _ = optimize_sunbird_module

    args = module.parse_args([])

    assert args.sampler == "tpe"


def test_parse_args_accepts_cmaes_sampler(
    optimize_sunbird_module: tuple[ModuleType, dict[str, object]],
) -> None:
    module, _ = optimize_sunbird_module

    args = module.parse_args(["--sampler", "cmaes"])

    assert args.sampler == "cmaes"


def test_parse_args_defaults_to_median_pruner(
    optimize_sunbird_module: tuple[ModuleType, dict[str, object]],
) -> None:
    module, _ = optimize_sunbird_module

    args = module.parse_args([])

    assert args.pruner == "median"


def test_parse_args_accepts_none_pruner(
    optimize_sunbird_module: tuple[ModuleType, dict[str, object]],
) -> None:
    module, _ = optimize_sunbird_module

    args = module.parse_args(["--pruner", "none"])

    assert args.pruner == "none"


@pytest.mark.parametrize(
    ("sampler_name", "sampler_type"),
    [("tpe", optuna.samplers.TPESampler), ("cmaes", optuna.samplers.CmaEsSampler)],
)
def test_load_or_create_study_uses_requested_sampler_for_new_study(
    tmp_path: Path,
    optimize_sunbird_module: tuple[ModuleType, dict[str, object]],
    sampler_name: str,
    sampler_type: type[optuna.samplers.BaseSampler],
) -> None:
    module, _ = optimize_sunbird_module

    study = module._load_or_create_study(
        study_fn=tmp_path / "study.pkl",
        statistic="projected_tpcf",
        seed=7,
        sampler_name=sampler_name,
        pruner_name="none",
    )

    assert isinstance(study.sampler, sampler_type)
    assert len(study.trials) == 1
    assert study.trials[0].state == optuna.trial.TrialState.WAITING
    assert study.trials[0].system_attrs["fixed_params"] == module.DEFAULT_INITIAL_TRIAL


@pytest.mark.parametrize(
    ("pruner_name", "pruner_type"),
    [("median", optuna.pruners.MedianPruner), ("none", optuna.pruners.NopPruner)],
)
def test_load_or_create_study_uses_requested_pruner_for_new_study(
    tmp_path: Path,
    optimize_sunbird_module: tuple[ModuleType, dict[str, object]],
    pruner_name: str,
    pruner_type: type[optuna.pruners.BasePruner],
) -> None:
    module, _ = optimize_sunbird_module

    study = module._load_or_create_study(
        study_fn=tmp_path / "study.pkl",
        statistic="projected_tpcf",
        seed=7,
        sampler_name="tpe",
        pruner_name=pruner_name,
    )

    assert isinstance(study.pruner, pruner_type)


def test_load_or_create_study_keeps_existing_sampler_and_pruner_on_resume(
    tmp_path: Path, optimize_sunbird_module: tuple[ModuleType, dict[str, object]]
) -> None:
    module, _ = optimize_sunbird_module
    study_fn = tmp_path / "study.pkl"
    original_study = module._load_or_create_study(
        study_fn=study_fn,
        statistic="projected_tpcf",
        seed=7,
        sampler_name="cmaes",
        pruner_name="none",
    )
    module.joblib.dump(original_study, study_fn)

    resumed_study = module._load_or_create_study(
        study_fn=study_fn,
        statistic="projected_tpcf",
        seed=7,
        sampler_name="tpe",
        pruner_name="median",
    )

    assert isinstance(resumed_study.sampler, optuna.samplers.CmaEsSampler)
    assert isinstance(resumed_study.pruner, optuna.pruners.NopPruner)


def test_optimize_sunbird_retains_trial_artifacts_and_publishes_best_model(
    tmp_path: Path, optimize_sunbird_module: tuple[ModuleType, dict[str, object]]
) -> None:
    module, captured = optimize_sunbird_module
    study_dir = tmp_path / "study"
    publish_dir = tmp_path / "published"
    publish_dir.mkdir()
    (publish_dir / "unrelated.txt").write_text("keep-me")

    module.main(
        [
            "--root_dir",
            str(tmp_path / "root"),
            "--study_dir",
            str(study_dir),
            "--model_dir",
            str(publish_dir),
            "--n_trials",
            "2",
            "--seed",
            "7",
            "--val_fraction",
            "0.25",
            "--transform_input",
            "log",
            "--transform_output",
            "arcsinh",
            "--statistic",
            "projected_tpcf",
        ]
    )

    assert captured["observable_kwargs"] == {
        "root_dir": tmp_path / "root",
        "statistic": "projected_tpcf",
    }
    assert len(captured["calls"]) == 2
    assert captured["calls"][0]["transform_input"] == "log"
    assert captured["calls"][0]["transform_output"] == "arcsinh"
    assert captured["calls"][0]["val_fraction"] == 0.25
    assert captured["calls"][0]["seed"] == 7
    assert captured["calls"][0]["enable_pruning"] is True
    assert captured["calls"][0]["trial"] is not None

    first_trial_dir = study_dir / "trials" / "trial_0000"
    second_trial_dir = study_dir / "trials" / "trial_0001"
    for trial_dir in (first_trial_dir, second_trial_dir):
        assert (trial_dir / "projected_tpcf.ckpt").exists()
        assert (trial_dir / "checkpoints" / "last.ckpt").exists()
        assert (trial_dir / "tensorboard" / "version_0" / "events.out.tfevents").exists()

    assert (study_dir / "study.pkl").exists()
    summary = json.loads((study_dir / "best_trial.json").read_text())
    assert summary["number"] == 1
    assert summary["value"] == pytest.approx(0.1)
    assert Path(summary["model_dir"]) == second_trial_dir
    assert Path(summary["checkpoint_path"]) == second_trial_dir / "projected_tpcf.ckpt"
    assert Path(summary["published_model_dir"]) == publish_dir

    assert (publish_dir / "projected_tpcf.ckpt").read_text() == "checkpoint-1"
    assert (publish_dir / "checkpoints" / "last.ckpt").read_text() == "last-1"
    assert (
        publish_dir / "tensorboard" / "version_0" / "events.out.tfevents"
    ).read_text() == "events-1"
    assert (publish_dir / "unrelated.txt").read_text() == "keep-me"


def test_optimize_sunbird_retains_artifacts_for_pruned_trials(
    tmp_path: Path, optimize_sunbird_module: tuple[ModuleType, dict[str, object]]
) -> None:
    module, captured = optimize_sunbird_module
    captured["prune_call_indices"] = {0}
    study_dir = tmp_path / "study"

    module.main(
        [
            "--study_dir",
            str(study_dir),
            "--n_trials",
            "2",
            "--statistic",
            "projected_tpcf",
        ]
    )

    study = module.joblib.load(study_dir / "study.pkl")
    summary = json.loads((study_dir / "best_trial.json").read_text())
    first_trial_dir = study_dir / "trials" / "trial_0000"
    second_trial_dir = study_dir / "trials" / "trial_0001"

    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    assert study.trials[1].state == optuna.trial.TrialState.COMPLETE
    assert Path(study.trials[0].user_attrs["model_dir"]) == first_trial_dir
    assert "checkpoint_path" not in study.trials[0].user_attrs
    assert first_trial_dir.exists()
    assert (first_trial_dir / "tensorboard" / "version_0" / "events.out.tfevents").exists()
    assert not (first_trial_dir / "projected_tpcf.ckpt").exists()
    assert (second_trial_dir / "projected_tpcf.ckpt").exists()
    assert summary["number"] == 1
    assert summary["value"] == pytest.approx(0.1)


def test_optimize_sunbird_resumes_existing_study(
    tmp_path: Path, optimize_sunbird_module: tuple[ModuleType, dict[str, object]]
) -> None:
    module, captured = optimize_sunbird_module
    study_dir = tmp_path / "study"

    module.main(
        [
            "--study_dir",
            str(study_dir),
            "--n_trials",
            "1",
            "--statistic",
            "projected_tpcf",
        ]
    )
    module.main(
        [
            "--study_dir",
            str(study_dir),
            "--n_trials",
            "2",
            "--statistic",
            "projected_tpcf",
        ]
    )

    assert len(captured["calls"]) == 2
    assert (study_dir / "trials" / "trial_0000").exists()
    assert (study_dir / "trials" / "trial_0001").exists()

    summary = json.loads((study_dir / "best_trial.json").read_text())
    assert summary["number"] == 1
    assert summary["value"] == pytest.approx(0.1)


def test_optimize_sunbird_fails_if_trial_dirs_exist_without_study_file(
    tmp_path: Path, optimize_sunbird_module: tuple[ModuleType, dict[str, object]]
) -> None:
    module, _ = optimize_sunbird_module
    study_dir = tmp_path / "study"
    (study_dir / "trials" / "trial_0000").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Refusing to start a new study"):
        module.main(
            [
                "--study_dir",
                str(study_dir),
                "--n_trials",
                "1",
                "--statistic",
                "projected_tpcf",
            ]
        )
