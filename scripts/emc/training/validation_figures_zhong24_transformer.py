import argparse
import inspect
from pathlib import Path

from acm import setup_logging
from acm.utils.modules import get_class_from_module
from train_zhong24_transformer_sunbird import (
    DEFAULT_ROOT_DIR,
    get_default_model_root_dir,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate validation figures for EMC Zhong24 transformer emulators.",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="acm.observables.emc",
        help="Module to load the observable classes from.",
    )
    parser.add_argument(
        "-s",
        "--statistics",
        nargs="+",
        default=["projected_tpcf"],
        help="List of statistics to validate.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=DEFAULT_ROOT_DIR.as_posix(),
        help="Base directory for default EMC input and output paths.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="fig/zhong24_transformer",
        help="Directory where validation figures will be written.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=(
            "Optional Zhong24 transformer model root or statistic-specific directory. "
            "When omitted, uses the default Zhong24 transformer best-model namespace."
        ),
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def get_observable_class(module_path, observable_name):
    return get_class_from_module(module_path, observable_name)


def get_observable_stat_name(module_path, observable_name):
    observable_cls = get_observable_class(module_path, observable_name)
    return inspect.signature(observable_cls.__init__).parameters["stat_name"].default


def get_observable_paths(root_dir):
    root_dir = Path(root_dir)
    return {
        "data_dir": root_dir / "emc/measurements/v1.3/abacus/compressed",
        "measurements_dir": root_dir / "emc/measurements/v1.3/abacus",
        "param_dir": None,
    }


def resolve_checkpoint_path(statistic, model_dir):
    model_dir = Path(model_dir)
    candidates = [
        model_dir / f"{statistic}.ckpt",
        model_dir / statistic / f"{statistic}.ckpt",
    ]
    for checkpoint_path in candidates:
        if checkpoint_path.exists():
            return checkpoint_path

    attempted_paths = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find a Zhong24 transformer checkpoint for '{statistic}'. Tried:\n"
        f"{attempted_paths}"
    )


def make_observable(
    observable_name,
    *,
    module_path,
    root_dir,
    model_dir,
    **kwargs,
):
    statistic = get_observable_stat_name(module_path, observable_name)
    checkpoint_path = resolve_checkpoint_path(statistic, model_dir)
    paths = get_observable_paths(root_dir)
    paths["model_dir"] = checkpoint_path.parent
    observable_cls = get_observable_class(module_path, observable_name)
    return observable_cls(
        paths=paths,
        **kwargs,
    )


def plot_model(
    observable_name,
    *,
    module_path,
    root_dir,
    save_dir,
    model_dir,
    cosmo_idx=0,
    hod_idx=0,
):
    observable = make_observable(
        observable_name,
        module_path=module_path,
        root_dir=root_dir,
        model_dir=model_dir,
        select_filters={"cosmo_idx": cosmo_idx, "hod_idx": hod_idx},
        numpy_output=True,
        squeeze_output=True,
    )
    save_fn = Path(save_dir) / f"{observable.stat_name}_model.png"
    observable.plot_observable(model_params=observable.x, save_fn=save_fn)


def plot_emulator_residuals(
    observable_name,
    *,
    module_path,
    root_dir,
    save_dir,
    model_dir,
):
    observable = make_observable(
        observable_name,
        module_path=module_path,
        root_dir=root_dir,
        model_dir=model_dir,
        select_filters={},
    )
    save_fn = Path(save_dir) / f"{observable.stat_name}_emulator_residuals.png"
    observable.plot_emulator_residuals(save_fn)


def main(argv=None):
    args = parse_args(argv)
    setup_logging()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.model_dir is None:
        model_dir = get_default_model_root_dir(root_dir=args.root_dir)
    else:
        model_dir = Path(args.model_dir)

    for statistic in args.statistics:
        plot_model(
            statistic,
            module_path=args.module,
            root_dir=args.root_dir,
            save_dir=save_dir,
            model_dir=model_dir,
        )
        plot_emulator_residuals(
            statistic,
            module_path=args.module,
            root_dir=args.root_dir,
            save_dir=save_dir,
            model_dir=model_dir,
        )


if __name__ == "__main__":
    main()
