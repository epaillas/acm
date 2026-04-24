import argparse
from pathlib import Path

import torch
from lightning import seed_everything
from sunbird.data import ArrayDataModule
from sunbird.emulators import Zhong24Transformer, train

from acm import setup_logging
from train_sunbird import (
    DEFAULT_ROOT_DIR,
    EMC_MODELS_RELATIVE_DIR,
    _build_trainer_callbacks,
    _build_transform,
    _compute_training_covariance_matrix,
    _export_best_checkpoint,
    _require_train_test_split,
    make_observable,
)

torch.set_float32_matmul_precision("high")

EMC_ZHONG24_MODELS_RELATIVE_DIR = EMC_MODELS_RELATIVE_DIR / "zhong24_transformer"
ZHONG24_BATCH_SIZE = 512
ZHONG24_MAX_EPOCHS = 200
ZHONG24_EARLY_STOP_PATIENCE = 25
ZHONG24_EARLY_STOP_MIN_DELTA = 1.0e-4
ZHONG24_SCHEDULER_PATIENCE = 8
ZHONG24_SCHEDULER_FACTOR = 0.5
ZHONG24_SCHEDULER_THRESHOLD = 1.0e-4


def get_default_model_root_dir(root_dir):
    return Path(root_dir) / EMC_ZHONG24_MODELS_RELATIVE_DIR / "best"


def get_default_model_dir(root_dir, statistic):
    return get_default_model_root_dir(root_dir) / statistic


def get_default_study_dir(root_dir, statistic):
    return Path(root_dir) / EMC_ZHONG24_MODELS_RELATIVE_DIR / "optuna" / statistic


def TrainZhong24Transformer(
    observable,
    model_dir=None,
    transform_input=None,
    transform_output=None,
    val_fraction=0.1,
    seed=None,
    trial=None,
    enable_pruning=False,
    n_tokens=10,
    d_model=96,
    n_heads=4,
    n_layers=2,
    dim_feedforward=192,
    dropout_rate=0.05,
    learning_rate=5.0e-4,
    weight_decay=1.0e-2,
    scheduler_patience=ZHONG24_SCHEDULER_PATIENCE,
    scheduler_factor=ZHONG24_SCHEDULER_FACTOR,
    scheduler_threshold=ZHONG24_SCHEDULER_THRESHOLD,
    batch_size=ZHONG24_BATCH_SIZE,
    max_epochs=ZHONG24_MAX_EPOCHS,
    early_stop_patience=ZHONG24_EARLY_STOP_PATIENCE,
    early_stop_min_delta=ZHONG24_EARLY_STOP_MIN_DELTA,
):
    seed = 42 if seed is None else int(seed)
    seed_everything(seed, workers=True)

    if enable_pruning and trial is None:
        raise ValueError("Optuna trial is required when pruning is enabled.")

    _require_train_test_split(observable)

    train_x = observable.x_train
    train_y = observable.y_train
    print(f"Loaded training split with shape: {train_x.shape}, {train_y.shape}")

    input_transform = _build_transform(transform_input)
    output_transform = _build_transform(transform_output)
    covariance_matrix = _compute_training_covariance_matrix(
        observable=observable,
        output_transform=output_transform,
    )
    print(f"Loaded covariance matrix with shape: {covariance_matrix.shape}")

    if input_transform is not None:
        train_x = input_transform.transform(train_x)

    if output_transform is not None:
        train_y = output_transform.transform(train_y)

    print(f"Using {len(train_y)} training samples before validation split")

    data = ArrayDataModule(
        x=train_x,
        y=train_y,
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_workers=0,
    )
    data.setup()

    train_x, train_y = data.ds_train.tensors
    train_mean = train_y.detach().cpu().numpy().mean(axis=0)
    train_std = train_y.detach().cpu().numpy().std(axis=0)

    train_mean_x = train_x.detach().cpu().numpy().mean(axis=0)
    train_std_x = train_x.detach().cpu().numpy().std(axis=0)

    model = Zhong24Transformer(
        n_input=data.n_input,
        n_output=data.n_output,
        n_tokens=n_tokens,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        scheduler_threshold=scheduler_threshold,
        loss="GaussianNLoglike",
        training=True,
        mean_output=train_mean,
        std_output=train_std,
        mean_input=train_mean_x,
        std_input=train_std_x,
        transform_input=input_transform,
        transform_output=output_transform,
        standarize_output=True,
        covariance_matrix=covariance_matrix,
    )

    model_dir = Path("./" if model_dir is None else model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / "checkpoints"
    print(f"Saving model to {model_dir}")

    trainer_kwargs = dict(
        max_epochs=max_epochs,
        devices=1,
        logger="tensorboard",
        log_dir=model_dir,
        tensorboard_name="tensorboard",
        checkpoint_dir=checkpoint_dir,
        early_stop_patience=early_stop_patience,
        early_stop_threshold=early_stop_min_delta,
        deterministic=True,
    )
    if enable_pruning:
        trainer_kwargs["callbacks"] = _build_trainer_callbacks(
            trial=trial,
            checkpoint_dir=checkpoint_dir,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
        )

    trainer = train.FCNTrainer(**trainer_kwargs)
    val_loss = trainer.fit(
        model=model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    checkpoint_path = _export_best_checkpoint(
        trainer,
        model_dir / f"{observable.stat_name}.ckpt",
    )
    print(f"Saved best checkpoint to {checkpoint_path}")
    return val_loss


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the Zhong et al. (2024) transformer emulator for EMC observables."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=DEFAULT_ROOT_DIR.as_posix(),
        help="Base directory for default EMC input and output paths.",
    )
    parser.add_argument(
        "--transform_input",
        type=str,
        choices=["log", "arcsinh"],
        default=None,
        help="Transform to apply to inputs.",
    )
    parser.add_argument(
        "--transform_output",
        type=str,
        choices=["log", "arcsinh"],
        default=None,
        help="Transform to apply to outputs.",
    )
    parser.add_argument(
        "-s",
        "--statistic",
        type=str,
        default="bispectrum",
        help="Statistic to train on.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory to save the model.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Random fraction of training samples to hold out for validation within ArrayDataModule.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n_tokens",
        type=int,
        default=10,
        help="Number of learned embedding tokens.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=96,
        help="Token embedding width.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of transformer blocks.",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=192,
        help="Width of the token-wise feedforward layers.",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.05,
        help="Dropout rate used in the transformer blocks.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5.0e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1.0e-2,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=ZHONG24_SCHEDULER_PATIENCE,
        help="ReduceLROnPlateau patience in epochs.",
    )
    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=ZHONG24_SCHEDULER_FACTOR,
        help="ReduceLROnPlateau multiplicative decay factor.",
    )
    parser.add_argument(
        "--scheduler_threshold",
        type=float,
        default=ZHONG24_SCHEDULER_THRESHOLD,
        help="Absolute improvement threshold for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=ZHONG24_BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=ZHONG24_MAX_EPOCHS,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=ZHONG24_EARLY_STOP_PATIENCE,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=ZHONG24_EARLY_STOP_MIN_DELTA,
        help="Minimum validation-loss improvement for early stopping.",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    setup_logging()

    root_dir = Path(args.root_dir)
    observable = make_observable(root_dir=root_dir, statistic=args.statistic)

    if args.model_dir is not None:
        model_dir = args.model_dir
    else:
        model_dir = get_default_model_dir(
            root_dir=root_dir,
            statistic=args.statistic,
        )

    TrainZhong24Transformer(
        observable=observable,
        model_dir=model_dir,
        transform_input=args.transform_input,
        transform_output=args.transform_output,
        val_fraction=args.val_fraction,
        seed=args.seed,
        n_tokens=args.n_tokens,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_threshold=args.scheduler_threshold,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
