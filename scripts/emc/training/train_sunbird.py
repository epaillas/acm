import numpy as np
import shutil
from pathlib import Path
from lightning import seed_everything
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule
from sunbird.data.transforms_array import LogTransform, ArcsinhTransform
import torch
from acm.observables import Observable
from acm import setup_logging
import argparse

torch.set_float32_matmul_precision('high')

DEFAULT_ROOT_DIR = Path('/global/cfs/cdirs/desicollab/users/epaillas/acm/')
EMC_MEASUREMENTS_RELATIVE_DIR = Path('emc/measurements/v1.3/abacus')
EMC_MODELS_RELATIVE_DIR = Path('emc/models/v1.3')
REQUIRED_SPLIT_VARS = ('x_train', 'y_train', 'x_test', 'y_test')


def _build_transform(transform_name):
    if transform_name is None:
        return None
    if transform_name == 'log':
        return LogTransform()
    if transform_name == 'arcsinh':
        return ArcsinhTransform()
    raise ValueError(f'Unknown transform: {transform_name}')


def get_emc_paths(root_dir):
    root_dir = Path(root_dir)
    measurements_dir = root_dir / EMC_MEASUREMENTS_RELATIVE_DIR
    return {
        'data_dir': measurements_dir / 'compressed',
        'measurements_dir': measurements_dir,
        'param_dir': None,
    }


def make_observable(root_dir, statistic):
    return Observable(
        stat_name=statistic,
        paths=get_emc_paths(root_dir),
        numpy_output=True,
        flat_output_dims=2,
    )


def get_default_model_dir(root_dir, statistic):
    return Path(root_dir) / EMC_MODELS_RELATIVE_DIR / 'best' / statistic


def get_default_study_dir(root_dir, statistic):
    return Path(root_dir) / EMC_MODELS_RELATIVE_DIR / 'optuna' / statistic


def _compute_training_covariance_matrix(
    observable,
    output_transform=None,
    volume_factor=64,
):
    covariance_samples = np.asarray(observable.covariance_y, dtype=float)
    if covariance_samples.ndim < 2:
        raise ValueError('Covariance samples must have at least two dimensions.')

    covariance_samples = covariance_samples.reshape(covariance_samples.shape[0], -1)

    if output_transform is not None:
        covariance_samples = np.asarray(
            output_transform.transform(covariance_samples),
            dtype=float,
        )
        if not np.isfinite(covariance_samples).all():
            stat_name = getattr(observable, 'stat_name', 'unknown')
            raise ValueError(
                f'Output transform produced non-finite covariance samples for '
                f"'{stat_name}'."
            )

    return np.cov(covariance_samples, rowvar=False) / volume_factor


def _require_train_test_split(observable):
    dataset_vars = set(observable._dataset.data_vars)
    missing = [name for name in REQUIRED_SPLIT_VARS if name not in dataset_vars]
    if missing:
        stat_name = getattr(observable, 'stat_name', 'unknown')
        missing_str = ', '.join(missing)
        raise ValueError(
            f"Compressed EMC dataset for '{stat_name}' is missing {missing_str}. "
            'This training script requires the train/test split produced during '
            f"compression. Re-run `scripts/emc/measurements/compress_files.py --statistic {stat_name}` "
            'for this statistic and try again.'
        )


def _get_best_checkpoint_path(trainer):
    for callback in getattr(trainer, 'callbacks', []):
        best_model_path = getattr(callback, 'best_model_path', None)
        if best_model_path:
            return Path(best_model_path)
    raise RuntimeError('Training completed without a saved best checkpoint.')


def _export_best_checkpoint(trainer, output_path):
    best_checkpoint_path = _get_best_checkpoint_path(trainer)
    output_path = Path(output_path)
    output_path.unlink(missing_ok=True)
    shutil.copy2(best_checkpoint_path, output_path)
    return output_path


def _get_lightning_callback_classes():
    from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar

    return LearningRateMonitor, RichProgressBar


def _get_pruning_callback_cls():
    try:
        from optuna.integration import PyTorchLightningPruningCallback
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Optuna pruning requires `optuna-integration[pytorch_lightning]` '
            'in the sourced cosmodesi environment.'
        ) from exc

    return PyTorchLightningPruningCallback


def _build_trainer_callbacks(
    trial,
    checkpoint_dir,
    early_stop_patience=30,
    early_stop_min_delta=1.0e-7,
):
    LearningRateMonitor, RichProgressBar = _get_lightning_callback_classes()
    pruning_callback_cls = _get_pruning_callback_cls()

    callbacks = [
        train.FCNTrainer.early_stop_callback(
            monitor='val_loss',
            patience=early_stop_patience,
            min_delta=early_stop_min_delta,
        ),
        train.FCNTrainer.checkpoint_callback(
            monitor='val_loss',
            checkpoint_dir=checkpoint_dir,
        ),
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar(),
        pruning_callback_cls(trial, monitor='val_loss'),
    ]
    return [cb for cb in callbacks if cb is not None]


def TrainFCN(observable, learning_rate, n_hidden, dropout_rate, weight_decay,
    model_dir=None, transform_input=None, transform_output=None, val_fraction=0.1,
    seed=None, trial=None, enable_pruning=False):

    seed = 42 if seed is None else int(seed)
    seed_everything(seed, workers=True)

    if enable_pruning and trial is None:
        raise ValueError('Optuna trial is required when pruning is enabled.')

    _require_train_test_split(observable)

    train_x = observable.x_train
    train_y = observable.y_train
    print(f'Loaded training split with shape: {train_x.shape}, {train_y.shape}')

    input_transform = _build_transform(transform_input)
    output_transform = _build_transform(transform_output)
    covariance_matrix = _compute_training_covariance_matrix(
        observable=observable,
        output_transform=output_transform,
    )
    print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

    if input_transform is not None:
        train_x = input_transform.transform(train_x)

    if output_transform is not None:
        train_y = output_transform.transform(train_y)

    print(f'Using {len(train_y)} training samples before validation split')

    data = ArrayDataModule(
        x=train_x,
        y=train_y,
        val_fraction=val_fraction,
        batch_size=128,
        num_workers=0
    )
    data.setup()

    train_x, train_y = data.ds_train.tensors
    train_mean = train_y.detach().cpu().numpy().mean(axis=0)
    train_std = train_y.detach().cpu().numpy().std(axis=0)

    train_mean_x = train_x.detach().cpu().numpy().mean(axis=0)
    train_std_x = train_x.detach().cpu().numpy().std(axis=0)

    model = FCN(
            n_input=data.n_input,
            n_output=data.n_output,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate, 
            learning_rate=learning_rate,
            scheduler_patience=10,
            scheduler_factor=0.5,
            scheduler_threshold=1.e-6,
            weight_decay=weight_decay,
            act_fn='learned_sigmoid',
            loss='GaussianNLoglike',
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

    model_dir = Path('./' if model_dir is None else model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / 'checkpoints'
    print(f'Saving model to {model_dir}')

    trainer_kwargs = dict(
        max_epochs=5000,
        devices=1,
        logger='tensorboard',
        log_dir=model_dir,
        tensorboard_name='tensorboard',
        checkpoint_dir=checkpoint_dir,
        deterministic=True,
    )
    if enable_pruning:
        trainer_kwargs['callbacks'] = _build_trainer_callbacks(
            trial=trial,
            checkpoint_dir=checkpoint_dir,
        )

    trainer = train.FCNTrainer(**trainer_kwargs)
    val_loss = trainer.fit(
        model=model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    checkpoint_path = _export_best_checkpoint(
        trainer,
        model_dir / f'{observable.stat_name}.ckpt',
    )
    print(f'Saved best checkpoint to {checkpoint_path}')
    return val_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train FCN for EMC observables.')
    parser.add_argument(
        '--root_dir',
        type=str,
        default=DEFAULT_ROOT_DIR.as_posix(),
        help='Base directory for default EMC input and output paths.',
    )
    parser.add_argument('--transform_input', type=str, choices=['log', 'arcsinh'], default=None, help='Transform to apply to inputs.')
    parser.add_argument('--transform_output', type=str, choices=['log', 'arcsinh'], default=None, help='Transform to apply to outputs.')
    parser.add_argument('-s', '--statistic', type=str, default='bispectrum', help='Statistic to train on.')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory to save the model.')
    parser.add_argument('--val_fraction', type=float, default=0.1, help='Random fraction of training samples to hold out for validation within ArrayDataModule.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    setup_logging()

    root_dir = Path(args.root_dir)
    observable = make_observable(root_dir=root_dir, statistic=args.statistic)

    if args.model_dir is not None:
        model_dir = args.model_dir
    else:
        model_dir = get_default_model_dir(root_dir=root_dir, statistic=args.statistic)

    TrainFCN(
        observable=observable,
        learning_rate=1.e-3,
        n_hidden=[512, 512, 512, 512],
        dropout_rate=0.,
        weight_decay=0,
        model_dir=model_dir,
        transform_input=args.transform_input,
        transform_output=args.transform_output,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
