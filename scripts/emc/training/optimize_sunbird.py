import argparse
import json
import joblib
import logging
from pathlib import Path
import shutil
import warnings

import optuna

from acm import setup_logging
from train_sunbird import (
    DEFAULT_ROOT_DIR,
    TrainFCN,
    get_default_study_dir,
    make_observable,
)

DEFAULT_INITIAL_TRIAL = {
    'n_layers': 4,
    'n_hidden': 512,
    'learning_rate': 1.0e-3,
    'dropout_rate': 0.0,
    'weight_decay': 0.0,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description='Optimize FCN hyperparameters for EMC observables.',
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default=DEFAULT_ROOT_DIR.as_posix(),
        help='Base directory for default EMC input and output paths.',
    )
    parser.add_argument(
        '--transform_input',
        type=str,
        choices=['log', 'arcsinh'],
        default=None,
        help='Transform to apply to inputs.',
    )
    parser.add_argument(
        '--transform_output',
        type=str,
        choices=['log', 'arcsinh'],
        default=None,
        help='Transform to apply to outputs.',
    )
    parser.add_argument(
        '--transform',
        type=str,
        choices=['log', 'arcsinh'],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '-s',
        '--statistic',
        type=str,
        default='bispectrum',
        help='Statistic to optimize.',
    )
    parser.add_argument(
        '--sampler',
        type=str,
        choices=['tpe', 'cmaes'],
        default='tpe',
        help='Optuna sampler to use when creating a new study.',
    )
    parser.add_argument(
        '--pruner',
        type=str,
        choices=['median', 'none'],
        default='median',
        help='Optuna pruner to use when creating a new study.',
    )
    parser.add_argument(
        '--study_dir',
        type=str,
        default=None,
        help='Directory to save optimization artifacts.',
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='Optional directory to publish the best trial artifacts.',
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=100,
        help='Total number of Optuna trials to run.',
    )
    parser.add_argument(
        '--val_fraction',
        type=float,
        default=0.1,
        help='Random fraction of training samples to hold out for validation within ArrayDataModule.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility.',
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _resolve_transform_output(args):
    transform_output = args.transform_output
    if args.transform is None:
        return transform_output

    warnings.warn(
        '`--transform` is deprecated; use `--transform_output` instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    if transform_output is not None:
        return transform_output
    return args.transform


def _get_study_dir(root_dir, statistic, study_dir=None):
    if study_dir is not None:
        return Path(study_dir)
    return get_default_study_dir(root_dir=root_dir, statistic=statistic)


def _get_trial_dir(study_dir, trial_number):
    return Path(study_dir) / 'trials' / f'trial_{trial_number:04d}'


def _validate_study_dir(study_dir):
    study_dir = Path(study_dir)
    study_fn = study_dir / 'study.pkl'
    trials_dir = study_dir / 'trials'
    if not study_fn.exists() and trials_dir.exists() and any(trials_dir.iterdir()):
        raise RuntimeError(
            f'Found retained trial directories in {trials_dir} but no {study_fn.name}. '
            'Refusing to start a new study because trial numbering could collide.'
        )


def _has_completed_trials(study):
    return any(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )


def _count_started_trials(study):
    return sum(
        trial.state != optuna.trial.TrialState.WAITING
        for trial in study.trials
    )


def _build_best_trial_payload(study, published_model_dir=None):
    if not _has_completed_trials(study):
        return None

    best_trial = study.best_trial
    payload = {
        'number': best_trial.number,
        'value': float(best_trial.value),
        'params': best_trial.params,
        'model_dir': best_trial.user_attrs.get('model_dir'),
        'checkpoint_path': best_trial.user_attrs.get('checkpoint_path'),
    }
    if published_model_dir is not None:
        payload['published_model_dir'] = str(published_model_dir)
    return payload


def _write_best_trial_summary(study, summary_path, published_model_dir=None):
    payload = _build_best_trial_payload(
        study=study,
        published_model_dir=published_model_dir,
    )
    if payload is None:
        return None

    summary_path = Path(summary_path)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload


def _replace_published_artifacts(source_model_dir, destination_model_dir, stat_name):
    source_model_dir = Path(source_model_dir)
    destination_model_dir = Path(destination_model_dir)
    destination_model_dir.mkdir(parents=True, exist_ok=True)

    source_checkpoint = source_model_dir / f'{stat_name}.ckpt'
    if not source_checkpoint.exists():
        raise FileNotFoundError(
            f'Best-trial checkpoint not found at {source_checkpoint}.'
        )

    destination_checkpoint = destination_model_dir / f'{stat_name}.ckpt'
    destination_checkpoint.unlink(missing_ok=True)
    shutil.copy2(source_checkpoint, destination_checkpoint)

    for artifact_dir_name in ('checkpoints', 'tensorboard'):
        source_artifact_dir = source_model_dir / artifact_dir_name
        destination_artifact_dir = destination_model_dir / artifact_dir_name
        if destination_artifact_dir.exists():
            shutil.rmtree(destination_artifact_dir)
        if source_artifact_dir.exists():
            shutil.copytree(source_artifact_dir, destination_artifact_dir)

    return destination_model_dir


class EMCObjective:
    def __init__(
        self,
        observable,
        study_dir,
        transform_input=None,
        transform_output=None,
        val_fraction=0.1,
        seed=42,
    ):
        self.observable = observable
        self.study_dir = Path(study_dir)
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.val_fraction = val_fraction
        self.seed = seed
        self.logger = logging.getLogger('EMCObjective')

    def __call__(self, trial):
        learning_rate = trial.suggest_float('learning_rate', 1.0e-3, 0.01)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.001)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.15)
        n_layers = trial.suggest_int('n_layers', 1, 10)
        n_hidden = [trial.suggest_int('n_hidden', 128, 1024)] * n_layers
        enable_pruning = not isinstance(
            trial.study.pruner,
            optuna.pruners.NopPruner,
        )

        trial_model_dir = _get_trial_dir(self.study_dir, trial.number)
        if trial_model_dir.exists():
            raise RuntimeError(
                f'Trial directory already exists for trial {trial.number}: {trial_model_dir}'
            )

        trial.set_user_attr('model_dir', str(trial_model_dir))
        self.logger.info('Running trial %s in %s', trial.number, trial_model_dir)
        val_loss = TrainFCN(
            observable=self.observable,
            learning_rate=learning_rate,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            model_dir=trial_model_dir,
            transform_input=self.transform_input,
            transform_output=self.transform_output,
            val_fraction=self.val_fraction,
            seed=self.seed,
            trial=trial if enable_pruning else None,
            enable_pruning=enable_pruning,
        )
        checkpoint_path = trial_model_dir / f'{self.observable.stat_name}.ckpt'
        if checkpoint_path.exists():
            trial.set_user_attr('checkpoint_path', str(checkpoint_path))
        return float(val_loss)


def _make_sampler(name, seed):
    if name == 'tpe':
        return optuna.samplers.TPESampler(seed=seed)
    if name == 'cmaes':
        return optuna.samplers.CmaEsSampler(seed=seed)
    raise ValueError(f'Unknown sampler: {name}')


def _make_pruner(name):
    if name == 'median':
        return optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=20,
            interval_steps=1,
        )
    if name == 'none':
        return optuna.pruners.NopPruner()
    raise ValueError(f'Unknown pruner: {name}')


def _load_or_create_study(study_fn, statistic, seed, sampler_name, pruner_name):
    study_fn = Path(study_fn)
    if study_fn.exists():
        study = joblib.load(study_fn)
        logger = logging.getLogger('optimize_sunbird')
        actual_sampler_name = study.sampler.__class__.__name__
        requested_sampler_name = _make_sampler(
            sampler_name,
            seed=seed,
        ).__class__.__name__
        actual_pruner_name = study.pruner.__class__.__name__
        requested_pruner_name = _make_pruner(pruner_name).__class__.__name__
        if (
            actual_sampler_name == requested_sampler_name
            and actual_pruner_name == requested_pruner_name
        ):
            logger.info(
                'Loading existing study from %s with sampler=%s and pruner=%s',
                study_fn,
                actual_sampler_name,
                actual_pruner_name,
            )
        else:
            logger.info(
                'Loading existing study from %s with sampler=%s and pruner=%s; '
                'ignoring requested sampler=%s pruner=%s',
                study_fn,
                actual_sampler_name,
                actual_pruner_name,
                sampler_name,
                pruner_name,
            )
        return study

    sampler = _make_sampler(sampler_name, seed=seed)
    pruner = _make_pruner(pruner_name)
    logging.getLogger('optimize_sunbird').info(
        'Creating new study with sampler=%s and pruner=%s',
        sampler.__class__.__name__,
        pruner.__class__.__name__,
    )
    study = optuna.create_study(
        study_name=statistic,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
    )
    study.enqueue_trial(DEFAULT_INITIAL_TRIAL)
    return study


def main(argv=None):
    args = parse_args(argv)
    setup_logging()
    logger = logging.getLogger('optimize_sunbird')

    transform_output = _resolve_transform_output(args)
    root_dir = Path(args.root_dir)
    study_dir = _get_study_dir(
        root_dir=root_dir,
        statistic=args.statistic,
        study_dir=args.study_dir,
    )
    study_dir.mkdir(parents=True, exist_ok=True)
    _validate_study_dir(study_dir)

    study_fn = study_dir / 'study.pkl'
    summary_fn = study_dir / 'best_trial.json'
    observable = make_observable(root_dir=root_dir, statistic=args.statistic)
    objective = EMCObjective(
        observable=observable,
        study_dir=study_dir,
        transform_input=args.transform_input,
        transform_output=transform_output,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    study = _load_or_create_study(
        study_fn=study_fn,
        statistic=args.statistic,
        seed=args.seed,
        sampler_name=args.sampler,
        pruner_name=args.pruner,
    )

    started_trials = _count_started_trials(study)
    logger.info(
        'Study has %s started trials before optimization.',
        started_trials,
    )

    trials_to_run = max(args.n_trials - started_trials, 0)
    for _ in range(trials_to_run):
        study.optimize(objective, n_trials=1)
        joblib.dump(study, study_fn)
        _write_best_trial_summary(study, summary_fn)

    if not study_fn.exists():
        joblib.dump(study, study_fn)
    published_model_dir = None
    if args.model_dir is not None and _has_completed_trials(study):
        best_model_dir = study.best_trial.user_attrs['model_dir']
        published_model_dir = _replace_published_artifacts(
            source_model_dir=best_model_dir,
            destination_model_dir=args.model_dir,
            stat_name=args.statistic,
        )

    payload = _write_best_trial_summary(
        study,
        summary_fn,
        published_model_dir=published_model_dir,
    )
    if payload is not None:
        logger.info(
            'Best trial %s achieved val_loss=%s',
            payload['number'],
            payload['value'],
        )
    return study


if __name__ == '__main__':
    main()
