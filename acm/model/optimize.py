import joblib
import optuna
import shutil
import logging
import tempfile
import numpy as np
from pathlib import Path
# from warnings import deprecated # Available only in Python 3.13+
from pytorch_lightning import seed_everything
from sunbird.emulators import FCN
from acm.model.train import TrainFCN

class FCNObjective:
    def __init__(self, same_n_hidden: bool = True, hyperparameters_info: dict = None, save_method: str = None, save_dir: str | Path = None, **kwargs):
        """
        Callable class to be used as the objective function for Optuna hyperparameter optimization.
        
        Parameters
        ----------
        same_n_hidden : bool
            Whether to use the same number of hidden units in each layer.
        hyperparameters_info : dict, optional
            Information about hyperparameter ranges and types. 
            Currently not used, but can be implemented to customize hyperparameter suggestions.
            Defaults to None.
        **kwargs :
            Additional keyword arguments to pass to the TrainFCN function.
        """
        self.same_n_hidden = same_n_hidden
        self.training_kwargs = kwargs
        self.logger = logging.getLogger('Objective')
        
        # TODO: Use hyperparameters_info to adjust hyperparameter suggestion ranges/types
    
    def __call__(self, trial):
        """
        Callable method to be used as the objective function for Optuna hyperparameter optimization.
        
        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial object for suggesting hyperparameters.
        
        Returns
        -------
        val_loss : float
            Validation loss obtained after training the model with the suggested hyperparameters.
        """
        # Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1.0e-3, 0.01)
        weight_decay = trial.suggest_float("weight_decay", 1.0e-5, 0.001)
        n_layers = trial.suggest_int("n_layers", 1, 10)
        if self.same_n_hidden:
            n_hidden = [trial.suggest_int("n_hidden", 200, 1024)] * n_layers
        else:
            n_hidden = [
                trial.suggest_int(f"n_hidden_{layer}", 200, 1024)
                for layer in range(n_layers)
            ]
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.15)
        
        kwargs = self.training_kwargs.copy()
        checkpoint_filename = kwargs.get('checkpoint_filename', '{epoch:02d}-{val_loss:.5f}')
        kwargs.update(dict(
            checkpoint_filename = f'trial={trial.number}-' + checkpoint_filename,
            return_trainer = True,
        ))
        
        # Train the model with the hyperparameters
        val_loss, trainer = TrainFCN(
            learning_rate=learning_rate,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
        # Store the trial checkpoint path as a user attribute
        for callback in trainer.callbacks:
            if hasattr(callback, 'best_model_path'):
                checkpoint_fn = callback.best_model_path
                trial.set_user_attr('checkpoint', checkpoint_fn)
        return val_loss
    
    #%% Callbacks
    def log_best_trial(self, study, trial):
        """
        Callback to log the best trial information after each trial.
        """
        best_trial = study.best_trial
        self.logger.info(f"Trial {trial.number} done. Best trial is {best_trial.number} with value {best_trial.value}")  
    
    

def StudyFCN(
    n_trials: int, 
    same_n_hidden: bool, 
    study_fn: str|Path, 
    seed: float = 42, 
    pruner: optuna.pruners.BasePruner|None = None,
    load_if_exists: bool = True, 
    hyperparameters_info: dict = None,
    save_dir: str|Path|None = None,
    **kwargs
    ) -> optuna.study.Study:
    """
    Runs an Optuna study to optimize hyperparameters for training a model.
    Saves the study to `study_fn` after each trial.

    Parameters
    ----------
    n_trials : int
        Number of trials to run in the study.
    same_n_hidden : bool
        Whether to use the same number of hidden units in the model.
    study_fn : str | Path
        File path to save or load the Optuna study.
    hyperparameters_info : dict, optional
        Information about hyperparameter ranges and types. 
        Currently not used, but can be implemented to customize hyperparameter suggestions.
        Defaults to None.
    pruner : optuna.pruners.BasePruner | None, optional
        Optuna pruner to use for early stopping of unpromising trials. 
        If None, optuna sets the pruner to `optuna.pruners.MedianPruner()`.
        Defaults to None.
    load_if_exists : bool, optional
        Whether to load the study if it exists. Defaults to True.
    seed : float, optional
        Random seed for reproducibility. Defaults to 42.
    save_dir : str | Path | None, optional
        Directory to save the best model checkpoints. If given, the best model checkpoint will be moved here after the study.
        If None, the best model is not saved separately. Defaults to None.
    **kwargs :
        Additional keyword arguments to pass to the Objective class (and by extension to TrainFCN function).
        
    Returns
    -------
    study : optuna.study.Study
        The completed Optuna study after running the specified number of trials.
        Also contains user attributes for `same_n_hidden` and `seed`.
    """
    logger = logging.getLogger('StudyFCN')
    seed_everything(seed, workers=True) # Ensure reproducibility
    
    # Load or create study
    study_fn = Path(study_fn)
    n_trials_done = 0
    if study_fn.exists() and load_if_exists:
        logger.info(f'Loading existing study from {study_fn}')
        study = joblib.load(study_fn)
        n_trials_done = len(study.trials)
        if n_trials_done >= n_trials:
            logger.info(f'Study already has {n_trials_done} trials, which is >= {n_trials}. Skipping optimization.')
            return study
    else:
        logger.info(f'Creating new study at {study_fn}')
        study = optuna.create_study(study_name=study_fn.stem, direction='minimize', pruner=pruner)
        study.set_metric_names(['val_loss'])
        study.set_user_attr('seed', seed) # Store seed used for the study
    
    # Run the optimization
    objective = FCNObjective(
        same_n_hidden=same_n_hidden, 
        hyperparameters_info=hyperparameters_info, 
        **kwargs)
    callbacks = [objective.log_best_trial]
    # Loop over remaining trials to allow saving after each trial
    for i in range(n_trials - n_trials_done):
        study.optimize(objective, n_trials=1, callbacks=callbacks)
        joblib.dump(study, study_fn)
    
    checkpoint_dir = kwargs.get('checkpoint_dir', None) # Get checkpoint directory from kwargs
    if checkpoint_dir is not None:
        best_checkpoint = sorted(Path(checkpoint_dir).glob(f'trial={study.best_trial.number}*.ckpt'))[0] # In theory, there should be only one
        logger.info(f'Best model checkpoint from study saved at {best_checkpoint}')
        # Move best checkpoint to save_dir if specified
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_best_fn = save_dir / f'{study_fn.stem}.ckpt'
            shutil.copy(best_checkpoint, save_best_fn)
            logger.info(f'Moved best model checkpoint to {save_best_fn}')
    return study

# @deprecated("Use StudyFCN save_dir instead")
def train_best_model(study_fn: str|Path, save_fn: str|Path = None, **kwargs):
    """
    Trains the best model found in the Optuna study saved at `study_fn`.
    Optionally saves the model checkpoint to `save_fn`.

    Parameters
    ----------
    study_fn : str | Path
        File path to the Optuna study.
    save_fn : str | Path, optional
        File path to save the best model checkpoint. If None, the model is not saved. Defaults to None.
    **kwargs :
        Keyword arguments to pass to the TrainFCN function, apart from the hyperparameters read from the study.

    Returns
    -------
    model : torch.nn.Module
        The trained model with the best hyperparameters from the study.

    Raises
    ------
    ValueError
        If the study has no trials.
    """
    logger = logging.getLogger('train_best_model')
        
    study = joblib.load(study_fn)
    n_trials = len(study.trials)
    if n_trials == 0:
        raise ValueError(f'Study file {study_fn} has no trials.')

    seed = study.user_attrs.get('seed', None)
    seed_everything(seed, workers=True)
    
    logger.info(f'Found best trial {study.best_trial.number} with value {study.best_value}')
    
    best_params = study.best_params
    
    # Handle n_hidden parameter reconstruction
    n_hidden = best_params.pop('n_hidden', None)
    n_layers = best_params.pop('n_layers')
    if n_hidden:
        best_params['n_hidden'] = [n_hidden] * n_layers
    else:
        best_params['n_hidden'] = [
            best_params.pop(f'n_hidden_{layer}') for layer in range(n_layers)
        ]
    
    # Since we can't manually handle checkpointing here, we store the checkpoints in a temp directory
    tmp_dir = Path(save_fn).parent if save_fn is not None else None
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        kwargs.update(dict(
            checkpoint_dir=tmp_dir_path,
        ))
        val_loss = TrainFCN(
            **kwargs,
            **best_params,
        )
        
        # Load the model from the best checkpoint
        checkpoint_symlink_fn = tmp_dir_path / 'last.ckpt'
        checkpoint_fn = checkpoint_symlink_fn.resolve()
        model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
        
        # Then we move the best checkpoint to save_fn if specified
        if save_fn is not None:
            save_fn = Path(save_fn)
            save_fn.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_fn.rename(save_fn)
            logger.info(f'Saved best model checkpoint to {save_fn}')
            
    logger.info(f'Trained best model with validation loss: {val_loss}')
    
    if val_loss != study.best_value:
        logger.warning(f'Validation loss from trained best model ({val_loss}) does not match best value from study ({study.best_value}).')
        
    return model

# Example usage in optimize_model.py
if __name__ == '__main__':
    from acm.utils.logging import setup_logging
    setup_logging(level='info')
    
    # Dummy data for testing
    lhc_x = np.random.rand(1000, 10)
    lhc_y = np.random.rand(1000, 5)
    lhc_x_names = [f'feature_{i}' for i in range(10)]
    
    study = StudyFCN(
        n_trials=2,
        same_n_hidden=True,
        study_fn='./study/example_study.pkl',
        lhc_x=lhc_x,
        lhc_y=lhc_y,
        lhc_x_names=lhc_x_names,
        max_epochs=10,
        deterministic=True,
        devices=1,
        checkpoint_dir='./study/checkpoints',
        # save_dir='./study/best_models',
    )