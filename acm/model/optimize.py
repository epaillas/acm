import joblib
import os, shutil
from pathlib import Path
from acm.model.train import TrainFCN


def objective(trial, same_n_hidden = True, **kwargs):
    """
    Train the model with the hyperparameters suggested by the optuna optimization.

    Parameters
    ----------
    trial : 
        `optuna` trial object.
    same_n_hidden : bool, optional
        If True, all the hidden layers will have the same number of neurons. If False, each hidden layer will have a different number of neurons.
        Defaults to True.
    **kwargs :
        Keyword arguments to pass to the TrainFCN function, apart from the hyperparameters.
        
    Returns
    -------
    value : float
        Validation loss of the model (from TrainFCN).
    """
    # Define the hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1.0e-3, 0.01)
    weight_decay = trial.suggest_float("weight_decay", 1.0e-5, 0.001)
    n_layers = trial.suggest_int("n_layers", 1, 10)
    if same_n_hidden:
        n_hidden = [trial.suggest_int("n_hidden", 200, 1024)] * n_layers
    else:
        n_hidden = [
            trial.suggest_int(f"n_hidden_{layer}", 200, 1024)
            for layer in range(n_layers)
        ]
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.15)
    
    # Train the model with the hyperparameters
    return TrainFCN(learning_rate=learning_rate,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    weight_decay=weight_decay,
                    **kwargs)


def get_best_model(
    statistic: str,
    study_dir: str,
    checkpoint_offset: int = 0,
    copy_to: str = False,
    model_symlink: str = 'last.ckpt', # To follow pytorch convention
    )-> Path: 
    """
    Get the best model checkpoint from the study.

    Parameters
    ----------
    statistic : str
        Statistic name
    study_dir : str
        Directory where the study is saved.
    checkpoint_offset : int, optional
        How many models already existed in the study directory before the training. Defaults to 0.
    copy_to : str, optional
        If given, the model will be copied to this path. Defaults to False.
        As the standard practice, the model will be copied to a '{statistic}' subdirectory in the given path.
    model_symlink : str, optional 
        Name of the symlink to create when copying the model. If set to None, the symlink will be named 'last.ckpt'. Defaults to None.

    Returns
    -------
    Path
        Path to the best model checkpoint.

    Raises
    ------
    FileNotFoundError
        If the model checkpoint does not exist.
    """
    study_dir = Path(study_dir)
    
    # Open the study, and get the best trial
    study_fn = study_dir / f'{statistic}.pkl'
    study = joblib.load(study_fn)
    best_trial = study.best_trial
    
    # get the best model ckeckpoint name
    if best_trial.number == 0 and checkpoint_offset == 0:
        ckpt = 'last.ckpt'
    else:
        ckpt = f'last-v{best_trial.number + checkpoint_offset}.ckpt'
    
    model_symlnk = study_dir / statistic / ckpt
    model_fn = model_symlnk.resolve()
    
    if not model_fn.exists():
        raise FileNotFoundError(f'The model checkpoint {model_fn} does not exist.')

    # Copy to the desired path and create the symlink
    if copy_to:
        copy_to = copy_to + f'{statistic}/' # ACM standard storage (see train and io_tools)
        Path(copy_to).mkdir(parents=True, exist_ok=True) # Check if the directory exists, if not create it
        model_fn = shutil.copy(model_fn, copy_to) # Copy the model to the desired path
        
        # Create the symlink
        symlink = Path(copy_to) / model_symlink
        os.symlink(model_fn, symlink)
        return model_fn
    
    return model_fn


# NOTE : toy example to test the function
if __name__ == '__main__':
    
    import optuna
    import argparse
    import logging
    from acm.utils import setup_logging 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--statistic", type=str, default='tpcf', help="Statistic to optimize")
    parser.add_argument("--same_n_hidden", type=bool, default=True, help="If True, all the hidden layers will have the same number of neurons. If False, each hidden layer will have a different number of neurons.")
    args = parser.parse_args()
    statistic = args.statistic
    same_n_hidden = args.same_n_hidden
    n_trials = args.n_trials
    
    from acm.data.paths import emc_paths
    study_dir = emc_paths['study_dir']
    Path(study_dir).mkdir(parents=True, exist_ok=True)
    study_fn = Path(study_dir) / f'{statistic}.pkl'
    
    # Setup logging
    logger_fn = Path(study_dir) / f'{statistic}.log'
    setup_logging(filename=logger_fn)
    logger = logging.getLogger(__file__.split('/')[-1])
    logger.info(f"Optimizing hyperparameters for {statistic}")
    logger.info(f"Running {n_trials} trials")
    
    # Check on the study directories to avoid errors later
    n_existing_models = len(list((Path(study_dir)/statistic).rglob('last*.ckpt')))
    if not study_fn.exists() and n_existing_models > 1:
        logger.warning(
            f"The study file {study_fn} does not exist, but {n_existing_models} models are already saved in the study directory. "
            f"When using the 'checkpoint_offset' argument in 'get_best_model', "
            f"make sure to set it to {n_existing_models + 1} to get the last model.")
    if study_fn.exists():
        study = joblib.load(study_fn)
        n_trials_saved = len(study.trials)
        if n_trials_saved != n_existing_models:
           raise ValueError(
                f"The number of trials saved in the study ({n_trials_saved}) is different "
                f"from the number of models saved in the study directory ({n_existing_models}). "
                f"Please check the study file and the saved models."
              )
    
    # TrainFCN parameters (except the hyperparameters)
    from sunbird.data.transforms import Log
    kwargs = {
        'statistic': statistic,
        'lhc_dir': emc_paths['lhc_dir'],
        'covariance_dir': emc_paths['covariance_dir'],
        'model_dir': emc_paths['study_dir'], # Save the intermediate models in the study directory
        'n_train': 600,
        'transform': Log(),
    }
    
    # Run the optimization
    for i in range(n_trials): # Loop to save each trial in the study
        if study_fn.exists():
            logger.info(f"Loading existing study from {study_fn}")
            study = joblib.load(study_fn)
        else:
            study = optuna.create_study(study_name=f'{statistic}')
        optimize_objective = lambda trial: objective(trial,  
                                                     same_n_hidden=same_n_hidden, 
                                                     **kwargs)
        study.optimize(optimize_objective, n_trials=1)

        last_trial_nb = study.trials[-1].number
        logger.info(f"{i+1}/{n_trials}, Trial {last_trial_nb}. Best trial is {study.best_trial.number} with value {study.best_trial.value}")

        joblib.dump(
            study,
            study_fn,
        )
    
    # Get the best model
    copy_to = emc_paths['model_dir']
    model_fn = get_best_model(statistic, study_dir, checkpoint_offset=0, copy_to=copy_to)
    logger.info(f"Best model saved at {model_fn} and copied to {copy_to}")