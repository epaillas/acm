import optuna
import joblib
import logging
from acm.utils import setup_logging 
from pathlib import Path
from acm.model.optimize import objective, get_best_model
from acm.projects.bgs import *
study_statistics = {
    'tpcf': GalaxyCorrelationFunctionMultipoles,
    'dsc_conf': DensitySplitCorrelationFunctionMultipoles,
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=100, help="Number of trials to run")
parser.add_argument("--statistic", type=str, default='tpcf', help="Statistic to optimize")
parser.add_argument("--same_n_hidden", type=bool, default=True, help="If True, all the hidden layers will have the same number of neurons. If False, each hidden layer will have a different number of neurons.")
args = parser.parse_args()
statistic = args.statistic
same_n_hidden = args.same_n_hidden
n_trials = args.n_trials

statistic = study_statistics[statistic](
    # No filters for now
)

# Set up the study directory
study_dir = statistic.paths['study_dir']
Path(study_dir).mkdir(parents=True, exist_ok=True)
study_fn = Path(study_dir) / f'{statistic.stat_name}.pkl'

# Setup logging
logger_fn = Path(study_dir) / f'{statistic.stat_name}.log'
setup_logging(filename=logger_fn)
logger = logging.getLogger(__file__.split('/')[-1])
logger.info(f"Optimizing hyperparameters for {statistic.stat_name}")
logger.info(f"Running {n_trials} trials")

# Check on the study directories to avoid errors later
n_existing_models = len(list((Path(study_dir)/statistic.stat_name).rglob('last*.ckpt')))
if not study_fn.exists() and n_existing_models > 1:
    logger.warning(
        f"The study file {study_fn} does not exist, but {n_existing_models} "
        f"models are already saved in the study directory. "
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
from sunbird.data.transforms_array import ArcsinhTransform
kwargs = {
    'statistic': statistic.stat_name,
    'lhc_dir': statistic.paths['lhc_dir'],
    'covariance_dir': statistic.paths['covariance_dir'],
    'model_dir': statistic.paths['study_dir'], # Save the intermediate models in the study directory
    'n_test': 600,
    'transform': ArcsinhTransform(),
    'summary_coords_dict': statistic.summary_coords_dict,
}

# Run the optimization
for i in range(n_trials): # Loop to save each trial in the study
    if study_fn.exists():
        logger.info(f"Loading existing study from {study_fn}")
        study = joblib.load(study_fn)
    else:
        study = optuna.create_study(study_name=f'{statistic.stat_name}')
    optimize_objective = lambda trial: objective(
        trial,  
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
copy_to = statistic.paths['model_dir']
model_fn = get_best_model(statistic.stat_name, study_dir, checkpoint_offset=0, copy_to=copy_to)
logger.info(f"Best model saved at {model_fn} and copied to {copy_to}")