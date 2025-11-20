import optuna
import joblib
from pathlib import Path
from train_sunbird import TrainFCN
from acm.observables import Observable
import argparse


def objective(
    trial,
):
    learning_rate = trial.suggest_float("learning_rate", 1.0e-3, 0.01)
    weight_decay = trial.suggest_float("weight_decay", 1.0e-5, 0.001)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.15)
    same_n_hidden = True
    n_layers = trial.suggest_int("n_layers", 1, 10)
    if same_n_hidden:
        n_hidden = [trial.suggest_int("n_hidden", 128, 1024)] * n_layers
    else:
        n_hidden = [
            trial.suggest_int(f"n_hidden_{layer}", 128, 1024)
            for layer in range(n_layers)
        ]
    return TrainFCN(learning_rate=learning_rate, n_hidden=n_hidden,
                    dropout_rate=dropout_rate, weight_decay=weight_decay,
                    observable=observable, model_dir=study_dir, transform=args.transform)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train FCN for EMC observables.')
    parser.add_argument('--transform', type=str, choices=['log', 'arcsinh'], default=None, help='Transform to apply to outputs.')
    parser.add_argument('-s', '--statistic', type=str, default='bispectrum', help='Statistic to train on.')
    args = parser.parse_args()

    paths = {
        'data_dir': '/pscratch/sd/e/epaillas/emc/v1.2/abacus/compressed/', # Loads x, y can also contain covariance_y
    }

    observable = Observable(stat_name=args.statistic, paths=paths, numpy_output=True, flat_output_dims=2)

    n_trials = 100
    study_dir = f'/pscratch/sd/e/epaillas/emc/v1.2/trained_models/optuna/{observable}/'
    Path(study_dir).mkdir(parents=True, exist_ok=True)
    study_fn = Path(study_dir) / f'{observable}.pkl'

    for i in range(n_trials):
        if study_fn.exists():
            print(f"Loading existing study from {study_fn}")
            study = joblib.load(study_fn)
        else:
            sampler = optuna.samplers.CmaEsSampler()
            study = optuna.create_study(study_name=f'{observable}', sampler=sampler)
            
            # initial guess
            study.enqueue_trial(
                {
                    "n_layers": 4,
                    "n_hidden": 512,
                    "learning_rate": 1e-3,
                    "dropout_rate" : 0,
                    "weight_decay": 0,
                }
            )
        print(f"Sampler is {study.sampler.__class__.__name__}")

        optimize_objective = lambda trial: objective(trial)
        study.optimize(optimize_objective, n_trials=1)

        joblib.dump(
            study,
            study_fn,
        )

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))