import optuna
import joblib
from pathlib import Path
from train_sunbird import TrainFCN
import argparse


def objective(
    trial,
):
    same_n_hidden = True
    learning_rate = trial.suggest_float(
        "learning_rate",
        1.0e-3,
        0.01,
    )
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
    return TrainFCN(learning_rate=learning_rate, n_hidden=n_hidden,
                    dropout_rate=dropout_rate, weight_decay=weight_decay,
                    observable=args.observable, model_dir=study_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--observable", type=str, default='GalaxyBispectrumMultipoles')
    args = parser.parse_args()
    observable = args.observable

    n_trials = 100
    study_dir = f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/{observable}/cosmo+hod/optuna/'
    Path(study_dir).mkdir(parents=True, exist_ok=True)
    study_fn = Path(study_dir) / f'{observable}.pkl'

    for i in range(n_trials):
        if study_fn.exists():
            print(f"Loading existing study from {study_fn}")
            study = joblib.load(study_fn)
        else:
            study = optuna.create_study(study_name=f'{observable}')
        optimize_objective = lambda trial: objective(trial)
        study.optimize(optimize_objective, n_trials=1)

        # print("Best trial:")
        # trial = study.best_trial
        # print("  Value: {}".format(trial.value))
        # print("  Params: ")
        # for key, value in trial.params.items():
        #     print("    {}: {}".format(key, value))
        joblib.dump(
            study,
            study_fn,
        )