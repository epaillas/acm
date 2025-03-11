import optuna
import joblib
from pathlib import Path
from train_sunbird_transformer import TrainTransformer
import argparse


def objective(
    trial,
):
    learning_rate = trial.suggest_float(
        "learning_rate",
        1.0e-3,
        0.01,
    )
    weight_decay = trial.suggest_float("weight_decay", 1.0e-5, 0.001)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.15)
    num_layers = trial.suggest_int("num_layers", 1, 10, 1)
    nhead = trial.suggest_int("nhead", 1, 16, 1)
    d_model = nhead * 64
    dim_feedforward = 2 * d_model
    return TrainTransformer(learning_rate=learning_rate, dropout_rate=dropout_rate,
                            weight_decay=weight_decay, num_layers=num_layers,
                            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                            statistic=statistic)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--statistic", type=str, default='tpcf')
    args = parser.parse_args()
    statistic = args.statistic

    n_trials = 100
    study_dir = f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/{statistic}/cosmo+hod/transformer/test/optuna/'
    Path(study_dir).mkdir(parents=True, exist_ok=True)
    study_fn = Path(study_dir) / f'{statistic}.pkl'

    for i in range(n_trials):
        if study_fn.exists():
            print(f"Loading existing study from {study_fn}")
            study = joblib.load(study_fn)
        else:
            study = optuna.create_study(study_name=f'{statistic}')
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