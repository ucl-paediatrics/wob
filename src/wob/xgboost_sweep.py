"""Run an XGBoost hyperparameter sweep to predict WOB outcome from signs"""
import argparse
from datetime import datetime
from functools import partial
import time
import warnings

import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import polars as pl

import wob
from wob import base

df_wob = pl.read_parquet(wob.WOB_MEASUREMENTS_PATH)
df_outcomes = pl.read_parquet(wob.OUTCOMES_PATH)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=UserWarning, 
        message="Sortedness of columns cannot be checked when 'by' groups provided"
    )
    train_data = base.join_observations_to_outcomes(
        df_wob,
        df_outcomes,
        tolerance_hours=12,
        max_obs_datetime=pl.datetime(2024,1,1) #type: ignore
    )

train_data = base.add_time_to_event_columns(train_data)

X_train = train_data.select(base.WOB_SIGNS)
y_train = train_data.select(pl.col('event_within_time_tolerance')).to_numpy().ravel()

def objective(trial, filename: str, random_seed: int) -> float:
    """Objective function for Optuna hyperparameter optimization."""
    train_x, val_x, train_y, val_y = train_test_split(
        X_train,
        y_train,
        test_size = 0.2,
        random_state = random_seed * 1_000_000 + trial.number
    )

    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',  # This is the only one that makes sense for our problem
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 100.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 500.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'base_score': trial.suggest_categorical('base_score', [0.5, 0.002597])
    }

    model = xgb.XGBClassifier(**param)
    model.fit(train_x, train_y)

    preds = model.predict_proba(val_x)[:,1]

    auroc = roc_auc_score(val_y, preds)

    # Save the parameters and AUROC for this trial to a CSV file
    with open(filename, 'a', encoding='utf-8') as f:
        if f.tell() == 0:  # If file is empty, write header
            f.write('trial_number,' + ','.join(param.keys()) + ',auroc\n')
        f.write(f"{trial.number}," +
                ','.join(str(param[key]) for key in list(param)) +
                f",{auroc}\n"
        )

    return float(auroc)

def main(n_trials, output_dir: str, random_seed: int):
    """Run the Optuna hyperparameter optimization for XGBoost."""
    start_time = time.time()
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna_param_filename = f"xgboost_optuna_params_{n_trials}_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    optuna_output_path = output_dir + optuna_param_filename

    # Use partial to wrap the objective function with the filename and random seed parameters,
    # so that Optuna can call it with just the trial parameter.
    # This is like a lambda, but lambdas shouldn't be assigned to variables.
    objective_function = partial(objective, filename=optuna_output_path, random_seed=random_seed)

    study.optimize(objective_function, n_trials=n_trials)
    end_time = time.time()

    print("\n\n------------------------------\n")
    print(f"Completed {n_trials} trials in {end_time - start_time:.2f} seconds.")
    print(f"Results saved to: {optuna_output_path}")
    print(f"Best trial AUROC: {study.best_trial.value}")
    print(f"Best parameters: {study.best_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_trials", type=int, default=50, 
        help="Number of Optuna trials to run for hyperparameter optimization."
    )
    parser.add_argument(
        "--output_dir", type=str, default="/home/workspace/files/Sairam - WOB/",
        help="Base directory to save the Optuna results CSV file."
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help=(
            "Random seed for reproducibility of results, combined with trial "
            "number to ensure different splits across trials. Default is 42."
        )
    )

    args = parser.parse_args()
    main(
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )
