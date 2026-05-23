"""Prefect pipeline: ingest → features → train → log to MLflow."""

import hashlib
import json
import os

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from prefect import flow, task
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chute_certo.features.build import WINDOW, build_features
from chute_certo.ingestion.processing import load_seasons
from chute_certo.training.train import FEATURE_COLS, TARGET, evaluate, summarize

load_dotenv()
dagshub.init(
    repo_owner=os.environ["DAGSHUB_REPO_OWNER"],
    repo_name=os.environ["DAGSHUB_REPO_NAME"],
    mlflow=True,
)

EXPERIMENT_NAME = "brasileirao-prediction"
SEASONS = [2023, 2024, 2025, 2026]
MIN_TRAIN_ROUNDS = 10
CV_STRATEGY = "walk_forward_by_round"


def _cv_tag(strategy: str, min_train_rounds: int, seasons: list[int]) -> str:
    config = {
        "strategy": strategy,
        "min_train_rounds": min_train_rounds,
        "seasons": seasons,
    }
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


@task(name="load-data")
def task_load_data(seasons: list[int]) -> pd.DataFrame:
    return load_seasons(seasons)


@task(name="build-features")
def task_build_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return build_features(df, window=window)


@task(name="train-and-log", retries=2)
def task_train_and_log(features: pd.DataFrame, cv_tag: str) -> pd.DataFrame:
    mlflow.set_experiment(EXPERIMENT_NAME)
    results = evaluate(features, min_train_rounds=MIN_TRAIN_ROUNDS)
    summary = summarize(results)

    for model_name, row in summary.iterrows():
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("cv_config", cv_tag)
            mlflow.log_params(
                {
                    "model": model_name,
                    "seasons": str(SEASONS),
                    "feature_window": WINDOW,
                    "features": FEATURE_COLS,
                    "cv_strategy": CV_STRATEGY,
                    "cv_min_train_rounds": MIN_TRAIN_ROUNDS,
                    "cv_n_folds": len(results[results["model"] == model_name]),
                }
            )
            mlflow.log_metrics(
                {
                    "f1_macro_mean": row[("f1_macro", "mean")],
                    "f1_macro_std": row[("f1_macro", "std")],
                    "log_loss_mean": row[("log_loss", "mean")],
                    "log_loss_std": row[("log_loss", "std")],
                }
            )

    return results


@task(name="register-final-model")
def task_register_final_model(features: pd.DataFrame) -> None:
    df = features.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    X = df[FEATURE_COLS].to_numpy()
    y = df[TARGET].to_numpy()

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X, y)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="logistic_regression_final"):
        mlflow.set_tag("model_type", "final")
        mlflow.log_params(
            {
                "model": "logistic_regression",
                "seasons": str(SEASONS),
                "feature_window": WINDOW,
                "features": str(FEATURE_COLS),
                "n_samples": len(X),
            }
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="brasileirao-predictor",
        )


@flow(name="brasileirao-training-pipeline", log_prints=True)
def training_pipeline(
    seasons: list[int] = SEASONS,
    window: int = WINDOW,
):
    cv_tag = _cv_tag(CV_STRATEGY, MIN_TRAIN_ROUNDS, seasons)

    df = task_load_data(seasons)
    features = task_build_features(df, window)
    task_train_and_log(features, cv_tag)
    task_register_final_model(features)


if __name__ == "__main__":
    training_pipeline()
