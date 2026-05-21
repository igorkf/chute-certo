"""Train all models with walk-forward CV and log results to MLflow."""

import mlflow

from chute_certo.features.build import WINDOW, build_features
from chute_certo.ingestion.processing import load_seasons
from chute_certo.training.train import FEATURE_COLS, evaluate, summarize

EXPERIMENT_NAME = "brasileirao-prediction"
SEASONS = [2022, 2023, 2024]
MIN_TRAIN_ROUNDS = 10


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_seasons(SEASONS)
    features = build_features(df, window=WINDOW)
    results = evaluate(features, min_train_rounds=MIN_TRAIN_ROUNDS)
    summary = summarize(results)

    for model_name, row in summary.iterrows():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(
                {
                    "model": model_name,
                    "seasons": str(SEASONS),
                    "feature_window": WINDOW,
                    "features": FEATURE_COLS,
                    "min_train_rounds": MIN_TRAIN_ROUNDS,
                    "n_folds": len(results[results["model"] == model_name]),
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


if __name__ == "__main__":
    main()
