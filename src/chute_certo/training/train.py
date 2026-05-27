import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chute_certo.training.cv import walk_forward_splits

FEATURE_COLS = [
    "home_form_points",
    "home_form_scored",
    "home_form_conceded",
    "away_form_points",
    "away_form_scored",
    "away_form_conceded",
]
TARGET = "result"


def make_models() -> dict:
    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "baseline_stratified": DummyClassifier(strategy="stratified", random_state=42),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
    }


def evaluate(df: pd.DataFrame, min_train_rounds: int = 10) -> pd.DataFrame:
    """
    Runs walk-forward CV for all models and returns per-fold metrics.
    Rows with NaN features (first round of first season) are dropped before splitting.
    """
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    records = []
    for train, val in walk_forward_splits(df, min_train_rounds):
        X_train = train[FEATURE_COLS].to_numpy()
        y_train = train[TARGET].to_numpy()
        X_val = val[FEATURE_COLS].to_numpy()
        y_val = val[TARGET].to_numpy()

        fold_season = val["season"].iloc[0]
        fold_round = val["round"].iloc[0]

        for name, model in make_models().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)

            records.append(
                {
                    "model": name,
                    "season": fold_season,
                    "round": fold_round,
                    "f1_macro": f1_score(y_val, y_pred, average="macro"),
                    "log_loss": log_loss(y_val, y_proba, labels=model.classes_),
                    "n_val": len(y_val),
                }
            )

    return pd.DataFrame(records)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregates per-fold metrics into mean ± std per model."""
    return (
        results.groupby("model")[["f1_macro", "log_loss"]]
        .agg(["mean", "std"])
        .round(4)
        .sort_values(("f1_macro", "mean"), ascending=False)
    )
